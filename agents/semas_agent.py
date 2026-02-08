from typing import Dict, List, Tuple, Any
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error, mean_squared_error
from agents.cloud_agents import CloudAgentD, CloudAgentE
from agents.edge_agents import EdgeSubagentA1, EdgeSubagentA2
from agents.fog_agents import FogOrchestrationAgentB, FogSubagentB1, FogSubagentB2, FogSubagentB3, FogSubagentB2_RUL, FogSubagentC
from agents.mqtt_agent import MQTTAgent
from messagebroker.broker import message_broker, MessageBroker
from config.config import MQTT_CONFIG


class SEMAS_COMPLETE(MQTTAgent):
    def __init__(self, name='SEMAS_COMPLETE', broker: MessageBroker = None, background_data: np.ndarray = None):
        super().__init__(name, broker or message_broker)
        self.broker = broker or message_broker
        self.name = name
        self.edge_a1 = EdgeSubagentA1(broker=self.broker)
        self.edge_a2 = EdgeSubagentA2(broker=self.broker)
        self.fog_b = FogOrchestrationAgentB(broker=self.broker)
        self.fog_b1 = FogSubagentB1(broker=self.broker)
        self.fog_b2 = FogSubagentB2(broker=self.broker)
        self.fog_b2_rul = FogSubagentB2_RUL(broker=self.broker)
        self.fog_b3 = FogSubagentB3(broker=self.broker)
        self.fog_c = FogSubagentC(broker=self.broker, use_mock=True)
        self.cloud_d = CloudAgentD(broker=self.broker)
        self.cloud_e = CloudAgentE(broker=self.broker, background_data=background_data)
        self.execution_history = []
        # Store separate thresholds for consensus, B1, and B2
        self.threshold_consensus = None
        self.threshold_b1 = None
        self.threshold_b2 = None

    def train(self, X_train: np.ndarray, RUL_train: np.ndarray = None):
        print(f"Training {self.name} models...")
        X_train_data = X_train.values if hasattr(X_train, 'values') else X_train
        self.fog_b1.train(X_train_data)
        self.fog_b2.train(X_train_data)
        if RUL_train is not None:
            self.fog_b2_rul.train(X_train_data, RUL_train)
            print("✅ RUL model trained")
        print("✅ Training complete")

    def create_predictions(self, X_test_data: np.ndarray, scores: np.ndarray, y_test_true: np.ndarray = None,
                         is_first_iteration: bool = False, score_type: str = 'consensus',
                         policy_threshold: float = None, adaptive_mode: bool = True):
        rul_result = self.fog_b2_rul.execute(X_test_data)
        scores = np.array(scores)

        # Select the appropriate threshold based on score_type
        if score_type == 'consensus':
            threshold_attr = 'threshold_consensus'
        elif score_type == 'b1':
            threshold_attr = 'threshold_b1'
        elif score_type == 'b2':
            threshold_attr = 'threshold_b2'
        else:
            threshold_attr = 'threshold_consensus'

        current_threshold = getattr(self, threshold_attr)

        # In adaptive mode, use policy-provided threshold if available
        if adaptive_mode and policy_threshold is not None:
            new_threshold = policy_threshold
            setattr(self, threshold_attr, new_threshold)
            current_threshold = new_threshold
        # Otherwise, calculate threshold on first iteration or if None
        elif is_first_iteration or current_threshold is None:
            if y_test_true is not None:
                from sklearn.metrics import precision_recall_curve
                prec, rec, thresholds = precision_recall_curve(y_test_true, scores)
                f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
                new_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else np.percentile(scores, 75)
                setattr(self, threshold_attr, new_threshold)
                print(f'   🎯 [{score_type.upper()}] Threshold calibrated: {new_threshold:.4f}')
            else:
                new_threshold = np.percentile(scores, 75)
                setattr(self, threshold_attr, new_threshold)
                print(f'   🎯 [{score_type.upper()}] Threshold set to 75th percentile: {new_threshold:.4f}')

            current_threshold = new_threshold

        # Use the stored threshold for predictions
        adjusted_predictions = np.where(scores >= current_threshold, 1, 0)
        return adjusted_predictions, rul_result, scores


    def evaluation_metrics(self, y_test_true: np.ndarray, adjusted_predictions: np.ndarray, scores: np.ndarray):
        metrics = {
            'accuracy': float(accuracy_score(y_test_true, adjusted_predictions)),
            'precision': float(precision_score(y_test_true, adjusted_predictions, zero_division=0)),
            'recall': float(recall_score(y_test_true, adjusted_predictions, zero_division=0)),
            'f1': float(f1_score(y_test_true, adjusted_predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test_true, scores) if len(np.unique(y_test_true)) > 1 else 0.5),
        }
        return metrics

    def execute(self, X_test: np.ndarray, y_test_true: np.ndarray, y_test_rul: np.ndarray = None, iteration: int = 1, adaptive_mode: bool = True) -> Dict[str, Any]:
        X_test_data = X_test.values if hasattr(X_test, 'values') else X_test
        exec_start = time.time()
        print(f'\nSEMAS Iteration {iteration}')

        # Get policy updates from knowledge graph (for iterations > 1)
        policy_thresholds = {}
        policy_contamination = None
        if iteration > 1 and adaptive_mode:
            kg_msg = self.broker.get_latest(MQTT_CONFIG['topics']['knowledge_graph'])
            if kg_msg:
                policy_thresholds['consensus'] = kg_msg.get('payload', {}).get('threshold_consensus')
                policy_thresholds['b1'] = kg_msg.get('payload', {}).get('threshold_b1')
                policy_thresholds['b2'] = kg_msg.get('payload', {}).get('threshold_b2')
                policy_contamination = kg_msg.get('payload', {}).get('contamination_rate')

                # Apply contamination update to B1 model
                if policy_contamination is not None:
                    self.fog_b1.update_contamination(policy_contamination)

        self.edge_a1.execute(X_test_data)
        self.edge_a2.execute(X_test_data)
        b1_result = self.fog_b1.execute(X_test_data)
        b2_result = self.fog_b2.execute(X_test_data)
        b3_result = self.fog_b3.execute()
        c_result = self.fog_c.execute()
        consensus_scores = None
        scores_b1_norm = None
        scores_b2_norm = None
        if b3_result:
            consensus_scores = np.array(b3_result['consensus_scores'])
            scores_b1_norm = np.array(b3_result['scores_b1_norm'])
            scores_b2_norm = np.array(b3_result['scores_b2_norm'])
            predict_exec_start_time = time.time()
            adjusted_predictions, rul_result, consensus_array = self.create_predictions(
                X_test_data, consensus_scores, y_test_true,
                is_first_iteration=(iteration == 1), score_type='consensus',
                policy_threshold=policy_thresholds.get('consensus'), adaptive_mode=adaptive_mode)
            predict_exec_end_time = time.time()
            predict_exec_time = predict_exec_end_time - predict_exec_start_time
            adjusted_predictions_b1, rul_result_b1, scores_b1_norm = self.create_predictions(
                X_test_data, scores_b1_norm, y_test_true,
                is_first_iteration=(iteration == 1), score_type='b1',
                policy_threshold=policy_thresholds.get('b1'), adaptive_mode=adaptive_mode)
            adjusted_predictions_b2, rul_result_b2, scores_b2_norm = self.create_predictions(
                X_test_data, scores_b2_norm, y_test_true,
                is_first_iteration=(iteration == 1), score_type='b2',
                policy_threshold=policy_thresholds.get('b2'), adaptive_mode=adaptive_mode)
        else:
            adjusted_predictions, rul_result = None, None
            adjusted_predictions_b1, rul_result_b1 = None, None
            adjusted_predictions_b2, rul_result_b2 = None, None
        if b3_result:
            self.fog_b.detect_drift(np.array(b3_result['consensus_scores']))
        metrics = {}
        if adjusted_predictions is not None:
            eval_exec_start_time = time.time()
            metrics_consensus = self.evaluation_metrics(y_test_true, adjusted_predictions, consensus_array)
            eval_exec_end_time = time.time()
            eval_exec_time = eval_exec_end_time - eval_exec_start_time
            metrics_consensus['eval_exec_time'] = eval_exec_time
            metrics_consensus['predict_exec_time'] = predict_exec_time
            metrics_b1 = self.evaluation_metrics(y_test_true, adjusted_predictions_b1, scores_b1_norm)
            metrics_b2 = self.evaluation_metrics(y_test_true, adjusted_predictions_b2, scores_b2_norm)
            if rul_result is not None and y_test_rul is not None:
                rul_preds = np.array(rul_result['rul_predictions'])
                y_test_rul_array = y_test_rul.values if hasattr(y_test_rul, 'values') else y_test_rul
                metrics_consensus['rul_mae'] = float(mean_absolute_error(y_test_rul_array[:len(rul_preds)], rul_preds))
                metrics_consensus['rul_rmse'] = float(np.sqrt(mean_squared_error(y_test_rul_array[:len(rul_preds)], rul_preds)))
            metrics['consensus'] = metrics_consensus
            metrics['scores_b1_norm'] = metrics_b1
            metrics['scores_b2_norm'] = metrics_b2
            d_result = self.cloud_d.execute(metrics)
            e_result = self.cloud_e.execute(adjusted_predictions, consensus_array, y_test_true, iteration)

        exec_end = time.time()
        execution_time_ms = (exec_end - exec_start) * 1000
        metrics['execution_time_ms'] = execution_time_ms

        # Store policy state and thresholds in execution result
        execution_result = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'metrics_b1': metrics_b1,
            'metrics_b2': metrics_b2,
            'agent': self.name,
            'execution_time_ms': execution_time_ms,
            'policy_state': {
                'contamination': policy_contamination if policy_contamination else self.fog_b1.contamination,
                'threshold_consensus': self.threshold_consensus,
                'threshold_b1': self.threshold_b1,
                'threshold_b2': self.threshold_b2
            }
        }
        self.publish(MQTT_CONFIG['topics']['monitoring_logs'], execution_result)

        self.execution_history.append(execution_result)
        return execution_result

print("✅ SEMAS COMPLETE System defined")
