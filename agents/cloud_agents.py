from typing import Dict, List, Tuple, Any
from agents.mqtt_agent import MQTTAgent
from messagebroker.broker import message_broker
from config.config import MQTT_CONFIG
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix


class CloudAgentD(MQTTAgent):
    def __init__(self, name='D_EvolutionAgent_PPO', broker=None):
        super().__init__(name, broker or message_broker)
        print("✅ Cloud Agents D defined")
        self.policy_params = {
            'contamination_rate': 0.32,
            'threshold_delta_consensus': 0.0,  # Store DELTAS not absolute values
            'threshold_delta_b1': 0.0,
            'threshold_delta_b2': 0.0,
            'w1': 0.4,
            'w2': 0.6
        }
        self.data_quantity = 0
        self.learning_rates = {
            'w1': 0.1,
            'contamination': 0.02,
            'threshold': 0.05
        }

    def update_policy_ppo(self, metrics: Dict) -> Dict:
        f1_consensus = metrics.get('consensus').get('f1', 0.5)
        f1_b1_score = metrics.get('scores_b1_norm').get('f1', 0.2)
        f1_b2_score = metrics.get('scores_b2_norm').get('f1', 0.2)

        # Update w1/w2 weights
        delta_w1 = self.learning_rates['w1'] * (f1_b1_score - f1_b2_score)

        weights = self.get_latest_message(MQTT_CONFIG['topics']['knowledge_graph'])
        if weights:
            w1_current = weights.get('payload').get('w1', 0.4)
            contamination_current = weights.get('payload').get('contamination_rate', 0.32)
        else:
            w1_current = self.policy_params['w1']
            contamination_current = self.policy_params['contamination_rate']

        epsilon = 0.005
        w1_new = np.clip(w1_current + delta_w1, 0.2, 0.6) + epsilon
        w2_new = 1.0 - w1_new

        # Tune contamination based on F1 performance (increase if F1 < 0.6, decrease if F1 > 0.7)
        if f1_consensus < 0.6:
            delta_contamination = self.learning_rates['contamination']
        elif f1_consensus > 0.7:
            delta_contamination = -self.learning_rates['contamination']
        else:
            delta_contamination = 0.0
        contamination_new = np.clip(contamination_current + delta_contamination, 0.05, 0.5)

        # Adjust thresholds based on precision-recall balance
        # Return DELTA values that will be added to current thresholds
        precision_consensus = metrics.get('consensus').get('precision', 0.5)
        recall_consensus = metrics.get('consensus').get('recall', 0.5)

        # If precision > recall, lower threshold to increase recall (negative delta)
        # If recall > precision, raise threshold to increase precision (positive delta)
        if precision_consensus > recall_consensus + 0.1:
            threshold_delta = -self.learning_rates['threshold']
        elif recall_consensus > precision_consensus + 0.1:
            threshold_delta = self.learning_rates['threshold']
        else:
            threshold_delta = 0.0

        print(f'   🧠 Policy Update: f1_consensus={f1_consensus:.4f}, f1_b1={f1_b1_score:.4f}, f1_b2={f1_b2_score:.4f}')
        print(f'   🧠 Weights: w1: {w1_current:.4f} → {w1_new:.4f}')
        print(f'   🧠 Contamination: {contamination_current:.4f} → {contamination_new:.4f} (Δ={delta_contamination:.4f})')
        print(f'   🧠 Threshold deltas: consensus: {threshold_delta:.4f}, b1: {threshold_delta*0.5:.4f}, b2: {threshold_delta*0.5:.4f}')

        self.policy_params['w1'] = w1_new
        self.policy_params['w2'] = w2_new
        self.policy_params['contamination_rate'] = contamination_new
        self.policy_params['threshold_delta_consensus'] = threshold_delta
        self.policy_params['threshold_delta_b1'] = threshold_delta * 0.5
        self.policy_params['threshold_delta_b2'] = threshold_delta * 0.5
        self.data_quantity += 1

        return {
            'status': 'updated',
            'w1': self.policy_params['w1'],
            'w2': self.policy_params['w2'],
            'contamination_rate': self.policy_params['contamination_rate'],
            'threshold_delta_consensus': self.policy_params['threshold_delta_consensus'],
            'threshold_delta_b1': self.policy_params['threshold_delta_b1'],
            'threshold_delta_b2': self.policy_params['threshold_delta_b2']
        }

    def execute(self, metrics: Dict) -> Dict[str, Any]:
        policy_update = self.update_policy_ppo(metrics)
        knowledge_graph = {
            'w1': float(self.policy_params['w1']),
            'w2': float(self.policy_params['w2']),
            'contamination_rate': float(self.policy_params['contamination_rate']),
            'threshold_delta_consensus': float(self.policy_params['threshold_delta_consensus']),
            'threshold_delta_b1': float(self.policy_params['threshold_delta_b1']),
            'threshold_delta_b2': float(self.policy_params['threshold_delta_b2']),
            'data_quantity': self.data_quantity
        }
        payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, 'policy': self.policy_params.copy(), 'knowledge_graph': knowledge_graph}
        self.publish(MQTT_CONFIG['topics']['policy_updates'], payload)
        self.publish(MQTT_CONFIG['topics']['knowledge_graph'], knowledge_graph)
        return payload

class CloudAgentE(MQTTAgent):
    def __init__(self, name='E_MetaAgent_SHAP', broker=None, background_data: np.ndarray = None):
        super().__init__(name, broker or message_broker)
        print("✅ Cloud Agents E defined")
        self.audit_log = []

    def execute(self, predictions: np.ndarray, scores: np.ndarray, y_true: np.ndarray, iteration: int) -> Dict[str, Any]:
        pred_binary = (predictions > 0).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_true, pred_binary)),
            'precision': float(precision_score(y_true, pred_binary, zero_division=0)),
            'recall': float(recall_score(y_true, pred_binary, zero_division=0)),
            'f1': float(f1_score(y_true, pred_binary, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else 0.5),
            'mcc': float(matthews_corrcoef(y_true, pred_binary))
        }
        audit_report = {'iteration': iteration, 'metrics': metrics, 'status': 'PASS' if metrics['f1'] > 0.6 else 'REVIEW_NEEDED'}
        self.audit_log.append(audit_report)
        payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, 'iteration': iteration, 'audit_metrics': metrics}
        self.publish(MQTT_CONFIG['topics']['monitoring_logs'], payload)
        return payload
