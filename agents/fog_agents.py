from typing import Dict, List, Tuple, Any
from agents.mqtt_agent import MQTTAgent
from messagebroker.broker import message_broker
from config.config import MQTT_CONFIG
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import numpy as np
from collections import defaultdict, deque
from scipy.stats import ks_2samp
from sklearn.ensemble import IsolationForest
from tensorflow.keras.layers import Dense, RepeatVector, TimeDistributed, Input, Dropout, MultiHeadAttention, LayerNormalization, Conv1D, Add, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class FogOrchestrationAgentB(MQTTAgent):
    def __init__(self, name='B_Orchestrator', broker=None):
        subscribed_topics = [MQTT_CONFIG['topics']['edge_stream_1'], MQTT_CONFIG['topics']['edge_stream_2']]
        super().__init__(name, broker or message_broker, subscribed_topics)
        print("✅ Fog Orchestration Agent defined")
        self.score_history = deque(maxlen=100)

    def detect_drift(self, anomaly_scores: np.ndarray, window_size: int = 50) -> bool:
        if len(anomaly_scores) > 0:
            self.score_history.extend(anomaly_scores)
        if len(self.score_history) < window_size:
            return False
        recent = list(self.score_history)[-window_size:]
        historical = list(self.score_history)[:-window_size]
        if len(historical) < 10:
            return False
        statistic, p_value = ks_2samp(historical, recent)
        if p_value < 0.05:
            self.publish(MQTT_CONFIG['topics']['local_feedback'], {'drift_detected': True, 'p_value': float(p_value)})
            return True
        return False

    def execute(self, X: np.ndarray = None) -> Dict[str, Any]:
        msg_a1 = self.get_latest_message(MQTT_CONFIG['topics']['edge_stream_1'])
        msg_a2 = self.get_latest_message(MQTT_CONFIG['topics']['edge_stream_2'])
        if msg_a1 and msg_a2:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'a1_status': 'received',
                'a2_status': 'received'
            }
            self.publish(MQTT_CONFIG['topics']['monitoring_logs'], payload)
            return {'status': 'orchestration_complete', 'agents_active': 2}
        return {'status': 'waiting_for_data', 'agents_active': 0}


class FogSubagentB1(MQTTAgent):
    def __init__(self, name='B1_IsolationForest', broker=None, contamination=0.32):
        super().__init__(name, broker or message_broker)
        print("✅ Fog Detection Agent B1 defined")
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=200, max_samples=256)
        self.is_trained = False
        self.X_train = None

    def train(self, X_train):
        self.X_train = X_train  # Store training data for retraining
        self.model.fit(X_train)
        self.is_trained = True
        print(f'   ✅ B1 trained with contamination={self.contamination:.4f}')

    def update_contamination(self, new_contamination: float):
        """Update contamination parameter and retrain model"""
        if abs(new_contamination - self.contamination) > 0.01:  # Only retrain if significant change
            print(f'   🔄 B1 Retraining: contamination {self.contamination:.4f} → {new_contamination:.4f}')
            self.contamination = new_contamination
            self.model = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=200, max_samples=256)
            if self.X_train is not None:
                self.model.fit(self.X_train)
                self.is_trained = True

    def execute(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return None
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, 'predictions': predictions.tolist(), 'anomaly_scores': scores.tolist()}
        self.publish(MQTT_CONFIG['topics']['if_scores'], payload)
        return {'predictions': predictions, 'scores': scores}

class FogSubagentB2(MQTTAgent):
    def __init__(self, name='B2_TimeSeriesTransformer', broker=None):
        super().__init__(name, broker or message_broker)
        print("✅ Fog Detection Agent B2 defined")
        self.model = None
        self.is_trained = False

    def build_transformer_model(self, input_dim: int, seq_length: int = 10, d_model: int = 128):
        inputs = Input(shape=(seq_length, input_dim))
        x = Conv1D(d_model, kernel_size=1, padding='same', activation='relu')(inputs)
        attention = MultiHeadAttention(num_heads=4, key_dim=16)
        attn_output = attention(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
        self.model = model

    def train(self, X_train):
        input_dim = X_train.shape[1]
        self.build_transformer_model(input_dim)
        X_seq = self.create_sequences(X_train, seq_length=10)
        y_train = np.zeros(len(X_seq))
        self.model.fit(X_seq, y_train, epochs=25, batch_size=16, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=2)])
        self.is_trained = True

    def execute(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return None
        X_seq = self.create_sequences(X, seq_length=10)
        scores = self.model.predict(X_seq, verbose=0).flatten()
        scores = np.pad(scores, (9, len(X) - len(scores) - 9), mode='edge')
        payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, 'anomaly_scores': scores.tolist()}
        self.publish(MQTT_CONFIG['topics']['transformer_scores'], payload)
        return {'scores': scores}

    def create_sequences(self, data, seq_length=10):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences) if sequences else np.array([])


class FogSubagentB2_RUL(MQTTAgent):
    def __init__(self, name='B2_RUL_Predictor', broker=None):
        super().__init__(name, broker or message_broker)
        print("✅ RUL Prediction Agent B2 defined")
        self.model = None
        self.is_trained = False

    def build_rul_model(self, input_dim: int):
        from tensorflow.keras.layers import LSTM
        inputs = Input(shape=(10, input_dim))
        x = LSTM(64, return_sequences=True)(inputs)
        x = LSTM(32)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')
        self.model = model

    def train(self, X_train, RUL_train):
        input_dim = X_train.shape[1]
        self.build_rul_model(input_dim)
        X_seq = self.create_sequences(X_train, seq_length=10)
        if len(X_seq) == 0:
            self.is_trained = True
            return
        RUL_train_array = RUL_train.values if hasattr(RUL_train, 'values') else RUL_train
        if len(RUL_train_array) < len(X_train):
            RUL_train_array = np.pad(RUL_train_array, (0, len(X_train) - len(RUL_train_array)), mode='edge')
        RUL_seq = RUL_train_array[9:9+len(X_seq)]
        if len(RUL_seq) != len(X_seq):
            if len(RUL_seq) < len(X_seq):
                RUL_seq = np.pad(RUL_seq, (0, len(X_seq) - len(RUL_seq)), mode='edge')
            else:
                RUL_seq = RUL_seq[:len(X_seq)]
        self.model.fit(X_seq, RUL_seq, epochs=10, batch_size=32, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=2)])
        self.is_trained = True

    def execute(self, X: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return None
        X_seq = self.create_sequences(X, seq_length=10)
        if len(X_seq) == 0:
            return None
        rul_predictions = self.model.predict(X_seq, verbose=0).flatten()
        rul_predictions = np.pad(rul_predictions, (9, len(X) - len(rul_predictions) - 9), mode='edge')
        payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, 'rul_predictions': rul_predictions.tolist()}
        self.publish(MQTT_CONFIG['topics']['rul_predictions'], payload)
        return {'rul_predictions': rul_predictions}

    def create_sequences(self, data, seq_length=10):
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i+seq_length])
        return np.array(sequences) if sequences else np.array([])


class FogSubagentB3(MQTTAgent):
    def __init__(self, name='B3_ConsensusVoting', broker=None):
        super().__init__(name, broker or message_broker)
        print("✅ Fog Detection Agent B3 defined")

    def aggregate(self) -> Dict[str, Any]:
        msg_b1 = self.get_latest_message(MQTT_CONFIG['topics']['if_scores'])
        msg_b2 = self.get_latest_message(MQTT_CONFIG['topics']['transformer_scores'])
        weights = self.get_latest_message(MQTT_CONFIG['topics']['knowledge_graph'])
        print(f'weights {weights}')
        if weights:
            w1 = weights.get('payload').get('w1', 0.4)
        else:
            w1 = 0.4
        w2 = 1-w1
        if msg_b1 and msg_b2:
            scores_b1 = np.array(msg_b1['payload']['anomaly_scores'])
            scores_b2 = np.array(msg_b2['payload']['anomaly_scores'])
            scores_b1_norm = (scores_b1 - scores_b1.min()) / (scores_b1.max() - scores_b1.min() + 1e-8)
            scores_b2_norm = (scores_b2 - scores_b2.min()) / (scores_b2.max() - scores_b2.min() + 1e-8)
            msg_b1 = self.get_latest_message(MQTT_CONFIG['topics']['if_scores'])
            msg_b1 = self.get_latest_message(MQTT_CONFIG['topics']['if_scores'])
            consensus_scores = w1 * scores_b1_norm + w2 * scores_b2_norm
            print(f"w1: {w1}; w2: {w2}")
            print(f'top 5 consensus_scores: {consensus_scores[:5]}')
            # FIX: Return normalized scores, not raw scores
            return {'consensus_scores': consensus_scores, 'scores_b1_norm': scores_b1_norm, 'scores_b2_norm': scores_b2_norm}
        return None

    def execute(self) -> Dict[str, Any]:
        result = self.aggregate()
        if result:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'agent': self.name,
                'consensus_scores': result['consensus_scores'].tolist(),
                'scores_b1_norm': result['scores_b1_norm'].tolist(),
                'scores_b2_norm': result['scores_b2_norm'].tolist()
            }
            self.publish(MQTT_CONFIG['topics']['anomalies'], payload)
            return result
        return None


class FogSubagentC(MQTTAgent):
    def __init__(self, name='C_ResponseGenerator', broker=None, use_mock=True):
        super().__init__(name, broker or message_broker)
        print("✅ Fog Detection Agent C defined")
        self.use_mock = use_mock

    def generate_explanation_llm(self, anomaly_scores: np.ndarray) -> Dict[str, Any]:
        severity = float(np.max(anomaly_scores))
        if severity > 0.7:
            return {'explanation': 'Critical bearing wear detected', 'action': 'IMMEDIATE_SHUTDOWN', 'priority': 'HIGH', 'downtime': 8}
        elif severity > 0.4:
            return {'explanation': 'Moderate anomaly', 'action': 'SCHEDULE_INSPECTION', 'priority': 'MEDIUM', 'downtime': 4}
        else:
            return {'explanation': 'Minor drift', 'action': 'CONTINUE_MONITORING', 'priority': 'LOW', 'downtime': 0}

    def execute(self) -> Dict[str, Any]:
        anomaly_msg = self.get_latest_message(MQTT_CONFIG['topics']['anomalies'])
        if anomaly_msg:
            scores = np.array(anomaly_msg['payload']['consensus_scores'])
            explanation_dict = self.generate_explanation_llm(scores)
            payload = {'timestamp': datetime.now().isoformat(), 'agent': self.name, **explanation_dict}
            self.publish(MQTT_CONFIG['topics']['actions'], payload)
            return explanation_dict
        return None
