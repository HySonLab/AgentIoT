from typing import Dict, List, Tuple, Any
from agents.mqtt_agent import MQTTAgent
from messagebroker.broker import message_broker
from config.config import MQTT_CONFIG
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


class EdgeSubagentA1(MQTTAgent):
    def __init__(self, name='A1_Temperature', broker=None, window_size=10):
        super().__init__(name, broker or message_broker)
        print("✅ Edge Layer Agent A1 defined")
        self.window_size = window_size
        self.publishing_topic = MQTT_CONFIG['topics']['edge_stream_1']

    def compute_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        rms = np.sqrt(np.mean(data**2, axis=0))
        kurtosis = np.mean((data - np.mean(data, axis=0))**4, axis=0) / (np.std(data, axis=0)**4 + 1e-8)
        skewness = np.mean((data - np.mean(data, axis=0))**3, axis=0) / (np.std(data, axis=0)**3 + 1e-8)
        x = np.arange(len(data))
        trend = np.array([np.polyfit(x, data[:, i], 1)[0] for i in range(data.shape[1])])
        fft_vals = np.fft.fft(data, axis=0)
        spectral_energy = np.abs(fft_vals).mean(axis=0)
        stats = {'mean': np.mean(data, axis=0), 'std': np.std(data, axis=0)}
        return {'rms': rms, 'kurtosis': kurtosis, 'skewness': skewness, 'trend': trend, 'spectral_energy': spectral_energy, 'stats': stats}

    def execute(self, data: np.ndarray, timestamp: float = None):
        timestamp = timestamp or datetime.now().timestamp()
        features = self.compute_features(data)
        payload = {
            'timestamp': timestamp,
            'agent': self.name,
            'window_size': self.window_size,
            'data_shape': data.shape,
            'features': {'rms': features['rms'].tolist(), 'kurtosis': features['kurtosis'].tolist()},
        }
        self.publish(self.publishing_topic, payload)
        return features

class EdgeSubagentA2(EdgeSubagentA1):
    def __init__(self, name='A2_Vibration', broker=None, window_size=10):
        super().__init__(name, broker, window_size)
        print("✅ Edge Layer Agent A2 defined")
        self.publishing_topic = MQTT_CONFIG['topics']['edge_stream_2']
