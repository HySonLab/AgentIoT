from typing import List, Dict, Any
from datetime import datetime
import queue
from config.config import MQTT_CONFIG
from abc import ABC, abstractmethod

class MQTTAgent(ABC):
    def __init__(self, name: str, broker: 'MessageBroker', subscribed_topics: List[str] = None):
        self.name = name
        self.broker = broker
        self.subscribed_topics = subscribed_topics or []
        self.message_queue = queue.Queue()
        self.execution_log = []
        for topic in self.subscribed_topics:
            self.broker.subscribe(topic, self.message_handler)

    def message_handler(self, topic: str, message: Dict[str, Any]):
        self.message_queue.put({'topic': topic, 'message': message})

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = None):
        qos = qos or MQTT_CONFIG['qos']
        self.broker.publish(topic, payload, qos=qos)
        self.execution_log.append({'timestamp': datetime.now().isoformat(), 'action': 'publish', 'topic': topic, 'agent': self.name})

    def get_latest_message(self, topic: str):
        return self.broker.get_latest(topic)

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass
