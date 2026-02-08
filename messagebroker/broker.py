from typing import Dict, Any, List, Callable
from collections import defaultdict, deque
from datetime import datetime
import threading

class MessageBroker:
    def __init__(self):
        self.topics = defaultdict(deque)
        self.subscribers = defaultdict(list)
        self.lock = threading.Lock()
        self.message_history = defaultdict(list)

    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 1):
        with self.lock:
            message = {'timestamp': datetime.now().isoformat(), 'topic': topic, 'payload': payload, 'qos': qos}
            self.topics[topic].append(message)
            self.message_history[topic].append(message)
            for callback in self.subscribers[topic]:
                try:
                    callback(topic, message)
                except Exception as e:
                    pass

    def subscribe(self, topic: str, callback):
        with self.lock:
            self.subscribers[topic].append(callback)

    def get_latest(self, topic: str, default=None):
        with self.lock:
            if self.topics[topic]:
                return self.topics[topic][-1]
            return default

    def get_all(self, topic: str):
        with self.lock:
            return list(self.topics[topic])

message_broker = MessageBroker()
