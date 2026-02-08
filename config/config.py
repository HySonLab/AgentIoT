MQTT_CONFIG = {
    'broker': 'localhost',
    'port': 1883,
    'keepalive': 60,
    'qos': 1,
    'tls_version': None,
    'use_tls': False,
    'topics': {
        'edge_stream_1': 'chunk/stream1',
        'edge_stream_2': 'chunk/stream2',
        'if_scores': 'scores/if',
        'transformer_scores': 'scores/transformer',
        'rul_predictions': 'predictions/rul',
        'anomalies': 'anomalies',
        'actions': 'actions',
        'local_feedback': 'feedback/edge',
        'operator_feedback': 'feedback/operator',
        'feedback_metrics': 'feedback/metrics',
        'policy_updates': 'policy/updates',
        'knowledge_graph': 'graph/sync',
        'monitoring_logs': 'monitor/logs',
        'evaluation_metrics': 'metrics/evaluation'
    }   
}
