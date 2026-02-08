from typing import Dict, Any
import pandas as pd
import numpy as np
from data_processing.processing import load_data, preprocess_data
from messagebroker.broker import message_broker
from config.config import MQTT_CONFIG

path_boiler_data = "dataset/Boiler_emulator_dataset.csv"
path_turbine_data = "dataset/ieee-phm-2012-data-challenge-dataset"
data_boiler, df_turbine, data_source = load_data(path_boiler_data=path_boiler_data, path_turbine_data=path_turbine_data)
# X_train_boiler, X_test_boiler, y_train_boiler_full, y_test_boiler, y_test_boiler_anomaly, y_test_boiler_rul, scaler_b, \
# X_train_turbine_scaled, X_test_turbine_scaled, y_train_turbine, y_test_turbine, y_test_turbine_anomaly, y_test_turbine_rul, scaler_wt \
#       = preprocess_data(data_boiler, df_turbine, data_source)

X_train_boiler, X_test_boiler, y_train_boiler_full, y_test_boiler, y_test_boiler_anomaly, y_test_boiler_rul, scaler_b, \
          X_train_turbine_scaled, X_test_turbine_scaled, y_train_turbine, y_test_turbine, y_test_turbine_anomaly, y_test_turbine_rul, scaler_wt \
              = preprocess_data(data_boiler, df_turbine, data_source)

CONFIG = {
    'datasets': {
        'boiler': {
            'X_train': X_train_boiler,
            'X_test': X_test_boiler,
            'y_train': y_train_boiler_full,
            'y_test': y_test_boiler,
            'y_test_anomaly': y_test_boiler_anomaly,
            'y_test_rul': y_test_boiler_rul,
            'scaler': scaler_b,
            'name': 'Boiler Emulator',
        },
        'wind_turbine': {
            'X_train': X_train_turbine_scaled,
            'X_test': X_test_turbine_scaled,
            'y_train': y_train_turbine,
            'y_test': y_test_turbine,
            'y_test_anomaly': y_test_turbine_anomaly,
            'y_test_rul': y_test_turbine_rul,
            'scaler': scaler_wt,
            'name': 'Wind Turbine',
        },
    },
    'mqtt_broker': message_broker,
}

from agents.semas_agent import SEMAS_COMPLETE

def run_semas_complete(dataset_key: str, dataset_config: Dict[str, Any], num_iterations: int = 3):
    print(f'\n📊 Dataset: {dataset_config["name"]}')
    print('=' * 80)
    semas = SEMAS_COMPLETE(broker=message_broker, background_data=dataset_config['X_train'].values if hasattr(dataset_config['X_train'], 'values') else dataset_config['X_train'])
    semas.train(dataset_config['X_train'], dataset_config.get('y_test_rul'))
    X_test = dataset_config['X_test']
    y_test_binary = dataset_config['y_test_anomaly'].values if hasattr(dataset_config['y_test_anomaly'], 'values') else dataset_config['y_test_anomaly']
    y_test_rul = dataset_config.get('y_test_rul')
    all_iterations = []
    for iteration in range(num_iterations):
        result = semas.execute(X_test, y_test_binary, y_test_rul, iteration=iteration + 1)
        if result['metrics']:
            print(f'\n   📈 Iteration {iteration + 1} Metrics:')
            for (metric, metric_values) in result['metrics'].items():
                print('-----------')
                print(f'metric: {metric}')
                print(f'      metric_values: {metric_values}')
        all_iterations.append(result)
    return {'dataset': dataset_key, 'iterations': all_iterations, 'semas_instance': semas}

print("\n" + "="*80)
print('🚀 RUNNING SEMAS-COMPLETE ON BOILER DATASET')
print("="*80)
semas_boiler = run_semas_complete('boiler', CONFIG['datasets']['boiler'], num_iterations=3)

print("\n" + "="*80)
print('🚀 RUNNING SEMAS-COMPLETE ON WIND TURBINE DATASET (FIXED LABELS)')
print("="*80)
semas_turbine = run_semas_complete('wind_turbine', CONFIG['datasets']['wind_turbine'], num_iterations=3)

print('\n' + '=' * 80)
print('📊 METRICS ANALYSIS')
print('=' * 80)

def analyze_results(semas_results, dataset_name):
    print(f'\n{dataset_name.upper()} DATASET')
    print('-' * 80)
    iterations = semas_results['iterations']
    all_metrics = []
    for i, iter_result in enumerate(iterations):
        if iter_result['metrics']['consensus']:
            metrics_copy = iter_result['metrics']['consensus'].copy()
            metrics_copy['iteration'] = i + 1
            all_metrics.append(metrics_copy)
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print(metrics_df.to_string(index=False))
        print(f'\nAverage F1: {metrics_df["f1"].mean():.4f}, Precision: {metrics_df["precision"].mean():.4f}, Recall: {metrics_df["recall"].mean():.4f}')

analyze_results(semas_boiler, 'boiler')
analyze_results(semas_turbine, 'wind_turbine')

# ============================================================================
# PERFORMANCE RESULTS COLLECTION FOR CROSS-PIPELINE COMPARISON
# ============================================================================
import pickle
from datetime import datetime

def collect_performance_results(semas_results, dataset_name, system_name='SEMAS'):
    """
    Collect comprehensive performance metrics for comparison across pipelines
    Enhanced version with policy evolution tracking (w1/w2 weights)
    """
    iterations_data = semas_results['iterations']
    semas_instance = semas_results.get('semas_instance')

    # Extract metrics from each iteration
    performance_log = {
        'system': system_name,
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'num_iterations': len(iterations_data),
        'iterations': []
    }

    for idx, iter_result in enumerate(iterations_data, 1):
        iteration_metrics = {
            'iteration': idx,
            'metrics': {},
            'latency_ms': iter_result.get('execution_time_ms', 0),
            'layer_times': {}
        }

        # Extract layer execution times if available
        if iter_result.get('metrics') and iter_result['metrics'].get('consensus'):
            consensus = iter_result['metrics']['consensus']
            iteration_metrics['layer_times'] = {
                'eval_time': consensus.get('eval_exec_time', 0) * 1000,  # Convert to ms
                'predict_time': consensus.get('predict_exec_time', 0) * 1000
            }

        # Extract consensus metrics
        if iter_result.get('metrics') and iter_result['metrics'].get('consensus'):
            consensus = iter_result['metrics']['consensus']
            iteration_metrics['metrics'] = {
                'accuracy': consensus.get('accuracy', 0),
                'precision': consensus.get('precision', 0),
                'recall': consensus.get('recall', 0),
                'f1': consensus.get('f1', 0),
                'roc_auc': consensus.get('roc_auc', 0),
                'mcc': consensus.get('mcc', 0),
                'rul_mae': consensus.get('rul_mae', 0),
                'rul_rmse': consensus.get('rul_rmse', 0)
            }

        performance_log['iterations'].append(iteration_metrics)

    # Calculate aggregated statistics
    if performance_log['iterations']:
        all_f1 = [it['metrics'].get('f1', 0) for it in performance_log['iterations']]
        all_precision = [it['metrics'].get('precision', 0) for it in performance_log['iterations']]
        all_recall = [it['metrics'].get('recall', 0) for it in performance_log['iterations']]
        all_latency = [it['latency_ms'] for it in performance_log['iterations']]

        performance_log['summary'] = {
            'avg_f1': np.mean(all_f1),
            'std_f1': np.std(all_f1),
            'avg_precision': np.mean(all_precision),
            'avg_recall': np.mean(all_recall),
            'avg_latency_ms': np.mean(all_latency),
            'f1_trajectory': all_f1,
            'f1_improvement': all_f1[-1] - all_f1[0] if len(all_f1) > 1 else 0
        }

        # Extract policy evolution (w1/w2 weights from CloudAgentD)
        if semas_instance and hasattr(semas_instance, 'cloud_d'):
            cloud_d = semas_instance.cloud_d

            # Get weight history from knowledge graph messages
            weight_history = []
            kg_messages = message_broker.get_all(MQTT_CONFIG['topics']['knowledge_graph'])

            for msg in kg_messages:
                if 'payload' in msg:
                    payload = msg['payload']
                    weight_history.append({
                        'w1': payload.get('w1', 0.4),
                        'w2': payload.get('w2', 0.6),
                        'data_quantity': payload.get('data_quantity', 0)
                    })

            if weight_history:
                performance_log['policy_evolution'] = {
                    'initial_w1': weight_history[0]['w1'],
                    'final_w1': weight_history[-1]['w1'],
                    'initial_w2': weight_history[0]['w2'],
                    'final_w2': weight_history[-1]['w2'],
                    'weight_changes': len(weight_history) - 1,
                    'weight_history': weight_history
                }

    return performance_log

# Collect performance results for both datasets
print('\n' + '='*80)
print('📦 COLLECTING SEMAS PERFORMANCE RESULTS')
print('='*80)

semas_performance_boiler = collect_performance_results(semas_boiler, 'boiler', 'SEMAS')
semas_performance_turbine = collect_performance_results(semas_turbine, 'wind_turbine', 'SEMAS')

print('\n✅ BOILER DATASET RESULTS:')
print(f'   Avg F1: {semas_performance_boiler["summary"]["avg_f1"]:.4f} ± {semas_performance_boiler["summary"]["std_f1"]:.4f}')
print(f'   Avg Precision: {semas_performance_boiler["summary"]["avg_precision"]:.4f}')
print(f'   Avg Recall: {semas_performance_boiler["summary"]["avg_recall"]:.4f}')
print(f'   Avg Latency: {semas_performance_boiler["summary"]["avg_latency_ms"]:.2f}ms')
print(f'   F1 Trajectory: {[f"{x:.4f}" for x in semas_performance_boiler["summary"]["f1_trajectory"]]}')
print(f'   F1 Improvement: {semas_performance_boiler["summary"]["f1_improvement"]:.4f}')

print('\n✅ WIND TURBINE DATASET RESULTS:')
print(f'   Avg F1: {semas_performance_turbine["summary"]["avg_f1"]:.4f} ± {semas_performance_turbine["summary"]["std_f1"]:.4f}')
print(f'   Avg Precision: {semas_performance_turbine["summary"]["avg_precision"]:.4f}')
print(f'   Avg Recall: {semas_performance_turbine["summary"]["avg_recall"]:.4f}')
print(f'   Avg Latency: {semas_performance_turbine["summary"]["avg_latency_ms"]:.2f}ms')
print(f'   F1 Trajectory: {[f"{x:.4f}" for x in semas_performance_turbine["summary"]["f1_trajectory"]]}')
print(f'   F1 Improvement: {semas_performance_turbine["summary"]["f1_improvement"]:.4f}')

# Save results to pickle files for cross-pipeline comparison
try:
    with open('semas_performance_boiler.pkl', 'wb') as f:
        pickle.dump(semas_performance_boiler, f)
    with open('semas_performance_turbine.pkl', 'wb') as f:
        pickle.dump(semas_performance_turbine, f)
    print('\n💾 Results saved to: semas_performance_boiler.pkl, semas_performance_turbine.pkl')
except Exception as e:
    print(f'\n⚠️  Could not save pickle files: {e}')

print('\n' + '='*80)
print('✅ PERFORMANCE COLLECTION COMPLETE')
print('='*80)
