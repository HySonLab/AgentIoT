from typing import Dict, Any, List, Callable, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from collections import Counter


def load_data(path_boiler_data: str = "dataset/Boiler_emulator_dataset.csv", path_turbine_data: str = "dataset/iiot-data-of-wind-turbine/"):
    print("\n📂 Loading Boiler Dataset...")
    data_boiler = pd.read_csv(path_boiler_data)
    print(f"✅ Boiler loaded: {data_boiler.shape}")

    print("\n📂 Loading Wind Turbine Dataset...")
    try:
        scada_df = pd.read_csv(f'{path_turbine_data}/scada_data.csv')
        scada_df['DateTime'] = pd.to_datetime(scada_df['DateTime'])

        fault_df = pd.read_csv(f'{path_turbine_data}/fault_data.csv')
        fault_df['DateTime'] = pd.to_datetime(fault_df['DateTime'])

        status_df = pd.read_csv(f'{path_turbine_data}/status_data.csv')
        status_df['Time'] = pd.to_datetime(status_df['Time'])
        status_df.rename(columns={'Time': 'DateTime'}, inplace=True)

        print(f"✅ Wind Turbine dataset loaded from {path_turbine_data}")
        data_source = "Real dataset"
    except FileNotFoundError as e:
        print(f"⚠️  Dataset not found: {e}")
        print("📝 To fix this: Download the wind turbine CSV files and place them in:")
        print(f"   {path_turbine_data}")
        print("   ├── scada_data.csv")
        print("   ├── fault_data.csv")
        print("   └── status_data.csv")
        print("\n💾 Generating synthetic wind turbine data as fallback...\n")

        n_samples = 1000
        n_features = 40
        np.random.seed(42)
        scada_data = {'DateTime': pd.date_range('2014-04-01', periods=n_samples, freq='10min')}
        for i in range(n_features):
            scada_data[f'Sensor_{i}'] = np.random.normal(50, 15, n_samples)

        scada_df = pd.DataFrame(scada_data)
        scada_df['DateTime'] = pd.to_datetime(scada_df['DateTime'])

        fault_types = ['gf', 'mf', 'ff', 'af', 'ef']
        fault_data = {
            'DateTime': scada_df['DateTime'].sample(n=200, random_state=42).sort_values().reset_index(drop=True),
            'Fault': np.random.choice(fault_types, size=200)
        }
        fault_df = pd.DataFrame(fault_data)

        status_df = pd.DataFrame({
            'DateTime': scada_df['DateTime'],
            'Status': np.random.choice(['OK', 'WARNING', 'ERROR'], size=n_samples, p=[0.7, 0.2, 0.1])
        })

        print("✅ Synthetic wind turbine data generated")
        data_source = "Synthetic (fallback)"

    df_turbine = scada_df.merge(fault_df[['DateTime', 'Fault']], on='DateTime', how='left')
    df_turbine['Fault'] = df_turbine['Fault'].replace(np.nan, 'NF')

    df_nf = df_turbine[df_turbine['Fault']=='NF']
    if len(df_nf) > 300:
        df_nf = df_nf.sample(n=300, random_state=42)

    df_f = df_turbine[df_turbine['Fault']!='NF']
    df_turbine = pd.concat((df_nf, df_f), axis=0).reset_index(drop=True)
    return data_boiler, df_turbine, data_source

def preprocess_data(data_boiler: pd.DataFrame, df_turbine: pd.DataFrame, data_source: str):
    irrelevant_cols = ['DateTime', 'WEC: ava. windspeed', 'WEC: ava. available P from wind',
                    'WEC: ava. available P technical reasons', 'WEC: ava. Available P force majeure reasons',
                    'WEC: ava. Available P force external reasons', 'WEC: max. windspeed', 'WEC: min. windspeed',
                    'WEC: Operating Hours', 'WEC: Production kWh', 'WEC: Production minutes']
    irrelevant_cols = [col for col in irrelevant_cols if col in df_turbine.columns]
    df_turbine_clean = df_turbine.drop(columns=irrelevant_cols, errors='ignore')

    X_turbine_full = df_turbine_clean.drop('Fault', axis=1)
    X_turbine_full['sensor_diff_0_1'] = X_turbine_full.iloc[:, 0] - X_turbine_full.iloc[:, 1]
    X_turbine_full['sensor_change_0'] = X_turbine_full.iloc[:, 0].diff().fillna(0)
    y_turbine_full = df_turbine_clean['Fault']

    le_turbine = LabelEncoder()
    y_turbine_encoded = le_turbine.fit_transform(y_turbine_full)

    X_train_turbine, X_test_turbine, y_train_turbine, y_test_turbine = train_test_split(
        X_turbine_full, y_turbine_encoded, test_size=0.2, random_state=42
    )

    scaler_wt = StandardScaler()
    X_train_turbine_scaled = pd.DataFrame(scaler_wt.fit_transform(X_train_turbine), columns=X_train_turbine.columns)
    X_test_turbine_scaled = pd.DataFrame(scaler_wt.transform(X_test_turbine), columns=X_test_turbine.columns)

    df_clean = data_boiler.copy()
    le_condition = LabelEncoder()
    df_clean['Condition'] = le_condition.fit_transform(df_clean['Condition'])
    le_class = LabelEncoder()
    df_clean['Class'] = le_class.fit_transform(df_clean['Class'])
    numeric_cols = ['Fuel_Mdot', 'Tair', 'Treturn', 'Tsupply', 'Water_Mdot']
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna()
    X_boiler = df_clean.drop('Class', axis=1)
    y_boiler = df_clean['Class']
    
    # Add domain-specific features BEFORE splitting
    #X_boiler['temp_diff'] = X_boiler['Tsupply'] - X_boiler['Treturn']
    #X_boiler['temp_change'] = X_boiler['Tsupply'].diff().fillna(0)

    # Add rolling statistics (3-point windows)
    #for col in ['Tsupply', 'Treturn', 'FuelMdot']:
    #   X_boiler[f'{col}_rolling_mean'] = X_boiler[col].rolling(window=3, min_periods=1).mean()
    #    X_boiler[f'{col}_rolling_std'] = X_boiler[col].rolling(window=3, min_periods=1).std().fillna(0)

    X_train_boiler_full, X_test_boiler, y_train_boiler_full, y_test_boiler = train_test_split(
        X_boiler, y_boiler, test_size=0.2, random_state=42, stratify=y_boiler
    )

    scaler_b = StandardScaler()
    X_train_boiler = pd.DataFrame(scaler_b.fit_transform(X_train_boiler_full), columns=X_train_boiler_full.columns)
    X_test_boiler = pd.DataFrame(scaler_b.transform(X_test_boiler), columns=X_test_boiler.columns)


    print(f"✅ Boiler - Train {X_train_boiler.shape}, Test {X_test_boiler.shape}")
    print(f"✅ Wind Turbine - Train {X_train_turbine_scaled.shape}, Test {X_test_turbine_scaled.shape}")

    y_test_boiler_anomaly = (y_test_boiler == 1).astype(int)
    y_test_turbine_anomaly = (y_test_turbine > 0).astype(int)

    y_test_boiler_rul = pd.Series(
        np.where(y_test_boiler == 1, np.random.randint(5, 30, len(y_test_boiler)), np.random.randint(30, 100, len(y_test_boiler)))
    )
    y_test_turbine_rul = pd.Series(
        np.where(y_test_turbine_anomaly == 1, np.random.randint(5, 25, len(y_test_turbine)), np.random.randint(25, 100, len(y_test_turbine)))
    )

    print(f"\n✅ Data loading complete")
    print(f"   Boiler anomaly class distribution: {y_test_boiler_anomaly.value_counts().to_dict()}")
    print(f"   Wind Turbine anomaly class distribution: {dict(Counter(y_test_turbine_anomaly))}")
    print(f"   Wind Turbine RUL threshold: 25 hours (RUL <= 25 = anomaly)")
    print(f"   Data source: {data_source}")

    Xtr_boiler, Xte_boiler = X_train_boiler, X_test_boiler
    yte_boiler_anom, yte_boiler_rul = y_test_boiler_anomaly.values, y_test_boiler_rul.values
    Xtr_turbine, Xte_turbine = X_train_turbine_scaled, X_test_turbine_scaled
    yte_turbine_anom, yte_turbine_rul = y_test_turbine_anomaly, y_test_turbine_rul.values
    return X_train_boiler, X_test_boiler, y_train_boiler_full, y_test_boiler, y_test_boiler_anomaly, y_test_boiler_rul, scaler_b, \
          X_train_turbine_scaled, X_test_turbine_scaled, y_train_turbine, y_test_turbine, y_test_turbine_anomaly, y_test_turbine_rul, scaler_wt
