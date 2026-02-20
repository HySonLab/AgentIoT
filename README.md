# INTRODUCTION
This repository implements SEMAS (Self-Enhanced Multi-Agent System), a multi-layer, multi-agent architecture for industrial anomaly detection in manufacturing environments.
The system is designed following a Fog–Edge–Cloud paradigm, enabling scalable, distributed intelligence for data processing, anomaly detection, and system coordination.

The project is organized into modular components, each responsible for a specific layer or function in the SEMAS architecture.
```
.
├── agents # Organize specific agents according to three layers fog, edge, and cloud
│   ├── cloud_agents.py
│   ├── edge_agents.py
│   ├── fog_agents.py
│   ├── mqtt_agent.py
│   └── semas_agent.py
├── config # Config message broker for agents
│   └── config.py
├── data_processing # Data processing pipeline
│   └── processing.py
├── dataset # Data input storage.
│   ├── Boiler_emulator_dataset.csv
│   └── ieee-phm-2012-data-challenge-dataset
├── messagebroker # Message broker for transfer message.
│   └── broker.py
├── pipeline.py # Initialize full pipeline
├── README.md
├── requirements.txt
```

Folder Description:
- agents/
Contains the implementation of all agents in the SEMAS architecture, organized by deployment layer:

    - Fog agents handle intermediate aggregation and contextual reasoning.
    - Edge agents perform real-time anomaly detection close to data sources.
    - Cloud agents manage global coordination, system optimization, and long-term knowledge enhancement.

- config/
Stores configuration files for the message broker and system-level parameters used by agents.

- data_processing/
Implements the data processing pipeline, including preprocessing, feature extraction, and preparation for anomaly detection models.

- dataset/
Holds raw input datasets used for experimental evaluation, including boiler fault data and industrial challenge datasets.

- messagebroker/
Provides the messaging infrastructure that enables communication between distributed agents using a broker-based architecture.

# INSTALLATION

Setup environment

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

# TRAINING

To run the full pipeline, execute the main script:
```
python pipeline.py
```
It will do following steps:
- Load and preprocess data of boiler and wind turbin from dataset folder
- Run training and evaluation anormaly detection based on Multi-Agent System

# DATASET
There are two datasets used in this project:
1. Boiler Dataset: a simulated industrial dataset that models the operation of a steam boiler system under both normal and faulty conditions.
- Source: https://ieee-dataport.org/open-access/simulated-boiler-fault-data

2. Turbin Dataset: this dataset contains SCADA (Supervisory Control and Data Acquisition) data collected from a real wind turbine.
- Source: https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset

You can download and organize into folder `dataset`:

```
dataset
├── Boiler_emulator_dataset.csv
└── ieee-phm-2012-data-challenge-dataset-master
    ├── Full_Test_Set
    ├── Learning_set
    └── Test_set
```

# RESULT

BOILER DATASET
| Iteration | Accuracy | Precision | Recall | F1-score | ROC-AUC | Eval Time (s) | Predict Time (s) | RUL MAE | RUL RMSE |
| --------: | -------: | --------: | -----: | -------: | ------: | ------------: | ---------------: | ------: | -------: |
|         1 |   0.5306 |    0.3929 | 0.8352 |   0.5344 |  0.6695 |        0.0139 |           0.6512 | 35.2992 |  42.5011 |
|         2 |   0.5154 |    0.3866 | 0.8557 |   0.5325 |  0.6695 |        0.0079 |           0.3224 | 35.2992 |  42.5011 |
|         3 |   0.4980 |    0.3786 | 0.8676 |   0.5272 |  0.6695 |        0.0080 |           0.4006 | 35.2992 |  42.5011 |

Average F1: 0.5314, Precision: 0.3860, Recall: 0.8528

WIND_TURBINE DATASET
| Iteration | Accuracy | Precision | Recall | F1-score | ROC-AUC | Eval Time (s) | Predict Time (s) | RUL MAE | RUL RMSE |
| --------: | -------: | --------: | -----: | -------: | ------: | ------------: | ---------------: | ------: | -------: |
|         1 |   0.5000 |    0.4898 | 1.0000 |   0.6575 |  0.5176 |        0.0017 |           0.3469 | 22.4011 |  28.3170 |
|         2 |   0.5000 |    0.4898 | 1.0000 |   0.6575 |  0.5180 |        0.0018 |           0.0385 | 22.4011 |  28.3170 |
|         3 |   0.5000 |    0.4898 | 1.0000 |   0.6575 |  0.5184 |        0.0018 |           0.0374 | 22.4011 |  28.3170 |

Average F1: 0.6575, Precision: 0.4898, Recall: 1.0000

# CITATION

```bibtex
@misc{saleh2026selfevolvingmultiagentnetworkindustrial,
      title={Self-Evolving Multi-Agent Network for Industrial IoT Predictive Maintenance}, 
      author={Rebin Saleh and Khanh Pham Dinh and Balázs Villányi and Truong-Son Hy},
      year={2026},
      eprint={2602.16738},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2602.16738}, 
}
```
