# Machine Learning-Based Anomaly Classification in Temperature Time Series of Fresh-produce (Berry)Cold-Chains with deep features

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Folder and Scripts Descritptions](#folder-and-script-descriptions)
- [Setup and Installation](#setup-and-installation)
- [Contributing](#contributing)
- [Contact Information](#contact-information)
- [License](#license)

## Project Overview

The project involves preprocessing temperature data, generating features, and applying various clustering and classification methods to identify anomalies. The implementation includes detailed analysis and comparison of different clustering techniques to evaluate their effectiveness in anomaly detection.

## Directory Structure

The following structure outlines the project's main components:

```plaintext
SENSITECH-MONITORING/
│
├── anomaly_detection/
│   ├── dl/
│   │   ├── config.yaml
│   │   ├── FC_Autoencoder.py
│   │   └── init.py
│   ├── hmm/
│   │   ├── hmm_interpret_states.py
│   │   ├── hmm_optimal_states.py
│   │   └── hmm_train.py
│   └── ml/
│       ├── classification/
        |   |── classifiers_2_label.py
        |   |── classifiers_multi_label.py 
│       └── clustering/
            |── best_cluster_assignment_spectral.py
        |   |── best_cluster_assignment.py 
├── assets/
│   ├── empa_logo.png
│   └── empa.png
├── data/
│   ├── augmented/
│   ├── classified/
|    |   |── 2_class/
|    |   |── 7_class/
│   ├── dtw_distance_matrix/
│   ├── features/
    |   |── frequency/
    |   |── latent/
│   ├── all_data_combined_meta.csv
│   ├── all_shipment_data_features.csv
│   ├── consolidated.csv
│   ├── shipments.csv
│   ├── SWP_BAMA_Sensor_Shipment_berries.csv
│   ├── SWP_COOP_Sensor_Shipment_berries.csv
│   └── SWP_COOPTrading_Sensor_Shipment.csv
├── datasets/
│   ├── __init__.py
│   ├── bama.py
│   ├── coop.py
│   └── data_merger.py
├── labelling/
│   ├── templates/
│   ├── app.py
│   ├── criteria_label.py
├── models/
│   ├── 2_class/
│   ├── 7_class/
│   ├── checkpoints/
│   └── hmm/
├── preprocessing/
│   ├── data_augmentation.py
│   ├── feature_generation.py
│   ├── init.py
│   ├── reshape.py
│   └── trim.py
├── results/
│   ├── clustering_results_2_class/
│   ├── clustering_results_7_class/
│   ├── clustering_results_spectral/
│   ├── hmm/
│   ├── hmm_interpretations/
│   ├── K-means/
│   ├── K-mediods/
│   ├── model_accuracy/
│   ├── model_parameters/
│   └── plots/
└── utils/
    ├── config.py
    └── init.py
├── config.ini
├── dtw.py
├── Procfile
├── README.md
└── requirements.txt
````

## Folder and Script Descriptions

### `anomaly_detection/`

This folder contains scripts and modules related to detecting anomalies in time-temperature signals. It is subdivided into three main categories:
- **`dl/`**: Contains deep learning-based anomaly detection scripts. This includes configuration files (`config.yaml`), the `FC_Autoencoder.py` script for autoencoder models, and an initialization script.
- **`hmm/`**: Contains scripts for Hidden Markov Models (HMM). Includes:
  - `hmm_interpret_states.py`: Script to interpret HMM states.
  - `hmm_optimal_states.py`: Script to determine the optimal number of states for HMM.
  - `hmm_train.py`: Script to train HMM models.
- **`ml/`**: Contains machine learning-based anomaly detection scripts, further categorized into:
  - **`classification/`**: Scripts for classification models handling different label types. Includes:
    - `classifiers_2_label.py`: Classification with two labels.
    - `classifiers_multi_label.py`: Classification with multiple labels.
  - **`clustering/`**: Scripts for clustering models, including:
    - `best_cluster_assignment.py`: Determines the best cluster assignment for general clustering.
    - `best_cluster_assignment_spectral.py`: Determines the best cluster assignment specifically for spectral clustering.

### `assets/`

This directory contains static files used in the project, such as images and logos. 

- `empa_logo.png`: Logo image for the project.
- `empa.png`: Additional project-related image.

### `data/`

This folder includes all datasets used for analysis and model training:
- **`augmented/`**: Contains data that has been augmented in json format.
- **`classified/`**: Data that has been classified into categories, including subfolders for different label types:
  - `2_class/`: Data classified into two categories.
  - `7_class/`: Data classified into seven categories.
- **`dtw_distance_matrix/`**: Contains data related to Dynamic Time Warping (DTW) distance matrices.
- **`features/`**: Features extracted from the raw data, including:
  - `frequency/`: Frequency domain features.
  - `latent/`: Latent features from CNN-AE and Timer.
- Various CSV files with shipment and feature data.

### `datasets/`

This directory contains scripts related to dataset handling and merging:
- `__init__.py`: Initialization script.
- `bama.py`: Script related to Bama dataset.
- `coop.py`: Script related to Coop dataset.
- `data_merger.py`: Script to merge different datasets.

### `labelling/`

This folder contains scripts and templates related to data labeling:
- **`templates/`**: Contains labeling templates.
- `app.py`: Main application script for labeling.
- `criteria_label.py`: Defines criteria for labeling data.

### `models/`

This directory stores trained models and checkpoints:
- **`2_class/`**: Models trained for two-class classification.
- **`7_class/`**: Models trained for seven-class classification.
- **`checkpoints/`**: Contains DL model checkpoints for saving and resuming training.
- **`hmm/`**: Models related to Hidden Markov Models.

### `preprocessing/`

Contains scripts for preprocessing data:
- `data_augmentation.py`: Script for augmenting data.
- `feature_generation.py`: Script for generating features from raw data.
- `init.py`: Initialization script.
- `reshape.py`: Script for reshaping time series data.
- `trim.py`: Script for trimming data to remove end ramp-up phase.

### `results/`

This folder includes results and outputs from the analysis:
- **`clustering_results_2_class/`**: Results of clustering for two-class data.
- **`clustering_results_7_class/`**: Results of clustering for seven-class data.
- **`clustering_results_spectral/`**: Results of spectral clustering.
- **`hmm/`**: Results related to Hidden Markov Models.
- **`hmm_interpretations/`**: Interpretations of HMM results.
- **`K-means/`**: Results of K-means clustering.
- **`K-mediods/`**: Results of K-medoids clustering.
- **`model_accuracy/`**: Model accuracy metrics and reports.
- **`model_parameters/`**: Parameters used in models.
- **`plots/`**: Various plots generated during analysis.

### `utils/`

Utility scripts and modules used across the project:
- `config.py`: Configuration settings.
- `init.py`: Initialization script.

### Other Files

- `config.ini`: Configuration file for the project.
- `dtw.py`: Script related to Dynamic Time Warping.
- `Procfile`: File for process management.
- `README.md`: This file.
- `requirements.txt`: List of Python dependencies required for the project.

## Setup and Installation

To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SENSITECH-MONITORING
   ```

2. **Create a Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Deep Learning Scripts (`dl/` Folder)

To train the autoencoder model, use the following command:

```bash
python anomaly_detection/dl/FC_Autoencoder.py --config <path_to_config_yaml> --learning_rate <learning_rate> --epochs <number_of_epochs> --batch_size <batch_size> --seq_len <sequence_length> --embedding_dim <embedding_dimension> --num_layers <number_of_layers> --input_dim <input_dimension>
```

Replace <path_to_config>, <learning_rate>, <num_epochs>, <batch_size>, <sequence_length>, <embedding_dim>, <num_layers>, and <input_dim> with your desired values. For example:

```bash
python train_autoencoder.py --config anomaly_detection/dl/config.yaml --learning_rate 0.001 --epochs 300 --batch_size 8 --seq_len 300 --embedding_dim 128 --num_layers 2 --input_dim 300
```

Running Other Scripts

For scripts that do not require arguments, you can run them with the following command:

python <script_name>.py

Replace <script_name> with the name of the script you wish to execute. For example:

```bash
python classifiers_2_label.py
```

## Contributing

Contributions are welcome! Please follow the standard GitHub flow for contributing to this project:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## Contact Information

For any inquiries or support, please reach out to:

- **Email:** [divineod9@gmail.com](mailto:divineod9@gmail.com)
- **Phone:** +49 1577 7149174

## License

This project is licensed under the MIT License.