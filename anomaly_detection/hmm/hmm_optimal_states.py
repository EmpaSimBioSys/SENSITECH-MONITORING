import pandas as pd
import os
import json
from hmmlearn import hmm
import numpy as np
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.model_selection import train_test_split
from joblib import dump, load
from itertools import product
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple

from datasets.coop import CoopData
from datasets.bama import BamaData
from datasets.data_merger import ShipmentDataMerger
from preprocessing.trim import TimeSeriesTrimmer
from utils.config import ConfigLoader

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_config(config_path: str) -> configparser.ConfigParser:
    """Read and return the configuration file."""
    config = configparser.ConfigParser()
    config.read(config_path, encoding='utf-8')
    return config

def get_current_date_formatted() -> str:
    """Get the current date formatted as 'dd-mm-yy'."""
    return datetime.now().strftime("%d-%m-%y")

def create_directory(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def extract_hmm_parameters(model: hmm.GaussianHMM, observations_array: np.ndarray) -> Dict[str, Any]:
    """Extract parameters from an HMM model."""
    log_likelihood = model.score(observations_array)
    transmat = model.transmat_
    hidden_states = model.predict(observations_array)
    emissions = {'means': model.means_, 'covariances': model.covars_} if model.covariance_type else {}
    start_probabilities = model.startprob_

    return {
        "log_likelihood": log_likelihood,
        "transmat": transmat,
        "hidden_states": hidden_states,
        "emissions": emissions,
        "start_probabilities": start_probabilities
    }

def get_color_map(num_states: int) -> Dict[int, Tuple[float, float, float, float]]:
    """Get a color map for the given number of states."""
    norm = Normalize(vmin=0, vmax=num_states - 1)
    color_map = cm.get_cmap('viridis', num_states)
    return {state: color_map(norm(state)) for state in range(num_states)}

def normalize_transmat(transmat: np.ndarray) -> np.ndarray:
    """Normalize the transition matrix."""
    row_sums = transmat.sum(axis=1)
    zero_rows = row_sums == 0
    transmat[zero_rows, :] = 1.0
    row_sums = transmat.sum(axis=1)
    return transmat / row_sums[:, np.newaxis]

def reverse_scaling(x: np.ndarray, mean_val: float, median_val: float) -> np.ndarray:
    """Reverse the scaling transformation."""
    return (x * median_val) + mean_val

def norm_median(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Normalize the DataFrame using median normalization."""
    df[feature_col] = (df[feature_col] - np.mean(df[feature_col])) / np.median(df[feature_col])
    return df

def initialize_parameters(X: np.ndarray, n_states: int, param_combo: Tuple[np.ndarray, np.ndarray]) -> hmm.GaussianHMM:
    """Initialize HMM parameters."""
    startprob_init, transmat_init = param_combo
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", init_params='cm', n_iter=100, min_covar=1e-3)
    model.startprob_ = startprob_init
    model.transmat_ = transmat_init
    return model

def find_best_model(X: np.ndarray, lengths: List[int], n_states: int, param_grid: Dict[str, List[np.ndarray]], iterations: int = 100) -> Tuple[hmm.GaussianHMM, float]:
    """Find the best HMM model using grid search."""
    best_model = None
    best_score = float('-inf')
    param_combinations = list(product(*param_grid.values()))

    for i, combo in enumerate(param_combinations):
        model = initialize_parameters(X, n_states, combo)
        model.fit(X, lengths)
        score = model.score(X)
        if score > best_score:
            best_score = score
            best_model = model
        print(f"Iteration {i}: Transmat: {combo[1]}, Start_prob: {combo[0]}, Log Likelihood = {score}, Convergence = {model.monitor_.converged}")
    return best_model, best_score

def load_and_merge_data(file_paths: List[str], class_names: List[str]) -> pd.DataFrame:
    """Load and merge data from different CSV files."""
    df_list = []
    for file_path, class_name in zip(file_paths, class_names):
        df = pd.read_csv(file_path)
        df['AnomalyClass'] = class_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def prepare_data_for_training(df: pd.DataFrame, feature_col: str) -> Tuple[List[np.ndarray], List[int]]:
    """Prepare data for HMM training."""
    shipments = []
    lengths = []
    for serial_number, group in df.groupby("Serial no"):
        observations = group[[feature_col]].to_numpy()
        shipments.append(observations)
        lengths.append(len(observations))
    return shipments, lengths

def evaluate_hmm_models(train_groups: pd.core.groupby.DataFrameGroupBy, n_states_range: range) -> Tuple[hmm.GaussianHMM, hmm.GaussianHMM, np.ndarray, List[int], List[float], List[float]]:
    """Evaluate HMM models based on AIC and BIC criteria."""
    best_aic_model = None
    best_aic_score = float('inf')
    best_bic_model = None
    best_bic_score = float('inf')
    aic_scores = []
    bic_scores = []

    for n_states in n_states_range:
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, min_covar=1e-3)
        
        # Prepare training data
        X_train = []
        lengths_train = []
        for name, group in train_groups:
            temperature_data = group['Temperature'].values.reshape(-1, 1)
            X_train.extend(temperature_data)
            lengths_train.append(len(temperature_data))
        
        X_train = np.array(X_train)
        
        model.fit(X_train, lengths_train)
        
        # Calculate AIC and BIC
        aic_score = model.aic(X_train, lengths_train)
        bic_score = model.bic(X_train, lengths_train)
        
        print(f"n_states={n_states}, AIC: {aic_score}, BIC: {bic_score}")
        aic_scores.append(aic_score)
        bic_scores.append(bic_score)
        
        if aic_score < best_aic_score:
            best_aic_score = aic_score
            best_aic_model = model
        
        if bic_score < best_bic_score:
            best_bic_score = bic_score
            best_bic_model = model
    
    return best_aic_model, best_bic_model, X_train, lengths_train, aic_scores, bic_scores

def plot_hidden_states(model: hmm.GaussianHMM, X: np.ndarray, lengths: List[int], title: str, save_path: str) -> None:
    """Plot the hidden states for the HMM model."""
    hidden_states = model.predict(X, lengths)
    plt.figure(figsize=(15, 8))
    plt.plot(X, label='Observed Temperatures')
    plt.plot(hidden_states, label='Hidden States', linestyle='--')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Temperature / State')
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def main():
    
    # Get the current date for saving plots to folder with current date as name
    formatted_date = get_current_date_formatted()

    # Create directories
    current_directory = os.getcwd()
    hmm_result_directory = os.path.join(current_directory, "results", "plots", "hmm", formatted_date)
    create_directory(hmm_result_directory)

    # Load the full dataset
    shipment_merger = ShipmentDataMerger("data/all_data_combined_meta.csv", "data/SWP_BAMA_Sensor_Shipment_berries.csv", 'config.ini', trim=True)
    data = shipment_merger.resample_time_series()
    data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
    data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()

    # Median normalization on the data
    median_value = np.median(data['Temperature'])
    mean_value = np.mean(data['Temperature'])
    data = norm_median(data, feature_col='Temperature')

    data_features = ['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown', 'Relative Time']

    # Group data by 'Serial no'
    grouped_data = data.groupby('Serial no')

    # Get the total count of the groups
    total_groups = len(grouped_data)

    # Calculate the number of groups for train and test sets
    train_size = int(np.ceil(total_groups * 0.8))  # 80% train
    test_size = total_groups - train_size  # 20% test

    # Get a list of group names (Serial no)
    group_names = list(grouped_data.groups.keys())

    # Shuffle the group names
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(group_names)

    # Split the group names into train and test sets
    train_group_names = group_names[:train_size]
    test_group_names = group_names[train_size:]

    # Create train and test datasets based on the split group names
    train_data = data[data['Serial no'].isin(train_group_names)]
    test_data = data[data['Serial no'].isin(test_group_names)]

    # Group the train and test data by 'Serial no'
    train_groups = train_data.groupby('Serial no')
    test_groups = test_data.groupby('Serial no')

    # Define the range for the number of states
    n_states_range = range(1, 11)

    # Evaluate HMM models and find the best models based on AIC and BIC
    best_aic_model, best_bic_model, X_train, lengths_train, aic_scores, bic_scores = evaluate_hmm_models(train_groups, n_states_range)

    print(f"Best AIC model number of states: {best_aic_model.n_components}")
    print(f"Best BIC model number of states: {best_bic_model.n_components}")

    # Plot AIC and BIC scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_states_range, aic_scores, label='AIC', marker='o')
    plt.plot(n_states_range, bic_scores, label='BIC', marker='o')
    plt.xlabel('Number of States', fontsize=20, fontweight='bold')
    plt.ylabel('Score', fontsize=20, fontweight='bold')
    plt.title('AIC and BIC Scores for Different Number of States', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=18, fontweight='bold')
    plt.yticks(fontsize=18, fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(hmm_result_directory, 'aic_bic_scores.png'))
    plt.show()

    # Save the best AIC and BIC models
    dump(best_aic_model, 'models/best_aic_model.joblib')
    dump(best_bic_model, 'models/best_bic_model.joblib')

    # Extract and save the parameters of the best models
    best_aic_params = extract_hmm_parameters(best_aic_model, X_train)
    best_bic_params = extract_hmm_parameters(best_bic_model, X_train)

    # with open(os.path.join(hmm_result_directory, 'best_aic_model_params.json'), 'w') as f:
    #     json.dump(best_aic_params, f, cls=NumpyEncoder)

    # with open(os.path.join(hmm_result_directory, 'best_bic_model_params.json'), 'w') as f:
    #     json.dump(best_bic_params, f, cls=NumpyEncoder)

    # Plot the hidden states for the best AIC and BIC models
    plot_hidden_states(best_aic_model, X_train, lengths_train, 'Best AIC Model Hidden States', os.path.join(hmm_result_directory, 'best_aic_model_hidden_states.png'))
    plot_hidden_states(best_bic_model, X_train, lengths_train, 'Best BIC Model Hidden States', os.path.join(hmm_result_directory, 'best_bic_model_hidden_states.png'))

if __name__ == "__main__":
    main()
