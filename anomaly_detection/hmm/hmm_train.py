import pandas as pd
import os
import json
from hmmlearn import hmm
import numpy as np
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
from collections import defaultdict
from datetime import datetime
from itertools import product
from typing import List, Dict, Any, Tuple

from datasets.coop import CoopData
from datasets.bama import BamaData
from datasets.data_merger import ShipmentDataMerger
from preprocessing.trim import TimeSeriesTrimmer

class NumpyEncoder(json.JSONEncoder):
    """Custom Json encoder class for numpy data types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def read_config(config_path: str) -> configparser.ConfigParser:
    """Read configuration file."""
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

def smooth_transmat(transmat: np.ndarray, smoothing_value: float = 1e-3) -> np.ndarray:
    """Smooth the transition matrix by adding a small value."""
    transmat += smoothing_value
    row_sums = transmat.sum(axis=1)
    return transmat / row_sums[:, np.newaxis]

def ensure_symmetric(covars: np.ndarray) -> np.ndarray:
    """Ensure the covariance matrix is symmetric."""
    return (covars + covars.T) / 2

def stabilize_covariance(covars: np.ndarray, min_covar: float) -> np.ndarray:
    """Stabilize the covariance matrix."""
    return covars + min_covar * np.eye(covars.shape[0])

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

    # Load anomaly classes
    anomaly_files = [
        "data/classified/7_class/data_spikes.csv",
        "data/classified/7_class/data_cyclical_events.csv",
        "data/classified/7_class/data_not_precooled.csv",
        "data/classified/7_class/data_normal.csv",
        "data/classified/7_class/data_initial_ramp.csv",
        "data/classified/7_class/data_below_freezing.csv",
        "data/classified/7_class/data_extended_drift.csv"
    ]

    anomaly_classes = [
        "Spike",
        "Cyclical events",
        "Excursion",
        "Not Precooled",
        "Normal",
        "Initial Ramp",
        "Below freezing",
        "Extended Drift"
    ]

    trimmed_shipments = {}
    for file, class_name in zip(anomaly_files, anomaly_classes):
        coop_data = CoopData(file, feature_list=data_features, dependent_var='Temperature')
        trimmer = TimeSeriesTrimmer(coop_data.data, temperature_column='Temperature')
        trimmed_shipments[class_name] = norm_median(trimmer.trim_time_series(), feature_col='Temperature')

    grouped_shipments = {class_name: df.groupby("Serial no") for class_name, df in trimmed_shipments.items()}

    # Split normal data into train and validation sets
    train_serials, val_serials = train_test_split(list(grouped_shipments["Normal"].groups.keys()), test_size=0.2, random_state=42)
    train_data = trimmed_shipments["Normal"][trimmed_shipments["Normal"]['Serial no'].isin(train_serials)]
    val_data = trimmed_shipments["Normal"][trimmed_shipments["Normal"]['Serial no'].isin(val_serials)]
    grouped_shipments["Train"] = train_data.groupby("Serial no")
    grouped_shipments["Val"] = val_data.groupby("Serial no")

    num_states = [2, 3, 8]

    # Prepare training data
    normal_shipments, normal_lengths = prepare_data_for_training(train_data, 'Temperature')
    X = np.concatenate(normal_shipments)

    # Define parameter grids for grid search
    param_grid_2_state = {
        'startprob': [np.full(2, 1 / 2)] + [np.random.dirichlet(np.ones(2), size=1)[0] for _ in range(10)],
        'transmat': [normalize_transmat(np.full((2, 2), 1 / 2))] + [normalize_transmat(np.random.rand(2, 2)) for _ in range(10)],
    }

    param_grid_3_state = {
        'startprob': [np.full(3, 1 / 3)] + [np.random.dirichlet(np.ones(3), size=1)[0] for _ in range(10)],
        'transmat': [normalize_transmat(np.full((3, 3), 1 / 3))] + [normalize_transmat(np.random.rand(3, 3)) for _ in range(10)],
    }

    param_grid_8_state = {
        'startprob': [np.full(8, 1 / 8)] + [np.random.dirichlet(np.ones(8), size=1)[0] for _ in range(10)],
        'transmat': [normalize_transmat(np.full((8, 8), 1 / 8))] + [normalize_transmat(np.random.rand(8, 8)) for _ in range(10)],
    }

    # Find the best HMM models
    best_model_2, best_score_2 = find_best_model(X, normal_lengths, num_states[0], param_grid_2_state, iterations=1000)
    best_model_3, best_score_3 = find_best_model(X, normal_lengths, num_states[1], param_grid_3_state, iterations=1000)
    best_model_8, best_score_8 = find_best_model(X, normal_lengths, num_states[2], param_grid_8_state, iterations=1000)

    # Save the best models
    if best_model_2 is not None:
        dump(best_model_2, f'models/hmm/best_hmm_model_{formatted_date}_2_state.joblib')

    if best_model_3 is not None:
        dump(best_model_3, f'models/hmm/best_hmm_model_{formatted_date}_3_state.joblib')

    if best_model_8 is not None:
        dump(best_model_8, f'models/hmm/best_hmm_model_{formatted_date}_8_state.joblib')

    # Load the saved HMM models
    best_model_2 = load(f"models/hmm/best_hmm_model_{formatted_date}_2_state.joblib")
    best_model_3 = load(f"models/hmm/best_hmm_model_{formatted_date}_3_state.joblib")
    best_model_8 = load(f"models/hmm/best_hmm_model_{formatted_date}_8_state.joblib")

    # Evaluate the models on the validation set
    val_shipments, val_lengths = prepare_data_for_training(val_data, 'Temperature')
    X_val = np.concatenate(val_shipments)

    # Predict and log likelihood for each model
    hidden_states_2 = best_model_2.predict(X_val, val_lengths)
    log_likelihood_2 = best_model_2.score(X_val)
    print(f"2-state HMM Validation Log Likelihood: {log_likelihood_2}")
    print(f"2-state HMM Hidden States: {hidden_states_2}")

    hidden_states_3 = best_model_3.predict(X_val)
    log_likelihood_3 = best_model_3.score(X_val)
    print(f"3-state HMM Validation Log Likelihood: {log_likelihood_3}")
    print(f"3-state HMM Hidden States: {hidden_states_3}")

    hidden_states_8 = best_model_8.predict(X_val)
    log_likelihood_8 = best_model_8.score(X_val)
    print(f"8-state HMM Validation Log Likelihood: {log_likelihood_8}")
    print(f"8-state HMM Hidden States: {hidden_states_8}")

    # Load manually labelled dataset
    merged_data = load_and_merge_data(anomaly_files, anomaly_classes)

    # Create a mapping from Serial no to AnomalyClass
    serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

    # Plot the HMM labels
    for serial_number, group in data.groupby('Serial no'):
        group_copy = group.copy()
        group_copy['PointValue'] = group_copy['Temperature']
        group['Datetime in Seconds'] = group["Date / Time"].apply(lambda ts: ts.value / 1e9)

        observations = group_copy[['PointValue']]
        x_axis = group["Date / Time"].apply(lambda ts: ts.value / 1e9)
        observations_array = observations.to_numpy()

        hidden_states = best_model_8.predict(observations_array)
        state_colors = get_color_map(8)
        
        observations['PointValue'] = reverse_scaling(observations['PointValue'].to_numpy(), mean_val=mean_value, median_val=median_value)

        plt.figure(figsize=(12, 6))
        plt.plot(group["Date / Time"], observations['PointValue'], label='PointValue', color='black')

        for idx, (x, y, state) in enumerate(zip(group["Date / Time"], observations['PointValue'], hidden_states)):
            plt.scatter(x, y, color=state_colors[state], s=50)

        centroid_positions = defaultdict(list)
        for (x, y, state) in zip(group["Datetime in Seconds"], observations['PointValue'], hidden_states):
            centroid_positions[state].append((x, y))

        for state, points in centroid_positions.items():
            x_centroid_sec = np.mean([p[0] for p in points])
            y_centroid = np.mean([p[1] for p in points])
            x_centroid = pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=x_centroid_sec)
            plt.scatter(x_centroid, y_centroid, color=state_colors[state], marker='X', s=100, label=f'Centroid State {state}')

        plt.legend()
        plt.xlabel('Datetime')
        plt.ylabel('PointValue')
        plt.title(f'Hidden States with Centroids for shipment {serial_number}')
        plt.savefig(f"results/plots/hmm/{formatted_date}/{serial_number}_segmentation.png", format="png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
