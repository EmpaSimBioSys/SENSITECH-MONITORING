import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, balanced_accuracy_score
from preprocessing.feature_generation import FeatureGenerator
from datasets.data_merger import ShipmentDataMerger
from utils.config import ConfigLoader 
from typing import Dict, List, Tuple, Any

warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")

# Load augmented data from JSON files
def load_augmented_data(file_path: str) -> Dict[str, List[str]]:
    """
    Load augmented data from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping class names to lists of serial numbers.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Load and merge data from different CSV files
def load_and_merge_data(file_paths: List[str], class_names: List[str]) -> pd.DataFrame:
    """
    Load and merge data from multiple CSV files.

    Args:
        file_paths (List[str]): List of file paths.
        class_names (List[str]): Corresponding class names.

    Returns:
        pd.DataFrame: Merged data.
    """
    df_list = []
    for file_path, class_name in zip(file_paths, class_names):
        df = pd.read_csv(file_path)
        df['AnomalyClass'] = class_name
        df['Date / Time'] = pd.to_datetime(df['Date / Time'])
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Configuration and paths
config_loader = ConfigLoader()
base_dir = config_loader.get_value('paths', 'data')
model_dir = config_loader.get_value('paths', 'model')

# File paths and class names
file_paths = [
    os.path.join(base_dir, "classified/7_class/data_spikes.csv"),
    os.path.join(base_dir, "classified/7_class/data_cyclical_events.csv"),
    os.path.join(base_dir, "classified/7_class/data_not_precooled.csv"),
    os.path.join(base_dir, "classified/7_class/data_normal.csv"),
    os.path.join(base_dir, "classified/7_class/data_initial_ramp.csv"),
    os.path.join(base_dir, "classified/7_class/data_below_freezing.csv"),
    os.path.join(base_dir, "classified/7_class/data_extended_drift.csv")
]

class_names = [
    "Spike",
    "Cool Defrost",
    "Not Precooled",
    "Normal",
    "Initial Ramp",
    "Top Freezing",
    "Extended Drift"
]

# Label names and mappings
class_names_2 = ["Normal", "Anomaly"]
label_names = ["2-class", "7-class"]

# Load and merge data
merged_data = load_and_merge_data(file_paths, class_names)
grouped_data = merged_data.groupby("Serial no")
serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

# Load classified dataset
classified_df = pd.read_csv(os.path.join(base_dir, 'classified/2_class/classified_data_2_classes_relaxed.csv'))
classified_df['Date / Time'] = pd.to_datetime(classified_df['Date / Time'])
serial_to_2class_label = dict(zip(classified_df['Serial no'], classified_df['Label']))

# Prepare data for DTW
shipment_time_series = {name: group['Temperature'].values for name, group in grouped_data}
serial_numbers = list(shipment_time_series.keys())

# Load and process distance matrix
distance_matrix = np.load("/Users/divinefavourodion/results/dtw_distance_matrix.npy")
rows_with_inf = np.isinf(distance_matrix).any(axis=1)
filtered_distance_matrix = distance_matrix[~rows_with_inf]
large_number = np.nanmax(filtered_distance_matrix) * 10
filtered_distance_matrix[np.isnan(filtered_distance_matrix)] = large_number

# Convert distance matrix to affinity matrix
beta = 1
distance_matrix_std = np.nanstd(filtered_distance_matrix)
if distance_matrix_std == 0:
    distance_matrix_std = 1
affinity_matrix = np.exp(-beta * filtered_distance_matrix / distance_matrix_std)

# Load dataset features 
coop_path = os.path.join(base_dir, "all_data_combined_meta.csv")
bama_path = os.path.join(base_dir, "SWP_BAMA_Sensor_Shipment_berries.csv")

shipment_merger = ShipmentDataMerger(coop_path, bama_path, 'config.ini')
data = shipment_merger.resample_time_series()
data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())

feature_generator = FeatureGenerator(data, 'config.ini', coop_path, bama_path)
feature_generator.data = feature_generator.load_data(df=False)
all_features, _, _, stat_feature_names = feature_generator.extract_all_features()
all_features = feature_generator.normalize_features(all_features)

# Get the list of features in the dictionary
feature_keys = ['statistical','frequency_200', 'frequency_300', 'ae_latent_features', 'deep_latent_features', 'combined']

# Update serial_numbers
serial_numbers = [serial for serial in serial_numbers if serial in all_features]

# Select the labelling scheme
def select_labelling_scheme(labels: str) -> Dict[str, str]:
    """
    Select the labelling scheme based on the provided label type.

    Args:
        labels (str): Label type ("2-class" or "7-class").

    Returns:
        Dict[str, str]: Mapping from serial numbers to class labels.
    """
    if labels == "2-class":
        return serial_to_2class_label
    else:
        return serial_to_anomaly_class

# Perform spectral clustering
def perform_spectral_clustering(feature_space: Dict[str, np.ndarray], affinity_matrix: np.ndarray, n_clusters: int, with_affinity: bool = True) -> np.ndarray:
    """
    Perform spectral clustering on the given feature space or affinity matrix.

    Args:
        feature_space (Dict[str, np.ndarray]): Feature space for clustering.
        affinity_matrix (np.ndarray): Precomputed affinity matrix.
        n_clusters (int): Number of clusters.
        with_affinity (bool): Whether to use the affinity matrix.

    Returns:
        np.ndarray: Cluster labels.
    """
    features = np.array([feature_space[serial] for serial in serial_numbers if serial in feature_space])
    if with_affinity:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
        return spectral.fit_predict(affinity_matrix)
    else:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='rbf', assign_labels='kmeans')
        return spectral.fit_predict(features)

# Perform K-means clustering
def perform_kmeans_clustering(feature_space: Dict[str, np.ndarray], n_clusters: int) -> np.ndarray:
    """
    Perform K-means clustering on the given feature space.

    Args:
        feature_space (Dict[str, np.ndarray]): Feature space for clustering.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels.
    """
    features = np.array([feature_space[serial] for serial in serial_numbers if serial in feature_space])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Perform K-medoids clustering
def perform_kmedoids_clustering(feature_space: Dict[str, np.ndarray], n_clusters: int) -> np.ndarray:
    """
    Perform K-medoids clustering on the given feature space.

    Args:
        feature_space (Dict[str, np.ndarray]): Feature space for clustering.
        n_clusters (int): Number of clusters.

    Returns:
        np.ndarray: Cluster labels.
    """
    features = np.array([feature_space[serial] for serial in serial_numbers if serial in feature_space])
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    return kmedoids.fit_predict(features)


# Compute balanced accuracy score
def compute_balanced_accuracy(labels: np.ndarray, ground_truth_labels: Dict[str, str]) -> float:
    """
    Compute the balanced accuracy score.

    Args:
        labels (np.ndarray): Cluster labels.
        ground_truth_labels (Dict[str, str]): Ground truth labels.

    Returns:
        float: Balanced accuracy score.
    """
    label_mapping = {label: idx for idx, label in enumerate(np.unique(list(ground_truth_labels.values())))}

    common_serial_numbers = [serial for serial in serial_numbers if serial in ground_truth_labels and serial in serial_to_anomaly_class]

    ground_truth = [label_mapping[ground_truth_labels[serial]] for serial in common_serial_numbers]
    predicted = [labels[serial_numbers.index(serial)] for serial in common_serial_numbers]

    return balanced_accuracy_score(ground_truth, predicted)


# Function to compute per-class accuracy scores
def compute_per_class_accuracy(labels: np.ndarray, ground_truth_labels: Dict[str, str], anomaly_classes: List[str], n_clusters: int) -> Dict[str, Any]:
    """
    Compute per-class accuracy scores for each class.

    Args:
        labels (np.ndarray): Cluster labels.
        ground_truth_labels (Dict[str, str]): Ground truth labels.
        anomaly_classes (List[str]): List of anomaly classes.
        n_clusters (int): Number of clusters used.

    Returns:
        Dict[str, Any]: Per-class accuracy scores and corresponding cluster information.
    """
    per_class_accuracy = {cls: 0 for cls in anomaly_classes}
    best_clusters = {cls: -1 for cls in anomaly_classes}
    max_accuracy = {cls: 0 for cls in anomaly_classes}
    cluster_info = {cls: {'accuracy': 0, 'n_clusters': 0} for cls in anomaly_classes}

    for cluster_label in np.unique(labels):
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_serials = [serial_numbers[i] for i in cluster_indices]
        valid_cluster_serials = [serial for serial in cluster_serials if serial in ground_truth_labels]
        
        class_counts = {cls: 0 for cls in anomaly_classes}
        for serial in valid_cluster_serials:
            cls = ground_truth_labels[serial]
            class_counts[cls] += 1

        for cls in anomaly_classes:
            TP = class_counts[cls]
            FP = sum(class_counts.values()) - TP
            FN = list(ground_truth_labels.values()).count(cls) - TP
            TN = len(ground_truth_labels) - (TP + FP + FN)
            
            if TP + FN == 0:
                continue

            accuracy = TP / (TP + FN)
            if accuracy > max_accuracy[cls]:
                max_accuracy[cls] = accuracy
                best_clusters[cls] = cluster_label
                cluster_info[cls] = {'accuracy': accuracy, 'n_clusters': n_clusters}

    return max_accuracy, best_clusters, cluster_info

# Find best cluster assignment based on balanced accuracy
def find_best_cluster_assignment_balanced_accuracy(label_scheme: Dict[str, str], curr_class_names: List[str], selected_feature_space: Dict[str, np.ndarray], 
                                                   feature_space_name: str, max_clusters: int = 20) -> Tuple[float, Tuple[int, str], np.ndarray]:
    """
    Find the best cluster assignment based on balanced accuracy.

    Args:
        label_scheme (Dict[str, str]): Ground truth label scheme.
        curr_class_names (List[str]): List of current class names.
        selected_feature_space (Dict[str, np.ndarray]): Selected feature space for clustering.
        feature_space_name (str): Name of the feature space.
        max_clusters (int): Maximum number of clusters.

    Returns:
        Tuple[float, Tuple[int, str], np.ndarray]: Best balanced accuracy score, parameters, and labels.
    """
    best_balanced_accuracy = 0
    best_params = None
    best_labels = None
    results = []

    for n_clusters in range(2, max_clusters + 1):
        for clustering_method in ['K-means', 'K-mediods']:
            if clustering_method == 'K-means':
                labels = perform_kmeans_clustering(selected_feature_space, n_clusters)
            elif clustering_method == 'K-mediods':
                labels = perform_kmedoids_clustering(selected_feature_space, n_clusters)
            else:
                labels = perform_spectral_clustering(selected_feature_space, affinity_matrix, n_clusters, with_affinity=False)

            balanced_accuracy = compute_balanced_accuracy(labels, label_scheme)
            results.append({'n_clusters': n_clusters, 'clustering_method': clustering_method, 'balanced_accuracy': balanced_accuracy})
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
                best_params = (n_clusters, clustering_method)
                best_labels = labels
            print(f"Evaluated {clustering_method} with {n_clusters} clusters: Balanced Accuracy = {balanced_accuracy}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/clustering_results_7_class/best_cluster_assignment_balanced_accuracy_{feature_space_name}_relaxed.csv', index=False)

    return best_balanced_accuracy, best_params, best_labels

# Main function to execute clustering analysis
def main() -> None:
    """
    Main function to execute clustering analysis and evaluate different cluster assignments.
    """
    label_config = '7-class'
    label_scheme = select_labelling_scheme(label_config)
    curr_class_names = class_names if label_config == '7-class' else class_names_2

    best_results_balanced_accuracy = []
    per_class_accuracy_results = []

    for selected_features in feature_keys:
        print(f"Running search for optimal clusters using {selected_features}...")

        selected_feature_space = {serial: all_features[serial][selected_features] for serial in serial_numbers if serial in all_features}

        # Find best cluster assignment based on balanced accuracy
        best_balanced_accuracy, best_balanced_accuracy_params, best_labels = find_best_cluster_assignment_balanced_accuracy(label_scheme, curr_class_names, selected_feature_space, selected_features)
        best_results_balanced_accuracy.append({
            'feature_space': selected_features,
            'balanced_accuracy': best_balanced_accuracy,
            'n_clusters': best_balanced_accuracy_params[0],
            'clustering_method': best_balanced_accuracy_params[1],
        })

        # Compute per-class accuracy
        per_class_accuracy, best_clusters, cluster_info = compute_per_class_accuracy(best_labels, serial_to_anomaly_class, curr_class_names, best_balanced_accuracy_params[0])
        per_class_accuracy_results.append({
            'feature_space': selected_features,
            'per_class_accuracy': per_class_accuracy,
            'best_clusters': best_clusters,
            'cluster_info': cluster_info
        })

    # Save the best results to CSV files
    best_results_balanced_accuracy_df = pd.DataFrame(best_results_balanced_accuracy)
    best_results_balanced_accuracy_df.to_csv(f'results/clustering_results_7_class/best_results_balanced_accuracy_relaxed.csv', index=False)

    per_class_accuracy_results_df = pd.DataFrame(per_class_accuracy_results)
    per_class_accuracy_results_df.to_csv(f'results/clustering_results_7_class/per_class_accuracy.csv', index=False)

if __name__ == "__main__":
    main()