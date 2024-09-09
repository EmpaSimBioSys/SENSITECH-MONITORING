import os
import json
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, balanced_accuracy_score
from utils.config import ConfigLoader
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")

# Load augmented data from JSON files
def load_augmented_data(file_path: str) -> Dict[str, List[str]]:
    with open(file_path, 'r') as f:
        return json.load(f)

# Load and merge data from different CSV files
def load_and_merge_data(file_paths: List[str], class_names: List[str]) -> pd.DataFrame:
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

class_names_2 = ["Normal", "Anomaly"]

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
distance_matrix = np.load("data/dtw_distance_matrix/dtw_distance_matrix.npy")

# Substitute inf values with a large number (4 times the maximum finite value in the matrix)
max_finite_value =  np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
large_number = max_finite_value * 2
distance_matrix[np.isinf(distance_matrix)] = large_number

# Handle NaN values if present, substituting them with the large number
distance_matrix[np.isnan(distance_matrix)] = large_number

# Convert distance matrix to affinity matrix
beta = 1
distance_matrix_std = np.nanstd(distance_matrix)
if distance_matrix_std == 0:
    distance_matrix_std = 1
affinity_matrix = np.exp(-beta * distance_matrix / distance_matrix_std)

# Select the labelling scheme
def select_labelling_scheme(labels: str) -> Dict[str, str]:
    if labels == "2-class":
        return serial_to_2class_label
    else:
        return serial_to_anomaly_class

# Perform spectral clustering
def perform_spectral_clustering(affinity_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    return spectral.fit_predict(affinity_matrix)

# Compute cluster F1 score
def compute_cluster_f1(labels: np.ndarray, ground_truth_labels: Dict[str, str], anomaly_classes: List[str]) -> Tuple[Dict[int, Dict[str, float]], Dict[int, str]]:
    cluster_f1 = {}
    best_class_labels = {}
    unique_labels = np.unique(labels)
    
    # Create a mapping from index to serial number for valid serials
    valid_serial_numbers = [serial for serial in serial_numbers if serial in ground_truth_labels]
    valid_indices = {i: serial_numbers[i] for i in range(len(serial_numbers)) if serial_numbers[i] in ground_truth_labels}

    for cluster_label in unique_labels:
        cluster_indices = np.where(labels == cluster_label)[0]
        cluster_serials = [valid_indices[i] for i in cluster_indices if i in valid_indices]
        
        if not cluster_serials:
            continue
        
        class_counts = {cls: 0 for cls in anomaly_classes}
        for serial in cluster_serials:
            cls = ground_truth_labels[serial]
            class_counts[cls] += 1

        for cls in anomaly_classes:
            TP = class_counts[cls]
            FP = len(cluster_serials) - TP
            FN = list(ground_truth_labels.values()).count(cls) - TP
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            cluster_f1.setdefault(cluster_label, {})[cls] = f1 * 100

        best_class_labels[cluster_label] = max(class_counts, key=class_counts.get)

    return cluster_f1, best_class_labels


# Compute balanced accuracy score
def compute_balanced_accuracy(labels: np.ndarray, ground_truth_labels: Dict[str, str]) -> float:
    label_mapping = {label: idx for idx, label in enumerate(np.unique(list(ground_truth_labels.values())))}

    # Ensure we only consider serial numbers present in ground_truth_labels
    valid_serial_numbers = [serial for serial in serial_numbers if serial in ground_truth_labels]

    ground_truth = [label_mapping[ground_truth_labels[serial]] for serial in valid_serial_numbers]
    predicted = [labels[serial_numbers.index(serial)] for serial in valid_serial_numbers]

    return balanced_accuracy_score(ground_truth, predicted)


# Compute per class accuracy
def compute_per_class_accuracy(labels: np.ndarray, ground_truth_labels: Dict[str, str], anomaly_classes: List[str]) -> Dict[str, Tuple[float, int]]:
    class_accuracy = {}
    unique_labels = np.unique(labels)
    
    # Create a mapping from index to serial number for valid serials
    valid_serial_numbers = [serial for serial in serial_numbers if serial in ground_truth_labels]
    valid_indices = {i: serial_numbers[i] for i in range(len(serial_numbers)) if serial_numbers[i] in ground_truth_labels}

    for anomaly_class in anomaly_classes:
        best_accuracy = 0
        best_cluster = -1
        
        for cluster_label in unique_labels:
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_serials = [valid_indices[i] for i in cluster_indices if i in valid_indices]
            
            tp = fp = fn = tn = 0
            for serial in cluster_serials:
                if serial in ground_truth_labels:
                    actual_class = ground_truth_labels[serial]
                    if actual_class == anomaly_class:
                        tp += 1
                    else:
                        fp += 1

            fn = list(ground_truth_labels.values()).count(anomaly_class) - tp
            tn = len(valid_serial_numbers) - (tp + fp + fn)

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_cluster = cluster_label
                
        class_accuracy[anomaly_class] = (best_accuracy, best_cluster)
    
    return class_accuracy


# Find best cluster assignment based on F1 score and balanced accuracy
def find_best_cluster_assignment(label_scheme: Dict[str, str], curr_class_names: List[str], affinity_matrix: np.ndarray, max_clusters: int = 30) -> Tuple[float, float, int, np.ndarray, Dict[int, str], Dict[str, Tuple[float, int, int]]]:
    best_f1 = 0
    best_balanced_accuracy = 0
    best_params = None
    best_labels = None
    best_class_labels = None
    best_class_accuracy = None
    results = []

    # Track the best accuracy and corresponding n_clusters for each class
    best_class_accuracy = {cls: (0, -1, -1) for cls in curr_class_names}

    for n_clusters in range(2, max_clusters + 1):
        labels = perform_spectral_clustering(affinity_matrix, n_clusters)
        
        # Compute F1 score
        cluster_f1, class_labels = compute_cluster_f1(labels, label_scheme, curr_class_names)
        avg_f1 = np.mean([max(cluster_f1[cluster].values()) for cluster in cluster_f1])
        
        # Compute balanced accuracy
        balanced_accuracy = compute_balanced_accuracy(labels, label_scheme)
        
        # Compute per class accuracy
        class_accuracy = compute_per_class_accuracy(labels, label_scheme, curr_class_names)
        
        results.append({
            'n_clusters': n_clusters,
            'f1_score': avg_f1,
            'balanced_accuracy': balanced_accuracy,
            'best_class_labels': class_labels,
            'class_accuracy': class_accuracy
        })
        
        if avg_f1 > best_f1 or balanced_accuracy > best_balanced_accuracy:
            best_f1 = avg_f1
            best_balanced_accuracy = balanced_accuracy
            best_params = n_clusters
            best_labels = labels
            best_class_labels = class_labels
        
        # Update the best accuracy and corresponding n_clusters for each class
        for cls, (accuracy, cluster) in class_accuracy.items():
            if accuracy > best_class_accuracy[cls][0]:
                best_class_accuracy[cls] = (accuracy, cluster, n_clusters)
        
        print(f"Evaluated Spectral Clustering with {n_clusters} clusters: F1 Score = {avg_f1}, Balanced Accuracy = {balanced_accuracy}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/best_cluster_assignment_spectral.csv', index=False)

    # Save per-class accuracy with the number of clusters that gave the best accuracy
    class_accuracy_df = pd.DataFrame.from_dict(best_class_accuracy, orient='index', columns=['Best Accuracy', 'Best Cluster', 'Best n_clusters'])
    class_accuracy_df.to_csv(f'results/clustering_results_spectral/per_class_accuracy.csv', index_label='Class')

    return best_f1, best_balanced_accuracy, best_params, best_labels, best_class_labels, best_class_accuracy


# Main function to execute clustering analysis
def main() -> None:
    label_config = '7-class'
    label_scheme = select_labelling_scheme(label_config)
    curr_class_names = class_names if label_config == '7-class' else class_names_2

    print(f"Running search for optimal clusters using Spectral Clustering...")

    # Find best cluster assignment
    best_f1, best_balanced_accuracy, best_params, best_labels, best_class_labels, best_class_accuracy = find_best_cluster_assignment(label_scheme, curr_class_names, affinity_matrix)

    # Print best results
    print(f"Best F1 Score: {best_f1} with {best_params} clusters")
    print(f"Best Balanced Accuracy: {best_balanced_accuracy} with {best_params} clusters")
    print("Best per-class accuracy:")
    for cls, (accuracy, cluster, num_clusters) in best_class_accuracy.items():
        print(f"Class {cls}: Accuracy = {accuracy}, Best Cluster = {cluster}, Best Cluster number = {num_clusters}")

if __name__ == "__main__":
    main()
