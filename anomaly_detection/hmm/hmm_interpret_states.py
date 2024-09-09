import os
import json
import numpy as np
import pandas as pd
from itertools import permutations
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

from preprocessing.feature_generation import FeatureGenerator
from datasets.data_merger import ShipmentDataMerger

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_and_merge_data(file_paths: List[str], class_names: List[str]) -> pd.DataFrame:
    """
    Load and merge data from different CSV files.

    Args:
    - file_paths (List[str]): List of file paths to the CSV files.
    - class_names (List[str]): List of class names corresponding to each file.

    Returns:
    - pd.DataFrame: Merged DataFrame with an additional 'AnomalyClass' column.
    """
    df_list = []
    for file_path, class_name in zip(file_paths, class_names):
        df = pd.read_csv(file_path)
        df['AnomalyClass'] = class_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

def norm_median(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """
    Normalize a DataFrame column using median normalization.

    Args:
    - df (pd.DataFrame): Input DataFrame.
    - feature_col (str): The name of the column to normalize.

    Returns:
    - pd.DataFrame: DataFrame with the normalized column.
    """
    df[feature_col] = (df[feature_col] - np.mean(df[feature_col])) / np.median(df[feature_col])
    return df

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], title: str) -> None:
    """
    Plot a confusion matrix with percentages.

    Args:
    - cm (np.ndarray): Confusion matrix.
    - class_names (List[str]): List of class names.
    - title (str): Title for the plot.
    """
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14, "weight": "bold", "color": "black"})
    for t in plt.gca().texts:
        t.set_text(t.get_text() + " %")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.savefig(f'results/plots/confusion_matrix/{title}_multi_class.png', dpi=300)
    plt.close()

def calculate_metrics(true_labels: List[str], pred_labels: List[str]) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """
    Calculate performance metrics for classification.

    Args:
    - true_labels (List[str]): True labels.
    - pred_labels (List[str]): Predicted labels.

    Returns:
    - Tuple containing accuracy, precision, recall, f1 score, balanced accuracy, confusion matrix, and per-class accuracy.
    """
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=["Normal", "Anomaly"])
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    return accuracy, precision, recall, f1, balanced_accuracy, cm, per_class_accuracy


def evaluate_permutations(true_labels: List[str], hidden_states: Dict[str, np.ndarray], state_permutations: List[Dict[int, str]], model: str) -> Tuple[Tuple[float, float, float, float, float, np.ndarray, np.ndarray], np.ndarray, Dict[int, str], np.ndarray]:
    """
    Evaluate different permutations of state labels.

    Args:
    - true_labels (List[str]): True labels.
    - hidden_states (Dict[str, np.ndarray]): Hidden states for each serial number.
    - state_permutations (List[Dict[int, str]]): List of state permutations to evaluate.
    - model (str): Model name.

    Returns:
    - Tuple containing the best metrics, best confusion matrix, best permutation, and best per-class accuracy.
    """
    best_metrics = None
    best_cm = None
    best_perm = None
    best_per_class_acc = None

    for perm in state_permutations:
        pred_labels = []
        for serial_no in hidden_states:
            mapped_states = [perm[state] for state in hidden_states[serial_no]]
            label = max(set(mapped_states), key=mapped_states.count)  # Majority vote
            pred_labels.append(label)
        
        metrics = calculate_metrics(true_labels, pred_labels)
        
        if best_metrics is None or metrics[4] > best_metrics[4]:  # Use balanced accuracy as the primary criterion
            best_metrics = metrics[:6]  # Exclude per_class_accuracy from best_metrics
            best_cm = metrics[5]
            best_perm = perm
            best_per_class_acc = metrics[6]
            
    return best_metrics, best_cm, best_perm, best_per_class_acc

def main():
    # File paths and class names
    file_paths = [
        "data/classified/7_class/data_spikes.csv",
        "data/classified/7_class/data_cyclical_events.csv",
        "data/classified/7_class/data_not_precooled.csv",
        "data/classified/7_class/data_normal.csv",
        "data/classified/7_class/data_initial_ramp.csv",
        "data/classified/7_class/data_below_freezing.csv",
        "data/classified/7_class/data_extended_drift.csv"
    ]

    class_names = [
        "Spike",
        "Cyclical events",
        "Not Precooled",
        "Normal",
        "Initial Ramp",
        "Below Freezing",
        "Extended Drift"
    ]

    class_names_2 = ["Normal", "Anomaly"]

    # Load and merge data
    merged_data = load_and_merge_data(file_paths, class_names)

    # Group by Serial no
    grouped_data = merged_data.groupby("Serial no")

    # Create a mapping from Serial no to AnomalyClass
    serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

    # Load classified dataset
    classified_df_stringent = pd.read_csv('data/classified/2_class/classified_data_2_classes_stringent.csv')
    classified_df_stringent['Date / Time'] = pd.to_datetime(classified_df_stringent['Date / Time'])
    serial_to_2class_label_stringent = dict(zip(classified_df_stringent['Serial no'], classified_df_stringent['Label']))

    classified_df_moderate = pd.read_csv('data/classified/2_class/classified_data_2_classes_mid.csv')
    classified_df_moderate['Date / Time'] = pd.to_datetime(classified_df_moderate['Date / Time'])
    serial_to_2class_label_moderate = dict(zip(classified_df_moderate['Serial no'], classified_df_moderate['Label']))

    classified_df_relaxed = pd.read_csv('data/classified/2_class/classified_data_2_classes_relaxed.csv')
    classified_df_relaxed['Date / Time'] = pd.to_datetime(classified_df_relaxed['Date / Time'])
    serial_to_2class_label_relaxed = dict(zip(classified_df_relaxed['Serial no'], classified_df_relaxed['Label']))

    # Load dataset features
    config_path = 'config.ini'
    coop_path = "data/all_data_combined_meta.csv"
    bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"

    shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
    data = shipment_merger.resample_time_series()
    data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
    data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    data = norm_median(data, 'Temperature')

    # Load the saved HMM models
    model_path_2 = f"models/hmm/best_hmm_model_06-07-24_2_state.joblib"
    best_model_2 = load(model_path_2)

    model_path_3 = f"models/hmm/best_hmm_model_06-07-24_3_state.joblib"
    best_model_3 = load(model_path_3)

    model_path_8 = f"models/hmm/best_hmm_model_06-07-24_8_state.joblib"
    best_model_8 = load(model_path_8)

    # Predict hidden states and classify time series
    hidden_states_dict = {
        '2-state': {},
        '3-state': {},
        '8-state': {}
    }
    true_labels = []

    for serial_no, group in data.groupby('Serial no'):
        if serial_no not in serial_to_2class_label_relaxed or serial_no not in serial_to_anomaly_class:
            continue
        temperatures = group['Temperature'].values.reshape(-1, 1)
        
        # 2-state model predictions
        hidden_states_dict['2-state'][serial_no] = best_model_2.predict(temperatures)
        
        # 3-state model predictions
        hidden_states_dict['3-state'][serial_no] = best_model_3.predict(temperatures)
        
        # 8-state model predictions
        hidden_states_dict['8-state'][serial_no] = best_model_8.predict(temperatures)
        
        # True labels
        true_labels.append(serial_to_anomaly_class[serial_no])

    # Define state permutations for each model
    state_permutations_2 = [
        {0: 'Normal', 1: 'Anomaly'},
        {0: 'Anomaly', 1: 'Normal'}
    ]

    state_permutations_3 = []
    for n in range(4):
        perms = set(permutations(['Normal'] * n + ['Anomaly'] * (3 - n)))
        for perm in perms:
            state_permutations_3.append({i: perm[i] for i in range(3)})

    state_permutations_8 = []
    for n in range(8):
        perms = set(permutations(['Normal'] * n + ['Anomaly'] * (8 - n)))
        for perm in perms:
            state_permutations_8.append({i: perm[i] for i in range(8)})

    # Evaluate permutations and find the best metrics
    best_metrics_2, best_cm_2, best_perm_2, best_per_class_acc_2 = evaluate_permutations(true_labels, hidden_states_dict['2-state'], state_permutations_2, "2-state HMM")
    best_metrics_3, best_cm_3, best_perm_3, best_per_class_acc_3 = evaluate_permutations(true_labels, hidden_states_dict['3-state'], state_permutations_3, "3-state HMM")
    best_metrics_8, best_cm_8, best_perm_8, best_per_class_acc_8 = evaluate_permutations(true_labels, hidden_states_dict['8-state'], state_permutations_8, "8-state HMM")

    # Print the best metrics
    print("Best 2-State Model Metrics:")
    print(f"Accuracy: {best_metrics_2[0]}")
    print(f"Precision: {best_metrics_2[1]}")
    print(f"Recall: {best_metrics_2[2]}")
    print(f"F1 Score: {best_metrics_2[3]}")
    print(f"Balanced Accuracy: {best_metrics_2[4]}")
    print(f"Per Class Accuracy: Normal: {best_per_class_acc_2[0]}, Anomaly: {best_per_class_acc_2[1]}")
    print(f"State to Label Mapping: {best_perm_2}")

    print("\nBest 3-State Model Metrics:")
    print(f"Accuracy: {best_metrics_3[0]}")
    print(f"Precision: {best_metrics_3[1]}")
    print(f"Recall: {best_metrics_3[2]}")
    print(f"F1 Score: {best_metrics_3[3]}")
    print(f"Balanced Accuracy: {best_metrics_3[4]}")
    print(f"Per Class Accuracy: Normal: {best_per_class_acc_3[0]}, Anomaly: {best_per_class_acc_3[1]}")
    print(f"State to Label Mapping: {best_perm_3}")

    print("\nBest 8-State Model Metrics:")
    print(f"Accuracy: {best_metrics_8[0]}")
    print(f"Precision: {best_metrics_8[1]}")
    print(f"Recall: {best_metrics_8[2]}")
    print(f"F1 Score: {best_metrics_8[3]}")
    print(f"Balanced Accuracy: {best_metrics_8[4]}")
    print(f"Per Class Accuracy: Normal: {best_per_class_acc_8[0]}, Anomaly: {best_per_class_acc_8[1]}")
    print(f"State to Label Mapping: {best_perm_8}")

    # Plot confusion matrices with percentages
    plot_confusion_matrix(best_cm_2, ["Normal", "Anomaly"], "Best Confusion Matrix for 2-State Model")
    plot_confusion_matrix(best_cm_3, ["Normal", "Anomaly"], "Best Confusion Matrix for 3-State Model")
    plot_confusion_matrix(best_cm_8, ["Normal", "Anomaly"], "Best Confusion Matrix for 8-State Model")

    # Save the best metrics to a CSV file
    best_results = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy', 'Per Class Accuracy Normal', 'Per Class Accuracy Anomaly', 'State to Label Mapping'],
        '2-State Model': list(best_metrics_2[:5]) + [best_per_class_acc_2[0], best_per_class_acc_2[1], json.dumps(best_perm_2)],
        '3-State Model': list(best_metrics_3[:5]) + [best_per_class_acc_3[0], best_per_class_acc_3[1], json.dumps(best_perm_3)],
        '8-State Model': list(best_metrics_8[:5]) + [best_per_class_acc_8[0], best_per_class_acc_8[1], json.dumps(best_perm_8)]
    }
    best_results_df = pd.DataFrame(best_results)
    best_results_df.to_csv('results/hmm/best_hmm_model_performance_metrics_multi_class.csv', index=False)

if __name__ == "__main__":
    main()
