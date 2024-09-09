import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils.config import ConfigLoader  
from preprocessing.feature_generation import FeatureGenerator
from typing import Dict, Any, Tuple

def load_data(base_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the datasets from the specified directories.

    Args:
        base_dir (str): The base directory path where data is stored.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames for relaxed, moderate, and stringent datasets.
    """
    relaxed_df = pd.read_csv(os.path.join(base_dir, 'classified/2_class/classified_data_2_classes_relaxed.csv'))
    moderate_df = pd.read_csv(os.path.join(base_dir, 'classified/2_class/classified_data_2_classes_mid.csv'))
    stringent_df = pd.read_csv(os.path.join(base_dir, 'classified/2_class/classified_data_2_classes_stringent.csv'))

    return relaxed_df, moderate_df, stringent_df

def prepare_datasets(relaxed_df: pd.DataFrame, moderate_df: pd.DataFrame, stringent_df: pd.DataFrame) -> None:
    """
    Prepare the datasets by renaming the label columns and removing duplicates.

    Args:
        relaxed_df (pd.DataFrame): The relaxed dataset.
        moderate_df (pd.DataFrame): The moderate dataset.
        stringent_df (pd.DataFrame): The stringent dataset.
    """
    relaxed_df.rename(columns={'Label': 'Label_relaxed'}, inplace=True)
    moderate_df.rename(columns={'Label': 'Label_moderate'}, inplace=True)
    stringent_df.rename(columns={'Label': 'Label_stringent'}, inplace=True)

    relaxed_df.drop_duplicates(subset=['Serial no'], inplace=True)
    moderate_df.drop_duplicates(subset=['Serial no'], inplace=True)
    stringent_df.drop_duplicates(subset=['Serial no'], inplace=True)

def extract_features_labels(feature_set_key: str, label_df: pd.DataFrame, label_col: str, all_features: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from the datasets.

    Args:
        feature_set_key (str): The feature set key.
        label_df (pd.DataFrame): The dataframe containing labels.
        label_col (str): The column name for the labels.
        all_features (Dict[str, Dict[str, Any]]): Dictionary containing all extracted features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
    """
    serial_to_label = dict(zip(label_df['Serial no'], label_df[label_col]))
    features, labels = [], []

    for serial, feature_sets in all_features.items():
        if serial in serial_to_label:
            features.append(feature_sets[feature_set_key])
            labels.append(serial_to_label[serial])

    label_mapping = {'Normal': 0, 'Anomaly': 1}
    labels = [label_mapping[label] for label in labels]

    return np.array(features), np.array(labels)

def balance_classes(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance the classes by under-sampling the majority class.

    Args:
        X (np.ndarray): The feature array.
        y (np.ndarray): The label array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The balanced feature and label arrays.
    """
    class_counts = np.bincount(y)
    minority_class = np.argmin(class_counts)
    minority_count = class_counts[minority_class]

    indices_minority = np.where(y == minority_class)[0]
    indices_majority = np.where(y != minority_class)[0]

    sampled_indices_majority = np.random.choice(indices_majority, size=minority_count, replace=False)
    balanced_indices = np.concatenate([indices_minority, sampled_indices_majority])

    np.random.shuffle(balanced_indices)

    return X[balanced_indices], y[balanced_indices]

def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                       split_key: str, model: Any, model_name: str, results: Dict[str, Dict[str, Any]], 
                       label_type: str, model_dir: str) -> None:
    """
    Train and evaluate the model, storing the results and saving the model.

    Args:
        X_train (np.ndarray): The training feature set.
        y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The testing feature set.
        y_test (np.ndarray): The testing labels.
        split_key (str): A key representing the split configuration.
        model (Any): The model instance to train.
        model_name (str): The name of the model.
        results (Dict[str, Dict[str, Any]]): Dictionary to store evaluation results.
        label_type (str): The type of label ('relaxed', 'moderate', 'stringent').
        model_dir (str): Base folder path to save the trained models.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100

    if label_type not in results:
        results[label_type] = {}

    if split_key not in results[label_type]:
        results[label_type][split_key] = {}

    results[label_type][split_key][model_name] = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'Confusion Matrix': cm.tolist(),
        'Normalized Confusion Matrix': cm_normalized.tolist()
    }

    model_filename = f"{model_name}_{label_type}_{split_key}.joblib"
    joblib.dump(model, os.path.join(model_dir, model_filename))

def plot_confusion_matrices_grid(label_type: str, results: Dict[str, Dict[str, Any]], feature_sets: list, 
                                 split_ratios: list, models: Dict[str, Any], plots_dir: str) -> None:
    """
    Plot and save the confusion matrices for all models in a grid.

    Args:
        label_type (str): The type of label.
        results (Dict[str, Dict[str, Any]]): The results dictionary containing evaluation metrics.
        feature_sets (list): List of feature sets used.
        split_ratios (list): List of split ratios used.
        model_dir (str): Directory to save the confusion matrices.
    """
    fig, axes = plt.subplots(len(feature_sets), len(split_ratios) * 3, figsize=(24, len(feature_sets) * 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    norm = Normalize(vmin=0, vmax=100)

    for i, feature_set_key in enumerate(feature_sets):
        for j, split_ratio in enumerate(split_ratios):
            split_key = f'{feature_set_key}_{int((1-split_ratio)*100)}_{int(split_ratio*100)}'
            for k, model_name in enumerate(models.keys()):
                ax = axes[i, j * 3 + k]
                cm_normalized = np.array(results[label_type][split_key][model_name]['Normalized Confusion Matrix'])
                
                cax = ax.matshow(cm_normalized, cmap='Reds', norm=norm)
                ax.set_xticks([])
                ax.set_yticks([])
    
    cbar = fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Percentage')
    plt.suptitle(f'Confusion Matrices Grid for {label_type.capitalize()} Labels', fontsize=16)
    # plt.savefig(os.path.join(plots_dir, f"confusion_matrix/confusion_matrices_grid_{label_type}.png"))
    plt.show()

def main() -> None:
    """
    Main function to execute the data loading, feature extraction, model training, and evaluation.
    """
    config_loader = ConfigLoader()
    base_dir = config_loader.get_value('paths', 'data')
    model_dir = os.path.join(config_loader.get_value('paths', 'models'), '2_class')
    plots_dir = config_loader.get_value('paths', 'plots')

    relaxed_df, moderate_df, stringent_df = load_data(base_dir)
    prepare_datasets(relaxed_df, moderate_df, stringent_df)

    feat_generator = FeatureGenerator(relaxed_df, config_path='config.ini', coop_path=os.path.join(base_dir, "all_data_combined_meta.csv"), bama_path=os.path.join(base_dir, "SWP_BAMA_Sensor_Shipment_berries.csv"))
    feat_generator.data = feat_generator.load_data(df=False)
    all_features, _, _, _ = feat_generator.extract_all_features()

    label_dfs = {'relaxed': relaxed_df, 'moderate': moderate_df, 'stringent': stringent_df}
    results = {}

    feature_sets = ['statistical', 'frequency_200', 'frequency_300', 'ae_latent_features', 'deep_latent_features', 'combined']
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42)
    }
    split_ratios = [0.4, 0.3, 0.2]

    for label_type, label_df in label_dfs.items():
        for feature_set_key in feature_sets:
            X, y = extract_features_labels(feature_set_key, label_df, f'Label_{label_type}', all_features)

            X_balanced, y_balanced = balance_classes(X, y)
            for test_size in split_ratios:
                X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size, random_state=42)
                split_key = f'{feature_set_key}_{int((1-test_size)*100)}_{int(test_size*100)}'

                for model_name, model in models.items():
                    train_and_evaluate(X_train, y_train, X_test, y_test, split_key, model, model_name, results, label_type, model_dir)

    with open(os.path.join(model_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f)

    for label_type in label_dfs.keys():
        plot_confusion_matrices_grid(label_type, results, feature_sets, split_ratios, models, plots_dir)

if __name__ == "__main__":
    main()
