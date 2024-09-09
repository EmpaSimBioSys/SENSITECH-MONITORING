import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from utils.config import ConfigLoader 
from preprocessing.feature_generation import FeatureGenerator
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning, message="y_pred contains classes not in y_true")

# Configuration and paths
config_loader = ConfigLoader()
base_dir = config_loader.get_value('paths', 'data')
model_dir = config_loader.get_value('paths', 'models')
plots_dir = config_loader.get_value('paths', 'plots')
model_parameters_dir = config_loader.get_value('paths', 'model_parameters')
config_dir = config_loader.get_value('paths', 'config')

# Load augmented data from JSON files
def load_augmented_data(file_path: str) -> Dict[str, List[Any]]:
    with open(file_path, 'r') as f:
        return json.load(f)

# Process JSON data to combine original and augmented data
def process_json_data(data: Dict[str, Dict[str, Any]], class_label: str) -> Dict[str, List[Any]]:
    combined_data = {}
    for serial, values in data.items():
        original_data = values['original']
        augmented_data = values['augmented']
        
        # Add original data
        combined_data[f'{serial}'] = {'series': original_data, 'label': class_label}
        
        # Add augmented data
        for i, series in enumerate(augmented_data):
            combined_data[f'{serial}_augmented_{i}'] = {'series': series, 'label': class_label}
            
    return combined_data

# Generate a DataFrame from the combined data
def create_dataframe(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    start_time = pd.Timestamp.now()
    
    for serial_no, info in data.items():
        series = info['series']
        label = info['label']
        
        for i, temperature in enumerate(series):
            row = {
                'Date / Time': start_time + pd.Timedelta(minutes=15 * i),
                'Serial no': serial_no,
                'Temperature': temperature,
                'H_ShipmentId': f'SH_{serial_no}',
                'OriginCityorTown': 'CityA',
                'DestinationCityorTown': 'CityB',
                'Relative Time': i,
                'Label': label
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


# Extract features and labels
def extract_features_labels(feature_set_key: str, all_features: Dict[str, Dict[str, Any]], combined_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels for a given feature set key.

    Args:
        feature_set_key (str): The key for the feature set.
        all_features (Dict[str, Dict[str, Any]]): Combined feature data.
        combined_df (pd.DataFrame): DataFrame containing the labels for each serial number.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features and labels arrays.
    """
    # Create a mapping from serial number to label, ensuring we only take one label per serial number
    serial_to_label = combined_df.groupby('Serial no')['Label'].first().to_dict()
    
    features, labels = [], []

    for serial, feature_sets in all_features.items():
        if serial in serial_to_label:
            if feature_set_key in feature_sets:
                features.append(feature_sets[feature_set_key])
                labels.append(serial_to_label[serial])

    return np.array(features), np.array(labels)

# Train and evaluate models
def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, 
                       split_key: str, model: Any, model_name: str, params: Dict[str, Any]) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    print(f"Training {model_name} with parameters: {params} on feature set: {split_key}")
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
    
    # Calculate per class accuracy
    sum_per_class = cm.sum(axis=1)
    sum_per_class_safe = np.where(sum_per_class == 0, 1, sum_per_class)
    per_class_accuracy = cm.diagonal() / sum_per_class_safe
    per_class_accuracy = np.where(sum_per_class == 0, 0, per_class_accuracy)
    
    return balanced_acc, cm, cm_normalized, per_class_accuracy

# Plot confusion matrices
def plot_confusion_matrices_grid() -> None:
    fig, axes = plt.subplots(len(combined_data.keys()), len(split_ratios), figsize=(24, len(combined_data.keys()) * 6))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    norm = Normalize(vmin=0, vmax=100)

    cax = None  # Initialize cax

    for i, feature_set_key in enumerate(combined_data.keys()):
        for j, split_ratio in enumerate(split_ratios):
            split_key = f'{feature_set_key}_{int((1-split_ratio)*100)}_{int(split_ratio*100)}'
            ax = axes[i, j]
            
            if split_key in best_models:
                for model_name, params in best_models[split_key].items():
                    if model_name in results[split_key]:
                        cm_normalized = np.array(results[split_key][model_name]['Normalized Confusion Matrix'])
                        
                        cax = ax.matshow(cm_normalized, cmap='Reds', norm=norm)
                        
                        # Annotate the confusion matrix with values
                        for (m, n), value in np.ndenumerate(cm_normalized):
                            ax.text(n, m, f"{value:.1f}%", ha='center', va='center', color='black')

                        ax.set_title(f"{model_name}\n{split_key}", fontsize=10)
                        ax.set_xticks(range(len(cm_normalized)))
                        ax.set_yticks(range(len(cm_normalized)))
                        ax.set_xticklabels(range(len(cm_normalized)), fontsize=8)
                        ax.set_yticklabels(range(len(cm_normalized)), fontsize=8)
                    else:
                        ax.axis('off')  # Turn off axes if no results
            else:
                ax.axis('off')  # Turn off axes if no results

            if i == 0:  # Add the column header on the top row
                ax.set_title(f"Split Ratio: {split_ratio}", fontsize=12)

            if j == 0:  # Add the row header on the first column
                ax.set_ylabel(f"Feature Set:\n{feature_set_key}", fontsize=12, rotation=0, labelpad=50)

    # Add a colorbar to the figure only if cax is set
    if cax:
        cbar = fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Percentage', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    plt.suptitle('Confusion Matrices Grid for Multi-Class Labels', fontsize=16)
    plt.savefig(os.path.join(plots_dir, "confusion_matrix/multi-label/confusion_matrices_grid_update.png"))
    plt.show()


# Load and process augmented data
augmented_data_files = [
    'augmented/augmented_spike_200.json', 'augmented/augmented_cyclical_events_200.json',
    'augmented/augmented_not_precooled_200.json', 'augmented/augmented_initial_ramp_200.json', 'augmented/augmented_below_freezing_200.json',
    'augmented/augmented_extended_drift_200.json', 'augmented/augmented_normal_200.json'
]
combined_data = {}
for file_path in augmented_data_files:

    # Extract class label using regex
    match = re.search(r'augmented_([a-zA-Z0-9_]+)_\d+', os.path.basename(file_path))
    if match:
        class_label = match.group(1)  # Get the matched class label
    else:
        continue  # Skip if no match is found

    file_data = load_augmented_data(os.path.join(base_dir, file_path))
    combined_data.update(process_json_data(file_data, class_label))

# Create DataFrame from combined data
df = create_dataframe(combined_data)

# Feature generation
feat_generator = FeatureGenerator(df, config_path=config_dir, coop_path=f"{base_dir}/all_data_combined_meta.csv", bama_path=f"{base_dir}/SWP_BAMA_Sensor_Shipment_berries.csv", new_data=True)
feat_generator.data = feat_generator.load_data(df=df)
all_features, _, _, _ = feat_generator.extract_all_features()

# Models and parameter sets
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42)
}

param_sets = {
    'RandomForest': [
        {'max_depth': d, 'n_estimators': n}
        for d in [3, 5, 7, 9, 12]
        for n in [50, 100, 200]
    ],
    'GradientBoosting': [
        {'max_depth': d, 'n_estimators': n, 'learning_rate': lr}
        for d in [3, 5, 7, 9, 12]
        for n in [50, 100, 200]
        for lr in [0.01, 0.05, 0.1, 0.2, 0.3]
    ]
}

# Split ratios and initialize results
split_ratios = [0.4, 0.3, 0.2]
results = {}
feature_sets = ['statistical', 'frequency_200', 'frequency_300', 'hmm_2_state', 'hmm_3_state', 'hmm_8_state', 'ae_latent_features', 'deep_latent_features', 'combined']
best_models = {split_key: {} for split_key in [f'{feature_set_key}_{int((1-test_size)*100)}_{int(test_size*100)}' for feature_set_key in feature_sets for test_size in split_ratios]}

augmented_spike = load_augmented_data(os.path.join(base_dir,'augmented/augmented_spike_200.json'))
augmented_cool_defrost = load_augmented_data(os.path.join(base_dir,'augmented/augmented_cyclical_events_200.json'))
augmented_not_precooled = load_augmented_data(os.path.join(base_dir,'augmented/augmented_not_precooled_200.json'))
augmented_initial_ramp = load_augmented_data(os.path.join(base_dir,'augmented/augmented_initial_ramp_200.json'))
augmented_top_freezing = load_augmented_data(os.path.join(base_dir,'augmented/augmented_below_freezing_200.json')) 
augmented_extended_drift = load_augmented_data(os.path.join(base_dir,'augmented/augmented_extended_drift_200.json'))
augmented_normal = load_augmented_data(os.path.join(base_dir,'augmented/augmented_normal_200.json')) 

label_dfs = {
    'spikes': augmented_spike,
    'cyclical events': augmented_cool_defrost,
    'initial ramp up': augmented_initial_ramp,
    'Not pre-cooled': augmented_not_precooled,
    'Below freezing': augmented_top_freezing,
    'Extended drift': augmented_extended_drift,
    'normal': augmented_normal,
}

# Training and evaluation loop
for feature_set_key in feature_sets:
    X, y = extract_features_labels(feature_set_key, all_features, combined_df=df)
    
    for test_size in split_ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train = np.nan_to_num(X_train, nan=np.nanmean(X_train, axis=0))
        X_test = np.nan_to_num(X_test, nan=np.nanmean(X_test, axis=0))
        y_train = np.nan_to_num(y_train, nan=-1)
        y_test = np.nan_to_num(y_test, nan=-1)

        split_key = f'{feature_set_key}_{int((1-test_size)*100)}_{int(test_size*100)}'
        
        best_balanced_acc = -1
        best_cm = None
        best_cm_normalized = None
        best_per_class_accuracy = None
        best_model_name = None
        best_model_params = None
        
        for model_name, model in models.items():
            if model_name in param_sets:
                for params in param_sets[model_name]:
                    balanced_acc, cm, cm_normalized, per_class_accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, split_key, model, model_name, params)
                    if balanced_acc > best_balanced_acc:
                        best_balanced_acc = balanced_acc
                        best_cm = cm
                        best_cm_normalized = cm_normalized
                        best_per_class_accuracy = per_class_accuracy
                        best_model_name = model_name
                        best_model_params = params
                        best_model = model.set_params(**params)
                # Save the best model and its results
                if split_key not in results:
                    results[split_key] = {}
                results[split_key][model_name] = {
                    'Balanced Accuracy': best_balanced_acc,
                    'Per Class Accuracy': best_per_class_accuracy.tolist(),
                    'Confusion Matrix': best_cm.tolist(),
                    'Normalized Confusion Matrix': best_cm_normalized.tolist()
                }
                best_models[split_key][model_name] = best_model_params
                model_filename = f"{best_model_name}_{split_key}.joblib"
                joblib.dump(best_model, os.path.join(model_dir, model_filename))
                
            else:
                # SVM case without additional parameter tuning
                balanced_acc, cm, cm_normalized, per_class_accuracy = train_and_evaluate(X_train, y_train, X_test, y_test, split_key, model, model_name, {})
                
                # Save the best model and its results
                if split_key not in results:
                    results[split_key] = {}
                results[split_key][model_name] = {
                    'Balanced Accuracy': balanced_acc,
                    'Per Class Accuracy': per_class_accuracy.tolist(),
                    'Confusion Matrix': cm.tolist(),
                    'Normalized Confusion Matrix': cm_normalized.tolist()
                }
                best_models[split_key][model_name] = 'SVM'

# Save the results and best model parameters
with open(os.path.join(model_parameters_dir, 'model_results_multi_label_200_exhaustive.json'), 'w') as f:
    json.dump(results, f)

with open(os.path.join(model_parameters_dir, 'best_model_params_200_exhaustive.json'), 'w') as f:
    json.dump(best_models, f)

# Plot confusion matrices grid
plot_confusion_matrices_grid()


