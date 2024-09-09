import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import CubicSpline
import configparser
from datasets.coop import CoopData
from preprocessing.trim import TimeSeriesTrimmer

def save_to_json(data_dict, filename):
    json_compatible_dict = {
        key: {
            'original': data['original'].tolist(),
            'augmented': [aug.tolist() for aug in data['augmented']]
        }
        for key, data in data_dict.items()
    }
    with open(filename, 'w') as f:
        json.dump(json_compatible_dict, f)

# Data augmentation functions
def time_warp(ts, sigma_range=(0.1, 0.3)):
    sigma = np.random.uniform(*sigma_range)
    time_steps = np.arange(len(ts))
    distorted_time_steps = np.cumsum(np.maximum(0, np.random.normal(loc=1.0, scale=sigma, size=len(ts))))
    distorted_time_steps /= distorted_time_steps[-1]
    return np.interp(np.linspace(0, 1, len(ts)), distorted_time_steps, ts)

def jitter(ts, sigma_range=(0.1, 0.3)):
    sigma = np.random.uniform(*sigma_range)
    return ts + np.random.normal(loc=0.0, scale=sigma, size=len(ts))

def scaling(ts, sigma_range=(0.05, 0.2)):
    sigma = np.random.uniform(*sigma_range)
    factor = np.random.normal(loc=1.0, scale=sigma)
    return ts * factor

def magnitude_warp(ts, sigma_range=(0.1, 0.3)):
    sigma = np.random.uniform(*sigma_range)
    start = len(ts) // 2
    end = len(ts)
    
    num_points = 5
    random_points = np.linspace(start, end, num_points)
    random_factors = np.random.normal(loc=1.0, scale=sigma, size=num_points)
    
    cs = CubicSpline(random_points, random_factors)
    warp_factor = cs(np.arange(start, end))
    
    ts[start:end] *= warp_factor
    return ts

def permutation(ts):
    start = len(ts) // 2
    end = len(ts)
    ts[start:end] = np.random.permutation(ts[start:end])
    return ts

def resample_time_series(df, interval='15min', fill_method='ffill'):
    resampled_data = []

    def resample_group(group):
        group = group.set_index('Date / Time')
        resampled_group = group.resample(interval).mean(numeric_only=True)
        resampled_group['Temperature'] = resampled_group['Temperature'].interpolate(method='linear')

        if resampled_group.isna().any().any():
            print("Warning: NaNs found after interpolation. These will be forward-filled.")
            resampled_group = resampled_group.fillna(method='ffill')

        resampled_group['Serial no'] = group['Serial no'].iloc[0]
        resampled_group['H_ShipmentId'] = group['H_ShipmentId'].iloc[0]
        resampled_group['OriginCityorTown'] = group['OriginCityorTown'].iloc[0]
        resampled_group['DestinationCityorTown'] = group['DestinationCityorTown'].iloc[0]
        
        resampled_data.append(resampled_group.reset_index())

    df.groupby('Serial no').apply(resample_group)

    return pd.concat(resampled_data, ignore_index=True).infer_objects()

def augment_time_series(grouped_data, target_count, label):
    original_count = len(grouped_data)
    num_augmentations_needed = max(0, target_count - original_count)
    augmentations_per_series = (num_augmentations_needed + original_count - 1) // original_count

    data_dict = {}
    serial_count = 1
    
    for name, group in grouped_data:
        original_ts = group['Temperature'].values
        augmentations = []
        for _ in range(augmentations_per_series):
            augmented_ts = original_ts.copy()
            augmentation_type = random.choice(['time_warp', 'jitter', 'scaling', 'magnitude_warp', 'permutation'])
            
            if augmentation_type == 'time_warp':
                augmented_ts = time_warp(augmented_ts)
            elif augmentation_type == 'jitter':
                augmented_ts = jitter(augmented_ts)
            elif augmentation_type == 'scaling':
                augmented_ts = scaling(augmented_ts)
            elif augmentation_type == 'magnitude_warp':
                augmented_ts = magnitude_warp(augmented_ts)
            elif augmentation_type == 'permutation':
                augmented_ts = permutation(augmented_ts)
                
            augmentations.append(augmented_ts)
        
        data_dict[name] = {'original': original_ts, 'augmented': augmentations}
        serial_count += 1
    
    if original_count >= target_count:
        selected_keys = list(grouped_data.groups.keys())[:target_count]
        data_dict = {key: data_dict[key] for key in selected_keys}
    
    return data_dict

# Function to visualize augmentations
def visualize_augmentations(data_dict, class_label, num_samples=5):
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, num_samples * 4))
    keys = list(data_dict.keys())
    
    for i in range(num_samples):
        key = keys[i]
        original_ts = data_dict[key]['original']
        augmented_ts = data_dict[key]['augmented']
        
        axs[i, 0].plot(original_ts, label="Original")
        axs[i, 0].set_title(f"Original {class_label} - {key}")
        axs[i, 0].set_xlabel("Relative Time")
        axs[i, 0].set_ylabel("Temperature")
        
        for j, aug_ts in enumerate(augmented_ts):
            axs[i, 1].plot(aug_ts, label=f"Augmented {j+1}", color='orange', alpha=0.6)
        axs[i, 1].set_title(f"Augmented {class_label} - {key}")
        axs[i, 1].set_xlabel("Relative Time")
        axs[i, 1].set_ylabel("Temperature")
        axs[i, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Load configuration items
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
coop_features = config["dataset"]["coop_features"].split(", ")

data_features = ['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown', 'Relative Time']

# Function to prepare and trim datasets
def prepare_and_trim_dataset(file_path, data_features):
    coop_data = CoopData(file_path, feature_list=data_features, dependent_var='Temperature')
    trimmer = TimeSeriesTrimmer(coop_data.data, temperature_column='Temperature')
    trimmed_data = trimmer.trim_time_series()
    resampled_data = resample_time_series(trimmed_data, interval='15min', fill_method='ffill')
    resampled_data['Relative Time'] = resampled_data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    resampled_data["Relative Time"] = pd.to_timedelta(resampled_data["Relative Time"]).dt.total_seconds()
    return resampled_data

# Load and prepare datasets
spike_data = prepare_and_trim_dataset("data/classified/7_class/data_spikes.csv", data_features)
cyclical_events_data = prepare_and_trim_dataset("data/classified/7_class/data_cyclical_events.csv", data_features)
not_precooled_data = prepare_and_trim_dataset("data/classified/7_class/data_not_precooled.csv", data_features)
norm_data = prepare_and_trim_dataset("data/classified/7_class/data_normal.csv", data_features)
initial_ramp_data = prepare_and_trim_dataset("data/classified/7_class/data_initial_ramp.csv", data_features)
below_freezing_data = prepare_and_trim_dataset("data/classified/7_class/data_below_freezing.csv", data_features)
extended_drift_data = prepare_and_trim_dataset("data/classified/7_class/data_extended_drift.csv", data_features)

# Group datasets by "Serial no"
grouped_spike = spike_data.groupby("Serial no")
grouped_cyclical_events = cyclical_events_data.groupby("Serial no")
grouped_not_precooled = not_precooled_data.groupby("Serial no")
grouped_norm = norm_data.groupby("Serial no")
grouped_initial_ramp = initial_ramp_data.groupby("Serial no")
grouped_below_freezing = below_freezing_data.groupby("Serial no")
grouped_extended_drift = extended_drift_data.groupby("Serial no")

# Augment datasets to reach the target count
target_count = 200
augmented_spike = augment_time_series(grouped_spike, target_count, label="spike")
augmented_cyclical_events = augment_time_series(grouped_cyclical_events, target_count, label="cyclical_events")
augmented_not_precooled = augment_time_series(grouped_not_precooled, target_count, label="not_precooled")
augmented_initial_ramp = augment_time_series(grouped_initial_ramp, target_count, label="initial_ramp")
augmented_below_freezing = augment_time_series(grouped_below_freezing, target_count, label="below_freezing")
augmented_extended_drift = augment_time_series(grouped_extended_drift, target_count, label="extended_drift")
augmented_normal = augment_time_series(grouped_norm, target_count, label="normal")

save_to_json(augmented_normal, 'data/augmented/augmented_normal_200.json')
save_to_json(augmented_spike, 'data/augmented/augmented_spike_200.json')
save_to_json(augmented_cyclical_events, 'data/augmented/augmented_cyclical_events_200.json')
save_to_json(augmented_not_precooled, 'data/augmented/augmented_not_precooled_200.json')
save_to_json(augmented_initial_ramp, 'data/augmented/augmented_initial_ramp_200.json')
save_to_json(augmented_below_freezing, 'data/augmented/augmented_below_freezing_200.json')
save_to_json(augmented_extended_drift, 'data/augmented/augmented_extended_drift_200.json')

print("Augmented data saved")