import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
from dtaidistance import dtw
from datasets.coop import CoopData
from datasets.data_merger import ShipmentDataMerger
from preprocessing.trim import TimeSeriesTrimmer

def z_normalize(ts):
    ts_mean = np.mean(ts)
    ts_std = np.std(ts)
    return (ts - ts_mean) / ts_std

# Load configuration items
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
coop_features = config["dataset"]["coop_features"].split(", ")

data_features = ['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown', 'Relative Time']

# Load  spikey anomaly classes:
coop_shipments_spike = CoopData("data/data_spike.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_spike.data, temperature_column='Temperature')
trimmed_coop_shipments_spike = trimmer.trim_time_series()
grouped_shipments_coop_spike = trimmed_coop_shipments_spike.groupby("Serial no")
spike_serials = list(coop_shipments_spike.data['Serial no'].unique())

# Load  cool defrost anomaly classes:
coop_shipments_cool_defrost = CoopData("data/data_cool_defrost.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_cool_defrost.data, temperature_column='Temperature')
trimmed_coop_shipments_cool_defrost = trimmer.trim_time_series()
grouped_shipments_coop_cool_defrost = trimmed_coop_shipments_cool_defrost.groupby("Serial no")
cool_defrost_serials = list(coop_shipments_cool_defrost.data['Serial no'].unique())

# Load  excursion anomaly classes:
coop_shipments_excursion = CoopData("data/data_excursion.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_excursion.data, temperature_column='Temperature')
trimmed_coop_shipments_excursion = trimmer.trim_time_series()
grouped_shipments_coop_excursion = trimmed_coop_shipments_excursion.groupby("Serial no")
excursion_serials = list(coop_shipments_excursion.data['Serial no'].unique())

# Load  not_precooled anomaly classes:
coop_shipments_not_precooled = CoopData("data/data_not_precooled.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_not_precooled.data, temperature_column='Temperature')
trimmed_coop_shipments_not_precooled= trimmer.trim_time_series()
grouped_shipments_coop_not_precooled = trimmed_coop_shipments_not_precooled.groupby("Serial no")
not_precooled_serials = list(coop_shipments_not_precooled.data['Serial no'].unique())

# Load  normal classes:
coop_shipments_norm = CoopData("data/data_norm.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_norm.data, temperature_column='Temperature')
trimmed_coop_shipments_norm = trimmer.trim_time_series()
grouped_shipments_coop_norm = trimmed_coop_shipments_norm.groupby("Serial no")
norm_serials = list(coop_shipments_norm.data['Serial no'].unique())

# Load  initial ramp anomaly classes:
coop_shipments_initial_ramp = CoopData("data/data_initial_ramp.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_initial_ramp.data, temperature_column='Temperature')
trimmed_coop_shipments_initial_ramp = trimmer.trim_time_series()
grouped_shipments_coop_initial_ramp= trimmed_coop_shipments_initial_ramp.groupby("Serial no")
initial_ramp_serials = list(coop_shipments_initial_ramp.data['Serial no'].unique())

# Load  Top freezing anomaly classes:
coop_shipments_top_freezing = CoopData("data/data_chilling_injury.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_top_freezing.data, temperature_column='Temperature')
trimmed_coop_shipments_top_freezing = trimmer.trim_time_series()
grouped_shipments_coop_top_freezing= trimmed_coop_shipments_top_freezing.groupby("Serial no")
top_freezing_serials = list(coop_shipments_top_freezing.data['Serial no'].unique())

# Load  Extended drift anomaly classes:
coop_shipments_extended_drift = CoopData("data/data_extended_drift.csv", feature_list=data_features, dependent_var='Temperature')
trimmer = TimeSeriesTrimmer(coop_shipments_extended_drift.data, temperature_column='Temperature')
trimmed_coop_shipments_extended_drift = trimmer.trim_time_series()
grouped_shipments_coop_extended_drift = trimmed_coop_shipments_extended_drift.groupby("Serial no")
extended_drift_serials = list(coop_shipments_extended_drift.data['Serial no'].unique())

# Merge all the different classes
merged_shipments = pd.concat([
    trimmed_coop_shipments_spike,
    trimmed_coop_shipments_cool_defrost,
    trimmed_coop_shipments_excursion,
    trimmed_coop_shipments_not_precooled,
    trimmed_coop_shipments_norm,
    trimmed_coop_shipments_initial_ramp,
    trimmed_coop_shipments_top_freezing,
    trimmed_coop_shipments_extended_drift
])

# Group the merged dataframe by "Serial no" and extract the temperature time series for each shipment
grouped_shipments_all = merged_shipments.groupby("Serial no")

# Prepare data for DTW
shipment_time_series = {name: group['Temperature'].values for name, group in grouped_shipments_all}

# Create a list of serial numbers to track index in the distance matrix
serial_numbers = list(shipment_time_series.keys())

distance_matrix = np.load("data/dtw_distance_matrix/dtw_distance_matrix.npy")

# Define serial number order
ordered_serials = norm_serials + spike_serials + cool_defrost_serials + excursion_serials + not_precooled_serials + initial_ramp_serials + top_freezing_serials + extended_drift_serials

# Reorder distance matrix
ordered_indices = [serial_numbers.index(serial) for serial in ordered_serials]
ordered_distance_matrix = distance_matrix[np.ix_(ordered_indices, ordered_indices)]

rows_with_inf = np.isinf(ordered_distance_matrix).any(axis=1)

# Filter out rows with infinite values
filtered_distance_matrix = ordered_distance_matrix[~rows_with_inf]

# Step 2: Replace NaN values with a large finite number
large_number = np.nanmax(filtered_distance_matrix) * 10
filtered_distance_matrix[np.isnan(filtered_distance_matrix)] = large_number

# Convert distance matrix to affinity matrix
beta = 1
distance_matrix_std = np.nanstd(filtered_distance_matrix)

if distance_matrix_std == 0:
    distance_matrix_std = 1

affinity_matrix = np.exp(-beta * filtered_distance_matrix / distance_matrix_std)

# Reorder affinity matrix by highest affinity to lowest
sorted_affinity_indices = np.argsort(-affinity_matrix.sum(axis=1))

# Plot the three heatmaps
fig, axes = plt.subplots(1, 3, figsize=(30, 20))

# Original affinity matrix heatmap
sns.heatmap(affinity_matrix, xticklabels=serial_numbers, yticklabels=serial_numbers, cmap='viridis', ax=axes[0])
axes[0].set_title("Original Affinity Matrix Heatmap")
axes[0].set_xlabel("Serial no")
axes[0].set_ylabel("Serial no")


# Reordered affinity matrix heatmap (based on the current serial number order in ordered_serials)
reordered_affinity_matrix = affinity_matrix[np.ix_(ordered_indices, ordered_indices)]
sns.heatmap(reordered_affinity_matrix, xticklabels=ordered_serials, yticklabels=ordered_serials, cmap='viridis', ax=axes[1])
axes[1].set_title("Reordered Affinity Matrix Heatmap")
axes[1].set_xlabel("Serial no")
axes[1].set_ylabel("Serial no")

# Affinity matrix reordered according to highest affinity to lowest
highest_affinity_matrix = affinity_matrix[np.ix_(sorted_affinity_indices, sorted_affinity_indices)]
sns.heatmap(highest_affinity_matrix, xticklabels=np.array(serial_numbers)[sorted_affinity_indices], yticklabels=np.array(serial_numbers)[sorted_affinity_indices], cmap='viridis', ax=axes[2])
axes[2].set_title("Highest to Lowest Affinity Matrix Heatmap")
axes[2].set_xlabel("Serial no")
axes[2].set_ylabel("Serial no")

# Remove the tick labels 
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].set_xticks([])
axes[1].set_yticks([])

axes[2].set_xticks([])
axes[2].set_yticks([])

plt.tight_layout()
plt.savefig("results/plots/affinity_matrices/affinity_matrices_heatmaps.png", format="png", dpi=300)
plt.show()
print("Done")



