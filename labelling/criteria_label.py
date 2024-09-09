import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, entropy, ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler
import configparser
from datasets.coop import CoopData
from scipy.signal import find_peaks
from preprocessing.trim import TimeSeriesTrimmer
from datasets.data_merger import ShipmentDataMerger

# Load the dataset
config_path = 'config.ini'
coop_path = "data/all_data_combined_meta.csv"
bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"
shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path, trim=True)
data = shipment_merger.merged_dataframe
data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()

# Function to add features to the dataset
def add_features(df):
    df['Date / Time'] = pd.to_datetime(df['Date / Time'], utc=True)
    df['hour'] = df['Date / Time'].dt.hour
    df['day_of_week'] = df['Date / Time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    return df

data = add_features(data)
break_counts = []

# Function to check for breaks in the measurement signal
def check_for_breaks(group, interval='30T'):
    group = group.sort_values('Date / Time')
    time_diff = group['Date / Time'].diff().dropna()
    if any(time_diff > pd.Timedelta(interval)):
        break_counts.append(group['Serial no'].unique()[0])
        return False
    return True

# Function to classify time series based on stringent Criteria set 
def classify_time_series(group):
    temperature = group['Temperature'].values
    date_time = group['Date / Time'].values
    
    # Criteria 1: All data points within 0 - 6 °C
    if not np.all((temperature >= 0) & (temperature <= 6)):
        return "Anomaly"
    
    # Criteria 2: Gradient check
    gradient = np.diff(temperature)
    if np.any(gradient > 2):
        return "Anomaly"
    
    # Criteria 3: Trip length check (within 3 days)
    trip_length = (date_time[-1] - date_time[0]).astype('timedelta64[h]') / np.timedelta64(1, 'h')
    if trip_length > 72:  # 3 days * 24 hours
        return "Anomaly"
    
    # Criteria 4: Initial temperature check
    if temperature[0] > 6:
        return "Anomaly"
    
    # Criteria 5: Temperature rising in the first 2 hours (8 data points)
    initial_period = 12  # First 3 hours 
    if all(np.diff(temperature[:initial_period]) > 0):
        return "Anomaly"
    
    # Criteria 6: Continuously rising temperature (smoothed signal)
    for window_size in [4, 8, 12]:  # 1 hour, 2 hours, 3 hours (
        smoothed_temp = np.convolve(temperature, np.ones(window_size)/window_size, mode='valid')
        if np.all(np.diff(smoothed_temp) > 0):
            return "Anomaly"
    
    return "Normal"

# Function to classify time series based on Moderate criteria set
def classify_time_series_mid(group):
    temperature = group['Temperature'].values
    date_time = group['Date / Time'].values
    
    # Criteria 1: 50% of the total data points are above 6°C or below -2°C
    total_points = len(temperature)
    if np.sum((temperature > 6) | (temperature < -2)) > 0.5 * total_points:
        return "Anomaly"
    
    # Criteria 2: Gradient check
    gradient = np.diff(temperature)
    if np.any(gradient > 2):
        return "Anomaly"
    
    
    # Criteria 4: Initial temperature check
    if temperature[0] > 6:
        return "Anomaly"
    
    # Criteria 5: Temperature rising in the first 2 hours (8 data points)
    initial_period = 24  # First 6 hours (15-minute intervals)
    if all(np.diff(temperature[:initial_period]) > 0):
        return "Anomaly"
    
    # Criteria 6: Continuously rising temperature (smoothed signal)
    #  6 hours (15-minute intervals)
    smoothed_temp = np.convolve(temperature, np.ones(24)/24, mode='valid')
    if np.all(np.diff(smoothed_temp) > 0):
        return "Anomaly"
    
    return "Normal"


# Function to classify time series based on Relaxed criteria set
def classify_time_series_relaxed(group):
    temperature = group['Temperature'].values
    date_time = group['Date / Time'].values
    
    # Criteria 1: 20% of the total data points are above 6°C or below -2°C
    total_points = len(temperature)
    if np.sum((temperature > 6) | (temperature < -2)) > 0.2 * total_points:
        return "Anomaly"
    
    # Criteria 2: Initial temperature check (initial temperature > 9°C)
    if temperature[0] > 9:
        return "Anomaly"
    
    # Criteria 3: Total positive trend 
    if np.all(np.diff(temperature) > 0):
        return "Anomaly"
    
    # Criteria 4: Gradient between consecutive points above 6°C
    gradient = np.diff(temperature)
    if np.any(gradient > 6):
        return "Anomaly"
    
    return "Normal"
    

# Remove groups with breaks in the measurement signal
shipment_groups = data.groupby('Serial no')
filtered_groups = {name: group for name, group in shipment_groups}

print(f"The number of shipments with breaks is {len(break_counts)} out of 1256")


# Classify each shipment
classification_results = []
for name, group in filtered_groups.items():
    classification = classify_time_series_mid(group)
    classification_results.append((name, classification))

# Add the classification results to the original dataset
classification_df = pd.DataFrame(classification_results, columns=['Serial no', 'Label'])
data = data[~data["Serial no"].isin(break_counts)]
df = data.merge(classification_df, on='Serial no', how='left')

# Save the new dataset
df.to_csv('data/classified/classified_data_2_classes_mid.csv', index=False)

# Print classification distribution
print(f"The number of normal shipments is {len(df[(df['Label'] == 'Normal')]['Serial no'].unique())}")
print(f"The number of anomalous shipments is {len(df[(df['Label'] == 'Anomaly')]['Serial no'].unique())}")

df_groups = df.groupby('Serial no')

# Function to compute statistical properties
def compute_statistical_properties(group):
    temperature = group['Temperature'].values
    
    mean_temp = np.mean(temperature)
    var_temp = np.var(temperature)
    std_temp = np.std(temperature)
    kurt_temp = kurtosis(temperature)
    skew_temp = skew(temperature)
    avg_gradient = np.mean(np.diff(temperature))
    
    # Degree minutes
    degree_minutes_0 = np.sum(np.maximum(temperature - 0, 0))
    degree_minutes_1 = np.sum(np.maximum(temperature - 1, 0))
    degree_minutes_2 = np.sum(np.maximum(temperature - 2, 0))
    degree_minutes_3 = np.sum(np.maximum(temperature - 3, 0))
    degree_minutes_4 = np.sum(np.maximum(temperature - 4, 0))
    degree_minutes_5 = np.sum(np.maximum(temperature - 5, 0))
    degree_minutes_6 = np.sum(np.maximum(temperature - 6, 0))

    degree_minutes_0_cool = np.sum(np.maximum(0 - temperature, 0))
    degree_minutes_1_cool = np.sum(np.maximum(1 - temperature, 0))
    degree_minutes_2_cool = np.sum(np.maximum(2 - temperature, 0))
    degree_minutes_3_cool = np.sum(np.maximum(3 - temperature, 0))
    degree_minutes_4_cool = np.sum(np.maximum(4 - temperature, 0))
    degree_minutes_5_cool = np.sum(np.maximum(5 - temperature, 0))
    degree_minutes_6_cool = np.sum(np.maximum(6 - temperature, 0))

    # New statistical properties for the second figure
    iqr_temp = np.percentile(temperature, 75) - np.percentile(temperature, 25)
    range_temp = np.max(temperature) - np.min(temperature)
    max_temp = np.max(temperature)
    min_temp = np.min(temperature)
    entropy_temp = entropy(temperature)
    percentage_above_6 = np.mean(temperature > 6) * 100
    percentage_below_minus_2 = np.mean(temperature < -2) * 100
    hourly_variations = np.mean(np.abs(np.diff(temperature, n=5)))  # Average hourly variations assuming 15-minute intervals
    
    # New statistical properties for the third figure
    autocorrs = [np.corrcoef(temperature[:-i], temperature[i:])[0, 1] for i in range(1, 9)]
    
    return mean_temp, var_temp, std_temp, kurt_temp, skew_temp, avg_gradient, degree_minutes_0, degree_minutes_1, degree_minutes_2, degree_minutes_3, degree_minutes_4, degree_minutes_5,  degree_minutes_6, degree_minutes_0_cool, degree_minutes_1_cool, degree_minutes_2_cool, degree_minutes_3_cool, degree_minutes_4_cool, degree_minutes_5_cool,  degree_minutes_6_cool, iqr_temp, range_temp, max_temp, min_temp, entropy_temp, percentage_above_6, percentage_below_minus_2, hourly_variations, autocorrs

# Compute statistical properties for each group
statistical_results = []
for name, group in df_groups:
    mean_temp, var_temp, std_temp, kurt_temp, skew_temp, avg_gradient, degree_minutes_0, degree_minutes_1, degree_minutes_2, degree_minutes_3, degree_minutes_4, degree_minutes_5, degree_minutes_6, degree_minutes_0_cool, degree_minutes_1_cool, degree_minutes_2_cool, degree_minutes_3_cool, degree_minutes_4_cool, degree_minutes_5_cool,  degree_minutes_6_cool,  iqr_temp, range_temp, max_temp, min_temp, entropy_temp, percentage_above_6, percentage_below_minus_2, hourly_variations, autocorrs = compute_statistical_properties(group)
    label = group['Label'].iloc[0]
    result = {
        'Serial no': name,
        'Mean': mean_temp,
        'Variance': var_temp,
        'Standard Deviation': std_temp,
        'Kurtosis': kurt_temp,
        'Skewness': skew_temp,
        'Average Gradient': avg_gradient,
        'Degree Minutes 0': degree_minutes_0,
        'Degree Minutes 1': degree_minutes_1,
        'Degree Minutes 2': degree_minutes_2,
        'Degree Minutes 3': degree_minutes_3,
        'Degree Minutes 4': degree_minutes_4,
        'Degree Minutes 5': degree_minutes_5,
        'Degree Minutes 6': degree_minutes_6,
        'Degree Minutes 0 cool': degree_minutes_0_cool,
        'Degree Minutes 1 cool': degree_minutes_1_cool,
        'Degree Minutes 2 cool': degree_minutes_2_cool,
        'Degree Minutes 3 cool': degree_minutes_3_cool,
        'Degree Minutes 4 cool': degree_minutes_4_cool,
        'Degree Minutes 5 cool': degree_minutes_5_cool,
        'Degree Minutes 6 cool': degree_minutes_6_cool,
        'IQR': iqr_temp,
        'Range': range_temp,
        'Max Temp': max_temp,
        'Min Temp': min_temp,
        'Entropy': entropy_temp,
        'Percentage Above 6°C': percentage_above_6,
        'Percentage Below -2°C': percentage_below_minus_2,
        'Average Hourly Variations': hourly_variations,
        'Label': label
    }
    for i, autocorr in enumerate(autocorrs, 1):
        result[f'Autocorrelation {i}'] = autocorr
    statistical_results.append(result)

statistical_df = pd.DataFrame(statistical_results)



# Function to perform t-test and Mann-Whitney U test
def perform_stat_tests(stat_df, feature):

    if feature == 'Kurtosis':
        print("Kurtosis")

    normal_data = stat_df[stat_df['Label'] == 'Normal'][feature].dropna()
    anomaly_data = stat_df[stat_df['Label'] == 'Anomaly'][feature].dropna()
    
    # Perform t-test
    t_stat, t_p_value = ttest_ind(normal_data, anomaly_data, equal_var=False)
    
    # Perform Mann-Whitney U test
    u_stat, u_p_value = mannwhitneyu(normal_data, anomaly_data)
    
    return t_stat, t_p_value, u_stat, u_p_value

# Perform statistical tests and create a dictionary to store the results
statistical_tests_results = {}
for feature in ['Mean', 'Variance', 'Standard Deviation', 'Kurtosis', 'Skewness', 'Average Gradient', 'Degree Minutes 0', 'Degree Minutes 1', 'Degree Minutes 2', 'Degree Minutes 3', 'Degree Minutes 4', 'Degree Minutes 5', 'Degree Minutes 6', 'Degree Minutes 0 cool', 'Degree Minutes 1 cool', 'Degree Minutes 2 cool', 'Degree Minutes 3 cool', 'Degree Minutes 4 cool', 'Degree Minutes 5 cool', 'Degree Minutes 6 cool', 'IQR', 'Range', 'Max Temp', 'Min Temp', 'Entropy', 'Percentage Above 6°C', 'Percentage Below -2°C', 'Average Hourly Variations']:
    t_stat, t_p_value, u_stat, u_p_value = perform_stat_tests(statistical_df, feature)
    statistical_tests_results[feature] = [t_p_value, u_p_value]

# Save the results to a JSON file
with open('statistical_tests_results.json', 'w') as f:
    json.dump(statistical_tests_results, f, indent=4)


# Plot the first set of distributions
properties_figure_1 = ['Mean', 'Variance', 'Standard Deviation', 'Kurtosis', 'Skewness', 'Average Gradient', 'Degree Minutes 0', 'Degree Minutes 1', 'Degree Minutes 2']



# Set font properties globally
plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# Define the properties for each set of distributions
properties_figure_2 = ['IQR', 'Entropy', 'Percentage Above 6°C', 'Percentage Below -2°C', 'Average Hourly Variations']
properties_figure_3 = [f'Autocorrelation {i}' for i in range(1, 9)]
