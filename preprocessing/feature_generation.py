import json
import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import SpectralClustering
from statsmodels.tsa.stattools import acf, pacf
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from joblib import load
from datasets.data_merger import ShipmentDataMerger

class FeatureGenerator:
    def __init__(self, data, config_path, coop_path, bama_path, validation=False, new_data=False):
        self.data = data
        self.config_path = config_path
        self.coop_path = coop_path
        self.bama_path = bama_path
        self.mean_temp = None
        self.median_temp = None
        self.norm_0 = None
        self.norm_6 = None
        self.common_frequencies = None
        self.sample_spacing = 15 * 60  # Assuming 15-minute intervals
        self.validation = validation
        self.new_data = new_data

    def load_and_merge_data(self, file_paths, class_names):
        df_list = []
        for file_path, class_name in zip(file_paths, class_names):
            df = pd.read_csv(file_path)
            df['AnomalyClass'] = class_name
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    def remove_breaks(self, group, interval='1h'):
        group = group.sort_values('Date / Time')
        time_diff = group['Date / Time'].diff().dropna()
        return not any(time_diff > pd.Timedelta(interval))

    def remove_trips_with_few_data_points(self, df, min_data_points=10):
        return df.groupby('Serial no').filter(lambda group: len(group) >= min_data_points)

    def norm_median(self, df, feature_col):
        mean_val = np.mean(df[feature_col])
        median_val = np.median(df[feature_col])
        df[feature_col] = (df[feature_col] - mean_val) / median_val
        return df, mean_val, median_val

    def normalize_value(self, value, mean_val, median_val):
        return (value - mean_val) / median_val

    def pad_or_clip(self, series, length):
        if len(series) > length:
            return series[:length]
        elif len(series) < length:
            return np.pad(series, (0, length - len(series)), 'constant')
        else:
            return series

    def create_fixed_length_datasets(self, data, lengths=[200, 300]):
        datasets = {length: [] for length in lengths}
        for serial_no, group in data.groupby("Serial no"):
            temp_data = group['Temperature'].values
            for length in lengths:
                fixed_length_data = self.pad_or_clip(temp_data, length)
                datasets[length].append((serial_no, fixed_length_data))
        return datasets

    def compute_target_frequencies(self):
        start_interval = 30 * 60  # 30 minutes in seconds
        end_interval = 12 * 60 * 60  # 12 hours in seconds
        interval_step = 15 * 60  # 15 minutes in seconds

        time_intervals = np.arange(start_interval, end_interval + interval_step, interval_step)
        frequencies = 1 / time_intervals
        return frequencies

    def create_feature_vectors(self, dataset, common_frequencies, sample_spacing):
        feature_vectors = {}
        for serial_no, series in dataset:
            yf = fft(series)
            xf = fftfreq(len(series), sample_spacing)
            mask = xf >= 0
            xf = xf[mask]
            yf = np.abs(yf[mask])
            amplitudes = []
            for freq in common_frequencies:
                idx = (np.abs(xf - freq)).argmin()
                amplitudes.append(yf[idx])
            feature_vectors[serial_no] = amplitudes
        return feature_vectors

    def compute_statistical_properties(self, group):
        temperature = group['Temperature'].values

        # Existing statistical properties
        mean_temp = np.mean(temperature)
        var_temp = np.var(temperature)
        std_temp = np.std(temperature)
        kurt_temp = kurtosis(temperature)
        skew_temp = skew(temperature)
        avg_gradient = np.mean(np.diff(temperature))

        # Degree minutes calculations
        time_interval = 15  # time interval in minutes
        degree_minutes_0 = np.sum(np.maximum(temperature - self.norm_0, 0)) * time_interval
        degree_minutes_1 = np.sum(np.maximum(temperature - self.normalize_value(1, self.mean_temp, self.median_temp), 0)) * time_interval
        degree_minutes_2 = np.sum(np.maximum(temperature - self.normalize_value(2, self.mean_temp, self.median_temp), 0)) * time_interval
        degree_minutes_3 = np.sum(np.maximum(temperature - self.normalize_value(3, self.mean_temp, self.median_temp), 0)) * time_interval
        degree_minutes_4 = np.sum(np.maximum(temperature - self.normalize_value(4, self.mean_temp, self.median_temp), 0)) * time_interval
        degree_minutes_5 = np.sum(np.maximum(temperature - self.normalize_value(5, self.mean_temp, self.median_temp), 0)) * time_interval
        degree_minutes_6 = np.sum(np.maximum(temperature - self.normalize_value(6, self.mean_temp, self.median_temp), 0)) * time_interval

        # Additional statistical features
        iqr_temp = np.percentile(temperature, 75) - np.percentile(temperature, 25)
        range_temp = np.max(temperature) - np.min(temperature)
        max_temp = np.max(temperature)
        min_temp = np.min(temperature)
        percentage_above_6 = np.mean(temperature > self.norm_6) * 100
        percentage_below_minus_2 = np.mean(temperature < self.normalize_value(-2, self.mean_temp, self.median_temp)) * 100
        hourly_variations = np.mean(np.abs(np.diff(temperature, n=5)))

        # Peaks features
        peaks, _ = find_peaks(temperature)
        num_peaks = len(peaks)
        if num_peaks > 1:
            peak_to_peak_distances = np.mean(np.diff(peaks))
        else:
            peak_to_peak_distances = 0  # Replace NaN with 0

        # Wavelet features
        wavelet_coeffs = pywt.dwt(temperature, 'db1')[0]
        mean_wavelet_coeff = np.mean(wavelet_coeffs)
        std_wavelet_coeff = np.std(wavelet_coeffs)

        # RMS temperature
        rms_temp = np.sqrt(np.mean(temperature ** 2))

        # New features: Initial temperature and rate of decrease
        initial_temp = temperature[0]
        initial_temp_above_6 = 1 if initial_temp > 6 else 0

        # Calculate the rate of decrease in the initial phase (e.g., first 2 hours)
        time_in_minutes = (group['Date / Time'].values - group['Date / Time'].values[0]) / np.timedelta64(1, 'm')
        initial_phase_indices = np.where(time_in_minutes <= 120)[0]
        if len(initial_phase_indices) > 1:
            initial_rate_of_decrease = np.polyfit(time_in_minutes[initial_phase_indices], temperature[initial_phase_indices], 1)[0]
        else:
            initial_rate_of_decrease = 0  # If not enough data points, set to 0


        # Replace NaN values if any
        features = [mean_temp, var_temp, std_temp, kurt_temp, skew_temp, avg_gradient,
                    degree_minutes_0, degree_minutes_1, degree_minutes_2, degree_minutes_3,
                    degree_minutes_4, degree_minutes_5, degree_minutes_6, iqr_temp, range_temp,
                    max_temp, min_temp, percentage_above_6, percentage_below_minus_2,
                    hourly_variations, num_peaks, peak_to_peak_distances, mean_wavelet_coeff,
                    std_wavelet_coeff, rms_temp, initial_temp, initial_rate_of_decrease]
        
        feature_names = ["mean_temp", "var_temp", "std_temp", "kurt_temp", "skew_temp", "avg_gradient",
                 "degree_minutes_0", "degree_minutes_1", "degree_minutes_2", "degree_minutes_3",
                 "degree_minutes_4", "degree_minutes_5", "degree_minutes_6", "iqr_temp", "range_temp",
                 "max_temp", "min_temp", "percentage_above_6", "percentage_below_minus_2",
                 "hourly_variations", "num_peaks", "peak_to_peak_distances", "mean_wavelet_coeff",
                 "std_wavelet_coeff", "rms_temp", "initial_temp", "initial_rate_of_decrease"]

        features = [0 if np.isnan(f) else f for f in features]

        return feature_names, features

    def plot_frequency_features(self, classified_csv_path, label_scheme, length=200):
        plt.rcParams.update({'font.weight': 'bold'})

        # Load the classified CSV file
        classified_df = pd.read_csv(classified_csv_path)
        classified_df['Serial no'] = classified_df['Serial no'].astype(str)

        # Load manually labelled dataset
        file_paths = [
            "data/data_spike.csv",
            "data/data_cool_defrost.csv",
            "data/data_excursion.csv",
            "data/data_not_precooled.csv",
            "data/data_norm.csv",
            "data/data_initial_ramp.csv",
            "data/data_chilling_injury.csv",
            "data/data_extended_drift.csv"
        ]

        class_names = [
            "Spike",
            "Cool Defrost",
            "Excursion",
            "Not Precooled",
            "Normal",
            "Initial Ramp",
            "Top Freezing",
            "Extended Drift"
        ]

        # Load and merge data
        merged_data = self.load_and_merge_data(file_paths, class_names)

        # Create a mapping from Serial no to AnomalyClass
        serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

        # Get the frequency feature vectors
        frequency_features = self.compute_frequency_features()

        # Choose one set of frequency features (either 200 or 300)
        selected_freq_features = frequency_features[length]
        common_frequencies = self.common_frequencies

        # Create a new DataFrame to hold the frequency features and corresponding labels
        feature_data = []
        for serial_no, features in selected_freq_features.items():
            if label_scheme == "2-class":
                label = classified_df[classified_df['Serial no'] == serial_no]['Label'].values[0] if serial_no in classified_df['Serial no'].values else None
            else:
                label = serial_to_anomaly_class[serial_no] if serial_no in serial_to_anomaly_class.keys() else None 

            if label is not None:
                feature_data.append([serial_no, label] + features)

        columns = ['Serial no', 'label'] + [f'Frequency {i+1}' for i in range(len(common_frequencies))]
        features_df = pd.DataFrame(feature_data, columns=columns)

        # Plot the histograms
        num_features = len(common_frequencies)
        fig, axes = plt.subplots(nrows=(num_features + 3) // 4, ncols=4, figsize=(20, 30))
        axes = axes.flatten()

        for i, feature in enumerate(columns[2:]):
            ax = axes[i]
            for label in features_df['label'].unique():
                subset = features_df[features_df['label'] == label]
                ax.hist(subset[feature], bins=20, alpha=0.5, label=label)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.annotate(
                f'{feature}', 
                xy=(0.5, 0.5), 
                xycoords='axes fraction', 
                fontsize=10, 
                fontweight='bold',
                ha='center', 
                va='center', 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
            )
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=4, direction='in')
            if i == 0:
                ax.legend(fontsize=8)

        axes[-1].set_visible(False)

        # Adjust layout and save the figure
        plt.subplots_adjust(top=0.92, bottom=0.5, left=0.05, right=0.95, hspace=1, wspace=1)
        plt.tight_layout()
        plt.savefig(f"results/plots/features/frequency_features_histograms_{label_scheme}_{length}.png", format="png", dpi=300)
        plt.show()

    def plot_hmm_features(self, classified_csv_path, label_scheme, state=2):
        plt.rcParams.update({'font.weight': 'bold'})

        # Load the classified CSV file
        classified_df = pd.read_csv(classified_csv_path)
        classified_df['Serial no'] = classified_df['Serial no'].astype(str)

        # Load manually labelled dataset
        file_paths = [
            "data/data_spike.csv",
            "data/data_cool_defrost.csv",
            "data/data_excursion.csv",
            "data/data_not_precooled.csv",
            "data/data_norm.csv",
            "data/data_initial_ramp.csv",
            "data/data_chilling_injury.csv",
            "data/data_extended_drift.csv"
        ]

        class_names = [
            "Spike",
            "Cool Defrost",
            "Excursion",
            "Not Precooled",
            "Normal",
            "Initial Ramp",
            "Top Freezing",
            "Extended Drift"
        ]

        # Load and merge data
        merged_data = self.load_and_merge_data(file_paths, class_names)

        # Create a mapping from Serial no to AnomalyClass
        serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

        # Get the HMM feature vectors
        hmm_features, _ = self.compute_hmm_features()

        # Choose one set of HMM features (either 2-state or 3-state)
        selected_hmm_features = hmm_features[state]
        hmm_feature_names = selected_hmm_features.columns.tolist()[1:]  # Exclude the 'Serial no' column

        # Create a new DataFrame to hold the HMM features and corresponding labels
        feature_data = []
        for _, row in selected_hmm_features.iterrows():
            serial_no = row['Serial no']
            if label_scheme == "2-class":
                label = classified_df[classified_df['Serial no'] == serial_no]['Label'].values[0] if serial_no in classified_df['Serial no'].values else None
            else:
                label = serial_to_anomaly_class[serial_no] if serial_no in serial_to_anomaly_class.keys() else None 

            if label is not None:
                feature_data.append([serial_no, label] + row[hmm_feature_names].tolist())

        columns = ['Serial no', 'label'] + hmm_feature_names
        features_df = pd.DataFrame(feature_data, columns=columns)

        # Plot the histograms
        num_features = len(hmm_feature_names)
        fig, axes = plt.subplots(nrows=(num_features + 3) // 4, ncols=4, figsize=(20, 30))
        axes = axes.flatten()

        for i, feature in enumerate(hmm_feature_names):
            ax = axes[i]
            for label in features_df['label'].unique():
                subset = features_df[features_df['label'] == label]
                ax.hist(subset[feature], bins=20, alpha=0.5, label=label)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.annotate(
                f'{feature}', 
                xy=(0.5, 0.5), 
                xycoords='axes fraction', 
                fontsize=10, 
                fontweight='bold',
                ha='center', 
                va='center', 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
            )
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=4, direction='in')
            if i == 0:
                ax.legend(fontsize=8)

        axes[-1].set_visible(False)

        # Adjust layout and save the figure
        plt.subplots_adjust(top=0.92, bottom=0.5, left=0.05, right=0.95, hspace=1, wspace=1)
        plt.tight_layout()
        plt.savefig(f"results/plots/features/hmm_features_histograms_{label_scheme}_{state}_state.png", format="png", dpi=300)
        plt.show()

    def load_data(self, df):
        
        shipment_merger = ShipmentDataMerger(self.coop_path, self.bama_path, self.config_path)
        
        if not self.new_data:
            data = shipment_merger.merged_dataframe
            data = shipment_merger.resample_time_series()
            data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
            data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
            data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()
        else:
            self.data = df
            data = df.copy()
            data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
            data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
            data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()
        
        data = data.groupby('Serial no').filter(lambda group: self.remove_breaks(group))
        data = self.remove_trips_with_few_data_points(data)
        normalized_data, self.mean_temp, self.median_temp = self.norm_median(data, feature_col='Temperature')
        self.norm_0 = self.normalize_value(0, self.mean_temp, self.median_temp)
        self.norm_6 = self.normalize_value(6, self.mean_temp, self.median_temp)
        self.common_frequencies = self.compute_target_frequencies()
        return normalized_data

    def compute_statistical_features(self):
        data = self.load_data(df=self.data)
        shipment_groups = data.groupby("Serial no")
        statistical_vectors = {}
        for name, group in shipment_groups:
            names, stats = self.compute_statistical_properties(group)
            statistical_vectors[name] = stats
        return names, statistical_vectors
    
    def plot_statistical_properties(self, classified_csv_path, label_scheme):
        plt.rcParams.update({'font.weight': 'bold'})

        # Load the classified CSV file
        classified_df = pd.read_csv(classified_csv_path)
        
        # Ensure that the Serial no is of the same type as in the features
        classified_df['Serial no'] = classified_df['Serial no'].astype(str)



        # Load manually labelled dataset 
        # File paths and class names
        file_paths = [
            "data/data_spike.csv",
            "data/data_cool_defrost.csv",
            "data/data_excursion.csv",
            "data/data_not_precooled.csv",
            "data/data_norm.csv",
            "data/data_initial_ramp.csv",
            "data/data_chilling_injury.csv",
            "data/data_extended_drift.csv"
        ]

        class_names = [
            "Spike",
            "Cool Defrost",
            "Excursion",
            "Not Precooled",
            "Normal",
            "Initial Ramp",
            "Top Freezing",
            "Extended Drift"
        ]

        # Load and merge data
        merged_data = self.load_and_merge_data(file_paths, class_names)

        # Create a mapping from Serial no to AnomalyClass
        serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

        # Get the statistical feature names and vectors
        stat_feature_names, statistical_features = self.compute_statistical_features()

        stat_feature_names = ["mean", "variance", "std", "kurtosis", "skewness", "gradient",
                 "degree minutes (0°C)", "degree minutes 1°C", "degree minutes 2°C", "degree_minutes 3°C",
                 "degree minutes 4°C", "degree minutes 5°C", "degree minutes 6°C", "iqr", "range",
                 "max", "min", "(%) above 6°C", " (%) below -2°C",
                 "mean var (hr)", "peak count", "peak to peak_distance", "mean wavelet coeff",
                 "std wavelet coeff", "rms", "initial temp", "initial rate of decrease"]

        # Create a new DataFrame to hold the statistical features and corresponding labels
        feature_data = []
        for serial_no, features in statistical_features.items():
            if label_scheme == "2-class":
                label = classified_df[classified_df['Serial no'] == serial_no]['Label'].values[0] if serial_no in classified_df['Serial no'].values else None
            else:
                label = serial_to_anomaly_class[serial_no] if serial_no in serial_to_anomaly_class.keys() else None 

            if label is not None:
                feature_data.append([serial_no, label] + features)

        columns = ['Serial no', 'label'] + stat_feature_names
        features_df = pd.DataFrame(feature_data, columns=columns)

        # Plot the histograms
        num_features = len(stat_feature_names)
        fig, axes = plt.subplots(nrows=(num_features + 3) // 4, ncols=4, figsize=(20, 30))
        axes = axes.flatten()

        for i, feature in enumerate(stat_feature_names):
            ax = axes[i]
            for label in features_df['label'].unique():
                subset = features_df[features_df['label'] == label]
                ax.hist(subset[feature], bins=20, alpha=0.5, label=label)
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            # ax.set_title(f'{feature} per Class', fontsize=4)

            # Annotate the title in the center of the subplot
            ax.annotate(
                f'{feature}', 
                xy=(0.5, 0.5), 
                xycoords='axes fraction', 
                fontsize=14, 
                fontweight='bold',
                ha='center', 
                va='center', 
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
            )
            ax.grid(False)
            ax.tick_params(axis='both', which='major', labelsize=8, direction='in')
            # ax.legend(['Cyclical pattern', 'Normal', 'Not pre-cooled', 'Excursion', 'Spikes', 'Extended drift', 'Initial ramp up', 'Below freezing'])
            if i == 0:
                ax.legend(fontsize=8)

        axes[-1].set_visible(False)

        # Adjust layout and save the figure
        # fig.suptitle('Histograms of Data Distribution for Statistical Properties by Class Label', fontsize=16, fontweight='bold', y=1.02)
        plt.subplots_adjust(top=0.92, bottom=0.5, left=0.05, right=0.95, hspace=1, wspace=1)
        plt.tight_layout()
        plt.savefig(f"results/plots/features/statistical_properties_histograms_stringent_{label_scheme}.png", format="png", dpi=300)
        plt.show()

    def compute_frequency_features(self):
        data = self.load_data(df=self.data)
        fixed_length_datasets = self.create_fixed_length_datasets(data)
        freq_feature_vectors_200 = self.create_feature_vectors(fixed_length_datasets[200], self.common_frequencies, self.sample_spacing)
        freq_feature_vectors_300 = self.create_feature_vectors(fixed_length_datasets[300], self.common_frequencies, self.sample_spacing)
        return {200: freq_feature_vectors_200, 300: freq_feature_vectors_300}


    def compute_hmm_features(self):
        data = self.load_data(df=self.data)
        model_path_2_state = "models/hmm/best_hmm_model_22-06-24_2_state.joblib"
        model_path_3_state = "models/hmm/best_hmm_model_22-06-24_3_state.joblib"
        model_path_8_state = "models/hmm/best_hmm_model_06-07-24_8_state.joblib"
        model_2_state = load(model_path_2_state)
        model_3_state = load(model_path_3_state)
        model_8_state = load(model_path_8_state)

        hidden_states_list = []
        for serial_no, group in data.groupby('Serial no'):
            group = group.sort_values('Date / Time')
            observations_array = group[['Temperature']].to_numpy()
            hidden_states_2 = model_2_state.predict(observations_array)
            hidden_states_3 = model_3_state.predict(observations_array)
            hidden_states_8 = model_8_state.predict(observations_array)
            group['Hidden States 2'] = hidden_states_2
            group['Hidden States 3'] = hidden_states_3
            group['Hidden States 8'] = hidden_states_8
            group['Log-Likelihood 2'] = model_2_state.score(observations_array)
            group['Log-Likelihood 3'] = model_3_state.score(observations_array)
            group['Log-Likelihood 8'] = model_8_state.score(observations_array)
            hidden_states_list.append(group)

        labeled_data = pd.concat(hidden_states_list)
        return self.extract_hmm_features(labeled_data), labeled_data

    def extract_hmm_features(self, labeled_data):
        num_states = [2, 3, 8]
        state_features = []
        transition_features = []
        serial_numbers = []
        num_transitions = []
        log_likelihoods = []

        for state in num_states:
            serial_numbers_buffer = []
            transitions_buffer = []
            state_features_buffer = []
            transition_matrix_buffer = []
            log_likelihoods_buffer = []

            for serial_no, group in labeled_data.groupby('Serial no'):
                hidden_states = group[f'Hidden States {state}'].values

                state_counts = np.bincount(hidden_states, minlength=state)
                state_freq = state_counts / np.sum(state_counts)
                state_features_buffer.append(state_freq)

                transition_matrix = np.zeros((state, state))
                transitions = 0
                for current, next_ in zip(hidden_states[:-1], hidden_states[1:]):
                    if current != next_:
                        transitions += 1
                    transition_matrix[current, next_] += 1
                total_transitions = np.sum(transition_matrix)
                if total_transitions > 0:
                    transition_matrix /= total_transitions

                transition_matrix_buffer.append(transition_matrix.flatten())
                serial_numbers_buffer.append(serial_no)
                transitions_buffer.append(transitions / len(hidden_states))  # Normalize by length of the time series
                log_likelihoods_buffer.append(group[f'Log-Likelihood {state}'].iloc[0])

            num_transitions.append(transitions_buffer)
            transition_features.append(transition_matrix_buffer)
            serial_numbers.append(serial_numbers_buffer)
            state_features.append(state_features_buffer)
            log_likelihoods.append(log_likelihoods_buffer)

        features_2_state = self.create_hmm_feature_df(state_features[0], transition_features[0], num_transitions[0], serial_numbers[0], log_likelihoods[0], 2)
        features_3_state = self.create_hmm_feature_df(state_features[1], transition_features[1], num_transitions[1], serial_numbers[1], log_likelihoods[1], 3)
        features_8_state = self.create_hmm_feature_df(state_features[2], transition_features[2], num_transitions[2], serial_numbers[2], log_likelihoods[2], 8)
        hmm_features = {2: features_2_state, 3: features_3_state, 8: features_8_state}
        return hmm_features

    def create_hmm_feature_df(self, state_features, transition_features, num_transitions, serial_numbers, log_likelihoods, num_states):
        state_columns = [f'State_{i}_Freq' for i in range(num_states)]
        transition_columns = [f'Transition_{i}_{j}_Prob' for i in range(num_states) for j in range(num_states)]
        state_features_df = pd.DataFrame(state_features, columns=state_columns)
        transition_features_df = pd.DataFrame(transition_features, columns=transition_columns)
        transitions_df = pd.DataFrame(num_transitions, columns=['Number of Transitions'])
        log_likelihoods_df = pd.DataFrame(log_likelihoods, columns=['Log-Likelihood'])
        feature_df = pd.concat([pd.DataFrame({'Serial no': serial_numbers}), state_features_df, transition_features_df, transitions_df, log_likelihoods_df], axis=1)
        return feature_df
    
    def compute_dtw_features(self):

        # Load ordered serial numbers for affinity matrix
        json_file_path = 'serial_numbers.json'
        with open(json_file_path, 'r') as json_file:
            serial_nums = json.load(json_file)

        # Load distance matrix and filter invalid values
        distance_matrix = np.load("/Users/divinefavourodion/results/dtw_distance_matrix.npy")
        ordered_indices = [serial_nums.index(serial) for serial in serial_nums]
        ordered_distance_matrix = distance_matrix[np.ix_(ordered_indices, ordered_indices)]
        rows_with_inf = np.isinf(ordered_distance_matrix).any(axis=1)
        filtered_distance_matrix = ordered_distance_matrix[~rows_with_inf]
        large_number = np.nanmax(filtered_distance_matrix) * 10
        filtered_distance_matrix[np.isnan(filtered_distance_matrix)] = large_number

        # Convert distance matrix to affinity matrix
        beta = 1
        distance_matrix_std = np.nanstd(filtered_distance_matrix)
        if distance_matrix_std == 0:
            distance_matrix_std = 1
        affinity_matrix = np.exp(-beta * filtered_distance_matrix / distance_matrix_std)

        dtw_features = {serial: {'Affinity': 0, 'Distance': 0}for serial in serial_nums}
        
        for i, serial in enumerate(serial_nums):
            dtw_features[serial]['Affinity'] = affinity_matrix[i, :]

        for i, serial in enumerate(serial_nums):
            dtw_features[serial]['Distance'] = ordered_distance_matrix[i, :]

        return dtw_features, affinity_matrix, filtered_distance_matrix
    
    def compute_ae_latent_features(self):
        latent_features = None

        # Load the latent features from FC_Autoencoder
        latent_features_filepath = "/Users/divinefavourodion/Documents/Sensitech-monitoring/data/features/latent/cnn_ae_latent.json"

        with open(latent_features_filepath, 'r') as file:
            latent_features = json.load(file)

       
        return latent_features
    
    def compute_deep_latent_features(self):
        latent_features = None

        # Load the latent features from FC_Autoencoder
        latent_features_filepath = "/Users/divinefavourodion/Documents/Sensitech-monitoring/data/features/latent/timer_embeddings.json"

        with open(latent_features_filepath, 'r') as file:
            latent_features = json.load(file)

       
        return latent_features
    
    def normalize_features(self, features):
        
        combined_features = []
        serial_numbers = []

        for serial_no, feature_dict in features.items():
            combined_features.append(feature_dict['combined'])
            serial_numbers.append(serial_no)
        
        combined_features = np.array(combined_features)
        # normalized_combined_features = scaler.fit_transform(combined_features)
        normalized_combined_features = (combined_features - np.mean(combined_features)) / np.median(combined_features)
        
        normalized_features = {}
        for idx, serial_no in enumerate(serial_numbers):
            normalized_features[serial_no] = {
                'statistical': features[serial_no]['statistical'],
                'frequency_200': features[serial_no]['frequency_200'],
                'frequency_300': features[serial_no]['frequency_300'],
                'ae_latent_features': features[serial_no]['ae_latent_features'],
                'deep_latent_features': features[serial_no]['deep_latent_features'],
                'combined': normalized_combined_features[idx]
            }
        return normalized_features

    def extract_all_features(self):
        stat_feature_names, statistical_features = self.compute_statistical_features()
        frequency_features = self.compute_frequency_features()
        # hmm_features, _ = self.compute_hmm_features()
        affinity_matrix, distance_matrix = None,None
        ae_latent_features = self.compute_ae_latent_features()
        deep_latent_features = self.compute_deep_latent_features()

        all_features = {}
        for serial_no in set(self.data['Serial no']):
            stats = statistical_features.get(serial_no, [np.nan] * 20)
            freq_200 = frequency_features[200].get(serial_no, [np.nan] * len(self.common_frequencies))
            freq_300 = frequency_features[300].get(serial_no, [np.nan] * len(self.common_frequencies))
            ae_latent =  ae_latent_features.get(serial_no, [np.nan] * 128)
            deep_latent =  deep_latent_features.get(serial_no, [np.nan] * 256)

            try:
                deep_latent = [item[0] for item in deep_latent] # Convert from list of lists to single list
            except:
                pass
            
            all_features[serial_no] = {
                'statistical': stats,
                'frequency_200': freq_200,
                'frequency_300': freq_300,
                'ae_latent_features': ae_latent,
                'deep_latent_features': deep_latent,
                'combined': np.concatenate([stats, freq_200, freq_300])
            }
        return all_features, affinity_matrix, distance_matrix, stat_feature_names

if __name__ == '__main__':
    config_path = 'config.ini'
    coop_path = "data/all_data_combined_meta.csv"
    bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"
    shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
    data = shipment_merger.merged_dataframe
    data = shipment_merger.resample_time_series()
    data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
    data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()
    feat_generator = FeatureGenerator(data=data, config_path=config_path, coop_path=coop_path, bama_path=bama_path)
    feat_generator.data = feat_generator.load_data()
    model_features, affinity_matrix, distance_matrix, feat_names = feat_generator.extract_all_features()
    feat_generator.plot_statistical_properties('data/classified/classified_data_2_classes_stringent.csv', label_scheme="2-class")
    feat_generator.plot_frequency_features('data/classified/classified_data_2_classes_stringent.csv', label_scheme="2-class", length=200)
    feat_generator.plot_frequency_features('data/classified/classified_data_2_classes_stringent.csv', label_scheme="2-class", length=300)
    
    # Normalize the feature vectors
    normalized_model_features = feat_generator.normalize_features(model_features)

    print("Done")
