import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class CoopData:
    """
    A class containing coop real-time sensor data.
    
    Attributes:
        data (DataFrame): The loaded data from the CSV file.
        preprocessing_steps (list): A list of steps applied during preprocessing.
    """
    
    def __init__(self, csv_file_path, feature_list, dependent_var='Sensor 1: Ambient Temperature (° C)'):
        """
        Initialize the CoopData object by loading and inspecting the data.

        :param csv_file_path: The path to the CSV file to be loaded.
        """
        self.data = self._load_data(csv_file_path, feature_list)
        self.dependent_var = dependent_var
        self.preprocessing_steps = []
        self._add_step("Loaded CSV and checked data structure")
        self.convert_data_types()
        self.handle_missing_data(important_columns=[])
        self.align_time_series()
        self.remove_duplicates(unique_columns=['Serial no', 'Date / Time'])
        self.calculate_relative_time()
        [self.generate_features(group) for name, group in self.data.groupby('Serial no')]

    def _load_data(self, csv_file_path, features):
        """
        Load data from a CSV file and perform basic inspection.

        :param csv_file_path: The path to the CSV file to be loaded.
        :return: A pandas DataFrame containing the loaded data.
        """
        data = pd.read_csv(csv_file_path, usecols=features, encoding='utf-8')
        return data

    def handle_missing_data(self, important_columns):
        """
        Handle missing data by dropping rows with NaNs in important columns.

        :param important_columns: A list of key columns to check for NaNs.
        """
        self.data.dropna(subset=important_columns, inplace=True)
        self._add_step("Dropped rows with NaNs in key columns")

    def convert_data_types(self):
        """
        Convert data types, including timestamps and numeric fields.
        """
        self.data['Date / Time'] = pd.to_datetime(self.data['Date / Time'], errors='coerce')
        self.data[self.dependent_var] = pd.to_numeric(
            self.data[self.dependent_var], errors='coerce'
        )
        self._add_step("Converted timestamps to datetime and ensured temperature data is numeric")

    def align_time_series(self):
        """
        Sort data by timestamp to ensure chronological order.
        """
        self.data.sort_values(by='Date / Time', inplace=True)
        self._add_step("Sorted data by timestamp")

    def remove_duplicates(self, unique_columns):
        """
        Remove duplicate rows based on unique columns.

        :param unique_columns: A list of columns used to identify duplicates.
        """
        self.data.drop_duplicates(subset=unique_columns, inplace=True)
        self._add_step("Removed duplicates based on unique identifiers")

    def calculate_relative_time(self):
        """
        Calculate relative time in hours since the first timestamp.
        """
        self.data['Relative Time'] = (
            self.data['Date / Time'] - self.data['Date / Time'].min()
        ).dt.total_seconds() / 3600
        self._add_step("Calculated relative time in hours")

    def generate_features(self, group):
        """
        Generate derived features for each group.
        
        :param group: The group of data to process.
        """
        # Step 1: Compute Temperature Gradients
        group["Temperature Gradient"] = group[self.dependent_var].diff().bfill()  # Temperature change between consecutive readings
        group["Temperature Gradient"] = group["Temperature Gradient"].fillna(0)

        # Step 2: Rolling Statistics
        window_size = 5  # Rolling window size
        group["Rolling_5_Mean"] = group[self.dependent_var].rolling(window=window_size).mean().bfill()
        group["Rolling_5_Std"] = group[self.dependent_var].rolling(window=window_size).std().bfill()

        # Step 3: Time-based Features
        group["Hour of the Day"] = group["Date / Time"].dt.hour  # Extract hour from timestamp
        group["Day of the Week"] = group["Date / Time"].dt.dayofweek  # Day of the week

        # Step 4: Lagged Features
        lag = 1  # Lag amount (adjust as needed)
        group["Lagged Temperature"] = group[self.dependent_var].shift(lag).bfill()

        # Step 5: Cumulative Features
        group["Cumulative Temp Change"] = group["Temperature Gradient"].cumsum()  # Cumulative temperature change
        
        self._add_step("Generated derived features")
        
        return group

    def normalize_and_standardize(self, column_name):
        """
        Normalize and standardize a specified column using StandardScaler.

        :param column_name: The name of the column to normalize.
        """
        scaler = StandardScaler()
        self.data[f'Normalized {column_name}'] = scaler.fit_transform(self.data[[column_name]])
        self._add_step(f"Normalized and standardized {column_name}")

    def detect_outliers(self, column, threshold=3):
        """
        Detect outliers in a specified column based on a given threshold.

        :param column: The column to check for outliers.
        :param threshold: The standard deviation threshold for outliers.
        :return: A DataFrame containing detected outliers.
        """
        mean = self.data[column].mean()
        std_dev = self.data[column].std()
        outlier_condition = np.abs(self.data[column] - mean) > threshold * std_dev
        outliers = self.data[outlier_condition]
        self._add_step(f"Detected outliers in {column} using {threshold} SD")
        return outliers

    def _add_step(self, step):
        """
        Add a preprocessing step to the list of steps.

        :param step: The step to add to the list.
        """
        self.preprocessing_steps.append(step)
