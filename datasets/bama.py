import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class BamaData:
    """
    A class to handle BAMA shipment data that includes temperature and light measurements.
    
    Attributes:
        data (DataFrame): The loaded data from the CSV file.
        preprocessing_steps (list): A list of steps applied during preprocessing.
    """
    
    def __init__(self, csv_file_path, feature_list, sensor_type):
        """
        Initialize the SensorData object by loading and inspecting the data.

        :param csv_file_path: The path to the CSV file to be loaded.
        :param feature_list: List of features to be loaded from the CSV.
        :param sensor_type: Type of sensor data to process ('temperature' or 'light').
        """
        self.sensor_type = sensor_type
        self.data = self._load_data(csv_file_path, feature_list)
        self.preprocessing_steps = []
        self._add_step("Loaded CSV and checked data structure")

        self.filter_by_sensor_type(sensor_type)
        self.convert_data_types()
        self.convert_temperature_to_celsius()
        self.handle_missing_data(important_columns=['Date / Time', 'PointValue'])
        self.align_time_series()
        self.remove_duplicates(unique_columns=['SerialNumber', 'Date / Time'])
        self.calculate_relative_time()
        self.normalize_and_standardize('PointValue')

    def _load_data(self, csv_file_path, feature_list):
        """
        Load data from a CSV file and perform basic inspection.

        :param csv_file_path: The path to the CSV file to be loaded.
        :param feature_list: List of features to be loaded from the CSV.
        :return: A pandas DataFrame containing the loaded data.
        """
        data = pd.read_csv(csv_file_path, usecols=feature_list, encoding='utf-8')
        return data

    def filter_by_sensor_type(self, sensor_type):
        """
        Filter the data by sensor type (e.g., temperature or light).

        :param sensor_type: The type of sensor data to filter ('temperature' or 'light').
        """
        if sensor_type.lower() not in ["temperature", "light"]:
            raise ValueError("sensor_type must be 'temperature' or 'light'.")

        self.data = self.data[self.data["SensorType"].str.lower() == sensor_type.lower()]
        self._add_step(f"Filtered data by sensor type: {sensor_type}")

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
        self.data['Date / Time'] = pd.to_datetime(self.data['CreatedOn'], errors='coerce')
        self.data['PointValue'] = pd.to_numeric(self.data['PointValue'], errors='coerce')
        self.data['ActualArrivalTime'] = pd.to_datetime(self.data['ActualArrivalTime'])
        self.data['ActualDepartureTime'] = pd.to_datetime(self.data['ActualDepartureTime'])
        self.data['Full Trip Duration'] = self.data['ActualArrivalTime'] - self.data['ActualDepartureTime']
        self._add_step("Converted data types to datetime and numeric")

    def align_time_series(self):
        """
        Sort the data by timestamp to ensure chronological order.
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

    def convert_temperature_to_celsius(self):
        if self.sensor_type.lower() == 'temperature':
            self.data['PointValue'] = (self.data['PointValue'] - 32) * (5.0 / 9.0)
            self._add_step("Converted temperature from Fahrenheit to Celsius")

    def calculate_relative_time(self):
        """
        Calculate relative time in hours since the first timestamp.
        """
        self.data['Relative Time'] = (
            self.data['Date / Time'] - self.data['Date / Time'].min()
        ).dt.total_seconds() / 3600
        self._add_step("Calculated relative time in hours")

    def normalize_and_standardize(self, column_name):
        """
        Normalize and standardize a specified column using StandardScaler.

        :param column_name: The name of the column to normalize.
        """
        scaler = StandardScaler()
        self.data[f'Normalized {column_name}'] = scaler.fit_transform(self.data[[column_name]])
        self._add_step(f"Normalized and standardized {column_name}")

    def _add_step(self, step):
        """
        Add a preprocessing step to the list of steps.

        :param step: The step to add to the list.
        """
        self.preprocessing_steps.append(step)
