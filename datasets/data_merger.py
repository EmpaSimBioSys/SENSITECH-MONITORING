import pandas as pd
import numpy as np
import configparser
from datasets.coop import CoopData
from datasets.bama import BamaData
from preprocessing.trim import TimeSeriesTrimmer

class ShipmentDataMerger:
    def __init__(self, coop_path, bama_path, config_path, trim=False, remove_short=False):
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='utf-8')
        self.coop_path = coop_path
        self.bama_path = bama_path
        self.coop_berries_path = 'data/SWP_COOP_Sensor_Shipment_berries.csv'
        self.merged_dataframe = self.load_and_merge_data(trim=trim)

        if remove_short:
            self.merged_dataframe = self.remove_short_shipments(self.merged_dataframe, min_duration_hours=20)

    def load_and_merge_data(self, trim=False):
        # Load datasets
        coop_features = self.config["dataset"]["coop_features"].split(", ")
        bama_features = self.config["dataset"]["bama_features"].split(", ")
        validation_features = self.config["dataset"]["validation_features"].split(",")

        coop_shipments = CoopData(self.coop_path, feature_list=coop_features)
        bama_shipments = BamaData(self.bama_path, feature_list=bama_features, sensor_type="Temperature")

        if trim:
            # Trim time series
            trimmer_coop = TimeSeriesTrimmer(coop_shipments.data, temperature_column='Sensor 1: Ambient Temperature (° C)')
            coop_shipments = trimmer_coop.trim_time_series(column="Serial no")

            trimmer_bama = TimeSeriesTrimmer(bama_shipments.data, temperature_column='PointValue')
            bama_shipments = trimmer_bama.trim_time_series(column="SerialNumber")

            coop_shipments = coop_shipments.copy()
            bama_shipments = bama_shipments.copy()
        else:
            coop_shipments = coop_shipments.data.copy()
            bama_shipments = bama_shipments.data.copy()

        # Rename columns to standardize
        coop_shipments.rename(columns={'Sensor 1: Ambient Temperature (° C)': 'Temperature'}, inplace=True)
        bama_shipments.rename(columns={'PointValue': 'Temperature', 'SerialNumber': 'Serial no'}, inplace=True)

        # Add empty columns for Coop shipments
        for col in ['H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown']:
            coop_shipments[col] = np.nan

        # Columns to keep
        bama_shipments = bama_shipments[['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown']]
        coop_shipments = coop_shipments[['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown']]

        # Combine dataframes
        combined_data = pd.concat([coop_shipments, bama_shipments], ignore_index=True)

        return combined_data

    def remove_short_shipments(self, df, min_duration_hours=20):
        """Remove all shipments that are less than the specified minimum duration in hours."""
        df['Date / Time'] = pd.to_datetime(df['Date / Time'], utc=True)
        shipment_duration = df.groupby('Serial no')['Date / Time'].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600)
        valid_shipments = shipment_duration[shipment_duration >= min_duration_hours].index
        print(f"Removed {1256 - len(valid_shipments)} shipments of < 20 hours")
        return df[df['Serial no'].isin(valid_shipments)]

    def remove_shipments_with_breaks(self, df, interval='30T'):
        """Remove shipments that have breaks in the temperature signal time series for the specified time period."""
        df['Date / Time'] = pd.to_datetime(df['Date / Time'], utc=True)
        break_counts = []
        
        def check_for_breaks(group):
            time_diff = group['Date / Time'].diff().dropna()
            if any(time_diff > pd.Timedelta(interval)):
                break_counts.append(group['Serial no'].unique()[0])
                return False
            return True

        df_groups = df.groupby('Serial no')
        valid_groups = {name: group for name, group in df_groups if check_for_breaks(group)}
        print(f"Removed {1256 - len(valid_groups)} shipments with {interval} minutes breaks")
        df_cleaned = pd.concat(valid_groups.values(), ignore_index=True)
        
        return df_cleaned
    
    def resample_time_series(self, interval='15min', fill_method='ffill'):
        """Resample each time series to a specified interval."""
        resampled_data = []

        def resample_group(group):
            group = group.set_index('Date / Time')

            # Calculate original intervals
            original_intervals = group.index.to_series().diff().dropna().unique()
            original_interval_mean = pd.Series(original_intervals).mean()
            new_interval = pd.Timedelta(interval)

            # Resample and interpolate
            resampled_group = group.resample(interval).mean(numeric_only=True)
            resampled_group['Temperature'] = resampled_group['Temperature'].interpolate(method='linear')

            # Check for NaNs
            if resampled_group.isna().any().any():
                print("Warning: NaNs found after interpolation. These will be forward-filled.")

                if fill_method == 'ffill':
                    resampled_group = resampled_group.fillna(method='ffill')
                elif fill_method == 'linear':
                    resampled_group.interpolate(method='linear')

            # Restore non-numeric columns
            resampled_group['Serial no'] = group['Serial no'].iloc[0]
            resampled_group['H_ShipmentId'] = group['H_ShipmentId'].iloc[0]
            resampled_group['OriginCityorTown'] = group['OriginCityorTown'].iloc[0]
            resampled_group['DestinationCityorTown'] = group['DestinationCityorTown'].iloc[0]
            
            resampled_data.append(resampled_group.reset_index())

        self.merged_dataframe.groupby('Serial no').apply(resample_group)

        self.merged_dataframe = pd.concat(resampled_data, ignore_index=True).infer_objects()
        return self.merged_dataframe



