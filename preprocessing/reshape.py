import pandas as pd
import numpy as np

class TimeSeriesReshaper:
    def __init__(self, df, date_col, serial_col, temp_col):
        self.df = df
        self.date_col = date_col
        self.serial_col = serial_col
        self.temp_col = temp_col

    def reshape(self, num_points=300):
        # Group by the serial number
        grouped = self.df.groupby(self.serial_col)

        # Initialize a list to collect padded sequences
        padded_data = []

        # Iterate through each group, pad sequences, and append to the list
        for serial_no, group in grouped:
            # Sort by date to ensure correct order
            group = group.sort_values(by=self.date_col)

            # Extract the relevant columns and clip or pad the sequences
            temperature = group[self.temp_col].values

            if len(temperature) > num_points:
                # Clip to the first `num_points` if longer
                clipped_temperature = temperature[:num_points]
            else:
                # Pad with zeros if shorter
                clipped_temperature = np.pad(temperature, (0, num_points - len(temperature)), 'constant', constant_values=0)

            # Create a dictionary for the padded data
            padded_dict = {
                self.serial_col: serial_no,
                **{f'{self.temp_col}_{i}': temp for i, temp in enumerate(clipped_temperature)},
            }

            padded_data.append(padded_dict)

        # Convert the list of dictionaries to a DataFrame
        padded_df = pd.DataFrame(padded_data)

        return padded_df

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('path_to_your_dataset.csv')

    # Create an instance of the class
    reshaper = TimeSeriesReshaper(df, 'Date / Time', 'Serial no', 'Temperature', 'Relative Time', 'Temperature Gradient')

    # Get the reshaped DataFrame
    reshaped_df = reshaper.reshape()

    # Save the reshaped DataFrame to a new CSV file
    reshaped_df.to_csv('reshaped_dataset.csv', index=False)

    # Print the reshaped DataFrame
    print(reshaped_df.head())
