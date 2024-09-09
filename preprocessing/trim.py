import numpy as np
import pandas as pd

class TimeSeriesTrimmer:
    def __init__(self, df, temperature_column):
        self.df = df
        self.temperature_column = temperature_column

    def calculate_gradient(self, group):
        """
        Calculate the gradient of the temperature column for a group.
        """
        if len(group) > 10:
            group['Temperature Gradient'] = np.gradient(group[self.temperature_column])
        else:
            group['Temperature Gradient'] = 0
        return group

    def find_stabilization_point(self, group, stability_threshold=0.05, side="ramp-up"):
        """
        Find the point where the rate of change stabilizes based on the gradient within a group.
        """
        stable_points = np.where(np.abs(group['Temperature Gradient']) < stability_threshold)[0]
        if len(stable_points) > 0:
            if side == "pre-cooling":
                return stable_points[0]
            elif side == "ramp-up":
                return stable_points[-15] if len(stable_points) > 15 else None
        return None

    def trim_time_series(self, side="ramp-up", column="Serial no"):
        """
        Apply gradient calculation and trimming to each group.
        """
        # Apply the calculate_gradient function to each group
        self.df = self.df.groupby(column, as_index=False).apply(self.calculate_gradient)

        # Apply trimming based on the stabilization point
        def trim_group(group):
            stabilization_index = self.find_stabilization_point(group, side=side)
            if stabilization_index is not None:
                if side == "pre-cooling":
                    return group.iloc[stabilization_index:]
                else:
                    return group.iloc[:stabilization_index]
            return group

        return self.df.groupby(column, as_index=False).apply(trim_group)

# Usage Example
if __name__ == "__main__":
    # Example DataFrame with multiple serial numbers
    data = {
        'Serial no': ['A']*50 + ['B']*50,
        'Time': pd.date_range(start='1/1/2020', periods=100, freq='H'),
        'Temperature': np.random.normal(loc=20, scale=2, size=100)  # Simulated temperature data
    }
    df = pd.DataFrame(data)

    # Create an instance of the TimeSeriesTrimmer
    trimmer = TimeSeriesTrimmer(df, 'Temperature')

    # Trim the time series based on pre-cooling stabilization
    trimmed_df = trimmer.trim_time_series(side="pre-cooling")
    print("Trimmed DataFrame (Pre-cooling):", trimmed_df)

    print("Trimmed DataFrame for Serial A (Pre-cooling):", trimmed_df[trimmed_df['Serial no'] == 'A'])
    print("Trimmed DataFrame for Serial B (Pre-cooling):", trimmed_df[trimmed_df['Serial no'] == 'B'])
