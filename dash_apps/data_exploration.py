import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from scipy.signal import stft
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# For proper parsing of the dataframe dtypes
dtype_dict = {
    'column_33': 'int',
    'column_34': 'str',
    'column_43': 'str',
    'column_61': 'str',
}


# Data preprocessing
shipments = pd.read_csv('data/SWP_BAMA_Sensor_Shipment_berries.csv', dtype=dtype_dict, low_memory=False)
shipments['Date / Time'] = pd.to_datetime(shipments['CreatedOn'])
shipments['ActualArrivalTime'] = pd.to_datetime(shipments['ActualArrivalTime'])
shipments['ActualDepartureTime'] = pd.to_datetime(shipments['ActualDepartureTime'])
shipments['Full Trip Duration'] = shipments['ActualArrivalTime'] - shipments['ActualDepartureTime']

# Filter out trips shorter than a day
print(f"The are {len(shipments.groupby("SerialNumber"))} shipments and {len(shipments)} number of rows in the original dataset")
shipments = shipments[shipments['Full Trip Duration'].dt.total_seconds() >= 86400]
print(f"The are {len(shipments.groupby("SerialNumber"))} shipments and {len(shipments)} number of rows in the filtered dataset")

relative_time_hours = lambda group: (group - group.min()).dt.total_seconds() / 3600
shipments['Relative Time'] = shipments.groupby("H_ProgramId")['ActualDepartureTime'].transform(relative_time_hours)
shipments = shipments[shipments['SensorType'] == 'Temperature']
shipments['PointValue'] = (shipments['PointValue'] - 32) * (5/9)
shipments = shipments[['Date / Time', 'Relative Time',  'SerialNumber', 'SensorType', 'ActualArrivalTime', 'ActualDepartureTime', 'ProductName', 'OriginLocationName', 'DestinationLocationName',  'Full Trip Duration', 'CreatedOn', 'PointValue']]

df_for_clustering_trip = None
numeric_cols = ['CreatedOn', 'SerialNumber', 'ActualDepartureTime', 'ActualArrivalTime']

def calculate_gradient(df, temperature_column):
    # Calculate the gradient of the temperature
    df['Temperature Gradient'] = np.gradient(df[temperature_column])
    return df


def find_stabilization_point(df, gradient_column, stability_threshold=0.05, side="pre-cooling"):
    # Find the point where the rate of change stabilizes
    stable_points = np.where(np.abs(df[gradient_column]) < stability_threshold)[0]
    if len(stable_points) > 0:
        # Return the index of the first stable point

        if side == "pre-cooling":
            return stable_points[0]
        elif side == "ramp-up":
            return stable_points[-15]
    else:
        # If no stabilization found, return None
        return None


def trim_time_series(df, stabilization_index, side="pre-cooling"):
    if stabilization_index is not None:
        if side == "pre-cooling":
            # Trim the DataFrame up to the stabilization point
            return df.iloc[stabilization_index:]
        # Trim the DataFrame from the stabilization point
        elif side == "ramp-up":
            return df.iloc[:stabilization_index]
    else:
        # If no stabilization point is found, return the original DataFrame
        return df


app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Img(src='/assets/empa.png', style={'height': '150px', 'width': 'auto'}),
    ], style={'textAlign': 'left'}),
    dcc.RadioItems(
        id='toggle-section',
        options=[
            {'label': 'Individual Shipment Analysis', 'value': 'section1'},
            {'label': 'Multiple Shipment Comparison', 'value': 'section2'}
        ],
        value='section1'
    ),
    html.Button('Select All Shipments', id='select-all-button', n_clicks=0),
    html.Div(id='section-content')
])


# Function to select all shipments
@app.callback(
    Output('dropdown-serial-multi', 'value'),
    [Input('select-all-button', 'n_clicks')],
    [State('dropdown-serial-multi', 'options')]
)
def select_all_shipments(n_clicks, options):
    if n_clicks > 0:
        # Extract the value field from all options to select all
        all_values = [option['value'] for option in options]
        return all_values
    raise dash.exceptions.PreventUpdate  # Prevent update for the first load or if the button wasn't clicked


# Callback to toggle sections
@app.callback(
    Output('section-content', 'children'),
    [Input('toggle-section', 'value')]
)
def toggle_section(selected_section):
    if selected_section == 'section1':
        return html.Div([
            dcc.Dropdown(
                id='dropdown-serial',
                options=[{'label': serial, 'value': serial} for serial in shipments['SerialNumber'].unique()],
                placeholder='Select a shipment number'
            ),
            html.Div(id='metadata-display'),
            dcc.Graph(id='line-plot'),
            dcc.Graph(id='line-plot-relative'),
            dcc.Graph(id='stft-line-plot'),
            dcc.Graph(id='stft-histogram-plot'),
            dcc.Graph(id='rolling-plot'),
            dcc.Graph(id='histogram-plot'),
            dcc.Graph(id='box-plot'),
            dcc.Graph(id='stft-multi-line-spectogram'),
            dcc.Graph(id='heatmap-plot'),
            dcc.Graph(id='decomposition-plot'),
            dcc.Graph(id='autocorrelation-plot')
        ])
    elif selected_section == 'section2':
        return html.Div([
            html.H1('Dataset Summary Statistics', style={'textAlign': 'left'}),
            html.Div(id='temperature-stats'),
            html.H1("Individual Plots"),
            dcc.Dropdown(
                id='dropdown-serial-multi',
                options=[{'label': serial, 'value': serial} for serial in shipments['SerialNumber'].unique()],
                placeholder='Select multiple shipment numbers',
                multi=True
            ),
            dcc.Graph(id='multi-line-plot'),
            dcc.Graph(id='stft-multi-line-plot'),
            dcc.Dropdown(
                id='dropdown-serial-multi-2',
                options=[{'label': serial, 'value': serial} for serial in shipments['SerialNumber'].unique()],
                placeholder='Select multiple shipment numbers',
                multi=True
            ),
            dcc.Graph(id='fourier-line-plot'),
            html.Button('Remove > 75th percentile', id='filter-button', n_clicks=0),
            dcc.Graph(id='filtered-multi-line-plot'),
            html.Div(id='75-percentile-filtered-paragraph'),
            html.Button('Remove pre-cooling', id='cut-button', n_clicks=0),
            dcc.Graph(id='cut-multi-line-plot'),
            html.Button('Remove ramp-up', id='cut-button-2', n_clicks=0),
            dcc.Graph(id='cut-multi-line-plot-2'),
            dcc.Graph(id='multi-box-plot'),
            html.H1("Cluster Plots"),
            html.Div([
                html.Label('Number of Clusters for Trip Duration:', style={'margin-right': '10px'}),
                dcc.Input(
                    id='tripduration-cluster-number',
                    type='number',
                    value=5,  # Default number of clusters
                    min=1,  # Minimum value to prevent invalid input
                    style={'margin-right': '20px'}
                ),
            ], style={'margin-bottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Label('Number of Clusters for Temperature:', style={'margin-right': '10px'}),
                dcc.Input(
                    id='temperature-cluster-number',
                    type='number',
                    value=5,  # Default number of clusters
                    min=1,  # Minimum value to prevent invalid input
                    style={'margin-right': '20px'}
                ),
            ], style={'margin-bottom': '20px', 'display': 'flex', 'alignItems': 'center'}),
            dcc.Graph(id='clustering-plot'),
            dcc.Graph(id='departure-time-clustering-plot'),
            dcc.Graph(id='arrival-time-clustering-plot'),
            dcc.Graph(id='temperature-clustering-plot'),
            html.H2("Cluster summary statistics"),
            html.Div(id='cluster-summary-stats'),
            html.H1("Correlation Plots and Average statistics"),
            html.Button('Generate correlation matrix', id='correlation-plot', n_clicks=0),
            dcc.Graph(id='correlation-heatmap'),
            html.Div(id='average-stats-display')
        ])


# Callback for the multi-selection line plot
@app.callback(
    Output('multi-line-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_multi_line_plot(selected_serials):
    if selected_serials:
        fig = go.Figure()
        for serial in selected_serials:
            filtered_df = shipments[shipments['SerialNumber'] == serial]
            fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=filtered_df['PointValue'],
                                     mode='lines', name=serial))
        # Add boundary lines for the temperature range
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[2] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='2°C Supplier lower boundary and Ideal upper boundary', line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[6] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='6°C Supplier upper boundary', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[-1] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='-1°C Ideal lower boundary', line=dict(color='green', dash='dash')))
        fig.update_layout(title='Comparative Temperature Time Series', xaxis_title='DateTime',
                          yaxis_title='Temperature')
        return fig
    return go.Figure()



@app.callback(
    Output('fourier-line-plot', 'figure'),
    [Input('dropdown-serial-multi-2', 'value')]
)
def update_fourier_plot(selected_serials):
    if selected_serials:
        # Create a figure object
        fig = go.Figure()

        # Determine the maximum length of shipments to standardize the data length
        max_length = max(shipments[shipments['SerialNumber'].isin(selected_serials)].groupby('SerialNumber').apply(
            lambda x: len(x['PointValue'])
        ))

        # Process each serial number
        for serial in selected_serials:
            filtered_df = shipments[shipments['SerialNumber'] == serial]
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
            sensor_data = filtered_df['PointValue'].values

            # Pad the data to the maximum length
            sensor_data_padded = np.pad(sensor_data, (0, max_length - len(sensor_data)), 'constant')

            # Remove the mean of the data
            sensor_data_detrended = sensor_data_padded - np.mean(sensor_data_padded)
            fs = 1/900  # Sampling frequency

            # Perform Fourier Transform on the detrended data
            yf = np.fft.fft(sensor_data_detrended)
            xf = np.fft.fftfreq(max_length, d=1/fs)
            xf = np.fft.fftshift(xf)  # Shift zero frequency to center
            yf_shifted = np.fft.fftshift(yf)  # Shift zero frequency component to center

            # Plot the line for this serial number
            fig.add_trace(
                go.Scatter(x=xf, y=np.abs(yf_shifted), mode='lines', name=f'Serial {serial}')
            )

        # Update plot layout
        fig.update_layout(title='Fourier Transform Magnitude of Temperature Frequency Components',
                          xaxis_title='Frequency (Hz)',
                          yaxis_title='Magnitude',
                          height=600)

        return fig
    return go.Figure()


@app.callback(
    Output('stft-multi-line-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_stft_multi_line_plot(selected_serials):
    if selected_serials:
        # Create a figure with subplots
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Segment 1', 'Segment 2', 'Segment 3'))

        # Determine the maximum length of shipments to standardize the data length
        max_length = max(shipments[shipments['SerialNumber'].isin(selected_serials)].groupby('SerialNumber').apply(
            lambda x: len(x['PointValue'])
        ))

        # Samples per segment based on the longest shipment
        samples_per_segment = max_length // 3

        # Process each serial number
        for serial in selected_serials:
            filtered_df = shipments[shipments['SerialNumber'] == serial]
            filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
            sensor_data = filtered_df['PointValue'].values

            # Pad the data to the maximum length
            sensor_data_padded = np.pad(sensor_data, (0, max_length - len(sensor_data)), 'constant')

            # Remove the mean of the data
            sensor_data_detrended = sensor_data_padded - np.mean(sensor_data_padded)
            fs = 1/900  # Sampling frequency

            # Perform STFT on data without the mean for each segment
            for segment in range(1, 4):
                start_index = (segment - 1) * samples_per_segment
                end_index = start_index + samples_per_segment if segment < 3 else max_length
                segment_data = sensor_data_detrended[start_index:end_index]

                f, t, Zxx = stft(segment_data, fs=fs, window='hann', nperseg=samples_per_segment, noverlap=samples_per_segment // 2)
                Zxx_magnitude = np.abs(Zxx)
                selected_time_segment = Zxx_magnitude[:, 0]  # First column for STFT result of this segment

                # Plot the line for this segment
                fig.add_trace(
                    go.Scatter(x=f, y=selected_time_segment, mode='lines', name=f'{serial} Segment {segment}'),
                    row=segment, col=1
                )

        # Update plot layout
        fig.update_layout(title='STFT Magnitude of Temperature Frequency Components',
                          xaxis_title='Frequency (Hz)',
                          yaxis_title='Amplitude',
                          height=1200)  # Increase the height to accommodate 3 subplots

        return fig
    return go.Figure()

@app.callback(
    Output('stft-multi-line-spectogram', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_stft_multi_line_spectogram(selected_serial):
    if selected_serial:
        fig = go.Figure()

        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
        sensor_data = filtered_df['PointValue'].values

        # Remove the mean of the data
        sensor_data_detrended = sensor_data - np.mean(sensor_data)
        fs = 1/900
        # Perform STFT on data without the mean
        f, t, Zxx = stft(sensor_data_detrended, fs=fs, window='hann', nperseg=57, noverlap=2)

        # Calculate magnitude of STFT result for plotting
        Zxx_magnitude = np.abs(Zxx)

        # Plot a spectrogram
        fig.add_trace(
            go.Heatmap(
                x=t,  # Time
                y=f,  # Frequency
                z=Zxx_magnitude,  # Magnitude
                colorscale='Viridis',
                name=f'{selected_serial}'
            )
        )

        # Update plot layout for a spectrogram
        fig.update_layout(
            title='Spectrogram of Temperature Frequency Components',
            xaxis_title='Time',
            yaxis_title='Frequency (Hz)',
            # Adjust the layout as needed
        )

        return fig
    return go.Figure()

@app.callback(
    Output('filtered-multi-line-plot', 'figure'),
    [[Input('dropdown-serial-multi', 'value')],
     Input('filter-button', 'n_clicks')]
)
def update_filtered_multi_line_plot(selected_serials, selected_vars):
    if not selected_vars:
        return go.Figure()

    filtered_shipments = shipments.copy()
    # Compute the 75th percentile (Q3) of the entire dataset's initial temperature
    temp_q3 = filtered_shipments['PointValue'].quantile(0.75)

    # Initialize a list to hold the serial numbers of shipments to remove
    groups_to_remove = []

    # Group by 'SerialNumber' to process each shipment group
    ship_groups = filtered_shipments.groupby('SerialNumber')

    # Iterate over each group to check the condition
    for group_name, ship_group in ship_groups:
        # Calculate the average of the initial temperature for each group
        # Assuming 'init_temp_check' defines how many initial readings to consider
        avg_init_temp = np.mean(ship_group['PointValue'])

        # If the average initial temperature is above the 75th percentile, add to removal list
        if avg_init_temp > temp_q3:
            groups_to_remove.append(group_name)

    # Filter out the groups to be removed
    filtered_shipments = filtered_shipments[~filtered_shipments['SerialNumber'].isin(groups_to_remove)]

    fig = go.Figure()

    if selected_serials:
        for serial in selected_serials[0]:
            filtered_df = filtered_shipments[filtered_shipments['SerialNumber'] == serial]
            if filtered_df.empty:
                continue
            fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=filtered_df['PointValue'],
                                     mode='lines', name=serial))
        # Add boundary lines for the temperature range
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[2] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='2°C lower boundary', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[6] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='6°C upper boundary', line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[0] * len(shipments['Date / Time'].unique()), mode='lines',
                                 name='6°C upper boundary', line=dict(color='green', dash='dash')))
        fig.update_layout(title='Comparative Temperature Time Series (<=75th percentile)', xaxis_title='DateTime',
                          yaxis_title='Temperature')
        return fig
    return go.Figure()


@app.callback(
    Output('cut-multi-line-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value'),
     Input('cut-button', 'n_clicks')]
)
def update_cut_multi_line_plot(selected_serials, n_clicks):
    if not selected_serials or n_clicks == 0:
        return go.Figure()

    filtered_shipments = shipments.copy()

    fig = go.Figure()

    for serial in selected_serials:
        ship_group = filtered_shipments[filtered_shipments['SerialNumber'] == serial]

        if ship_group.empty:
            continue

        # Calculate the temperature gradient for the group
        ship_group = calculate_gradient(ship_group, 'PointValue')

        # Find the stabilization point
        stabilization_index = find_stabilization_point(ship_group, 'Temperature Gradient', side="pre-cooling")

        # Trim the pre-cooling section
        trimmed_group = trim_time_series(ship_group, stabilization_index, side="pre-cooling")

        # Plot if the group is not empty after trimming
        if not trimmed_group.empty:
            fig.add_trace(go.Scatter(
                x=trimmed_group['Date / Time'],
                y=trimmed_group['PointValue'],
                mode='lines',
                name=serial
            ))

    # Add boundary lines for the temperature range
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[2] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='2°C Supplier lower boundary and Ideal upper boundary',
                             line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[6] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='6°C Supplier upper boundary', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[-1] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='-1°C Ideal lower boundary', line=dict(color='green', dash='dash')))

    # Update layout
    fig.update_layout(title='Comparative Temperature Time Series After Pre-Cooling Trim', xaxis_title='DateTime',
                      yaxis_title='Temperature')

    return fig


@app.callback(
    Output('cut-multi-line-plot-2', 'figure'),
    [Input('dropdown-serial-multi', 'value'),
     Input('cut-button-2', 'n_clicks')]
)
def update_cut_multi_line_plot_2(selected_serials, n_clicks):
    if not selected_serials or n_clicks == 0:
        return go.Figure()

    filtered_shipments = shipments.copy()

    fig = go.Figure()

    for serial in selected_serials:
        ship_group = filtered_shipments[filtered_shipments['SerialNumber'] == serial]

        if ship_group.empty:
            continue

        # Calculate the temperature gradient for the group
        ship_group = calculate_gradient(ship_group, 'PointValue')

        # Find the stabilization point for pre-cooling
        stabilization_index_precool = find_stabilization_point(ship_group, 'Temperature Gradient', side="pre-cooling")

        # Find the stabilization point for ramp-up
        stabilization_index_ramp = find_stabilization_point(ship_group, 'Temperature Gradient', side="ramp-up")

        # Trim the pre-cooling and ramp-up sections
        trimmed_group = trim_time_series(ship_group, stabilization_index_ramp, side="ramp-up")

        # Plot if the group is not empty after trimming
        if not trimmed_group.empty:
            fig.add_trace(go.Scatter(
                x=trimmed_group['Date / Time'],
                y=trimmed_group['PointValue'],
                mode='lines',
                name=serial
            ))

    # Add boundary lines for the temperature range
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[2] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='2°C Supplier lower boundary and Ideal upper boundary',
                             line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[6] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='6°C Supplier upper boundary', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=shipments['Date / Time'], y=[-1] * len(shipments['Date / Time'].unique()), mode='lines',
                             name='-1°C Ideal lower boundary', line=dict(color='green', dash='dash')))

    # Update layout
    fig.update_layout(title='Comparative Temperature Time Series After Ramp-up Trim', xaxis_title='DateTime',
                      yaxis_title='Temperature')

    return fig

@app.callback(
    Output('75-percentile-filtered-paragraph', 'children'),
    [[Input('dropdown-serial-multi', 'value')],
     Input('filter-button', 'n_clicks')]
)
def update_filtered_75_count(selected_serials, selected_vars):
    if not selected_vars:
        children = None
        return children

    filtered_shipments = shipments.copy()
    total_shipments = len(filtered_shipments['SerialNumber'].unique())
    # Compute the 75th percentile (Q3) of the entire dataset's initial temperature
    temp_q3 = filtered_shipments['PointValue'].quantile(0.75)

    # Initialize a list to hold the serial numbers of shipments to remove
    groups_to_remove = []

    # Group by 'SerialNumber' to process each shipment group
    ship_groups = filtered_shipments.groupby('SerialNumber')

    # Iterate over each group to check the condition
    for group_name, ship_group in ship_groups:
        # Calculate the average of the initial temperature for each group
        # Assuming 'init_temp_check' defines how many initial readings to consider
        avg_init_temp = np.mean(ship_group['PointValue'])

        # If the average initial temperature is above the 75th percentile, add to removal list
        if avg_init_temp > temp_q3:
            groups_to_remove.append(group_name)

    num_removed = len(groups_to_remove)

    children = html.P(f"Total shipments removed: {num_removed} out of {total_shipments}")

    return children


@app.callback(
    Output('multi-box-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_multi_box_plot(selected_serials):
    if selected_serials:
        fig = go.Figure()

        for serial in selected_serials:
            filtered_df = shipments[shipments['SerialNumber'] == serial]
            # Add box plot for each serial
            fig.add_trace(
                go.Box(y=filtered_df['PointValue'],
                       name=f"Serial {serial}",
                       boxmean=True))

        # Update layout to adjust for a large number of shipments
        fig.update_layout(
            title='Distribution of Temperature Measurements per Shipment',
            xaxis_title='Serial Number',
            yaxis_title='Temperature (°C)',
            xaxis={'tickangle': 45, 'tickfont': {'size': 8}},  # Rotate x-axis labels for readability
            showlegend=False  # Optionally hide the legend if it's not necessary
        )

        return fig
    return go.Figure()


# Callback for displaying average statistics in a table
@app.callback(
    Output('average-stats-display', 'children'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_average_stats(selected_serials):
    if selected_serials:
        # Initialize a list to hold each row of the table
        table_rows = []

        for serial in selected_serials:
            filtered_df = shipments[shipments['SerialNumber'] == serial]

            # Calculate statistics for each shipment
            avg_temp = filtered_df['PointValue'].mean()
            avg_trip_duration = filtered_df['Full Trip Duration'].iloc[0]
            total_relative_time = filtered_df['Relative Time'].iloc[-1]
            departure_time = filtered_df['ActualDepartureTime'].unique()[0]
            arrival_time = filtered_df['ActualArrivalTime'].unique()[0]

            # Append statistics as a new row in the table
            table_rows.append({
                'SerialNumber': serial,
                'Average Temperature (°C)': avg_temp,
                'Trip Duration': avg_trip_duration,
                'Total Relative Time': total_relative_time,
                'Departure Time': departure_time,
                'Arrival Time': arrival_time
            })

        # Create a DataTable to display the stats
        return dash.dash_table.DataTable(
            columns=[
                {'name': 'SerialNumber', 'id': 'SerialNumber', 'type': 'text'},
                {'name': 'Average Temperature (°C)', 'id': 'Average Temperature (°C)', 'type': 'numeric'},
                {'name': 'Trip Duration', 'id': 'Trip Duration', 'type': 'numeric'},
                {'name': 'Total Relative Time', 'id': 'Total Relative Time', 'type': 'numeric'},
                {'name': 'Departure Time', 'id': 'Departure Time', 'type': 'datetime'},
                {'name': 'Arrival Time', 'id': 'Arrival Time', 'type': 'datetime'},
            ],
            data=table_rows,
            sort_action='native',  # Enable sorting
            filter_action='native',  # Enable filtering
            style_table={'overflowX': 'auto'}  # Horizontal scroll
        )
    return "Select shipments to display average statistics."


# Callback for clustering plot based on 'TripLength'
@app.callback(
    Output('clustering-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value'), Input('tripduration-cluster-number', 'value')]
)
def update_clustering_plot(selected_serials, n_clusters):
    # Check if selected_serials or n_clusters is None
    if selected_serials is None or n_clusters is None:
        return go.Figure()  # Return an empty figure

    # Filter dataframe based on selected_serials
    df_for_clustering_trip = shipments[shipments['SerialNumber'].isin(selected_serials)]
    df_for_clustering_trip['Full Trip Duration'] = pd.to_timedelta(df_for_clustering_trip['Full Trip Duration'])
    df_for_clustering_trip['TripLengthHours'] = df_for_clustering_trip['Full Trip Duration'].dt.total_seconds() / 3600
    df_for_clustering_trip = df_for_clustering_trip[['SerialNumber', 'TripLengthHours']].dropna()

    # Use the provided number of clusters, converting it to an integer
    kmeans = KMeans(n_clusters=int(n_clusters))
    df_for_clustering_trip['Cluster'] = kmeans.fit_predict(df_for_clustering_trip[['TripLengthHours']])

    # Save the cluster assignments to a CSV file
    filename = f"results/trip_duration_cluster_{n_clusters}_clusters.csv"
    df_for_clustering_trip.to_csv(filename, index=False)
    print(f"Saved trip duration cluster assignments to {filename}")

    # Getting centroids
    centroids = kmeans.cluster_centers_

    fig = go.Figure()

    # Scatter plot for shipments
    for cluster in df_for_clustering_trip['Cluster'].unique():
        df_cluster = df_for_clustering_trip[df_for_clustering_trip['Cluster'] == cluster]
        fig.add_trace(go.Scatter(x=df_cluster['SerialNumber'], y=df_cluster['TripLengthHours'],
                                 mode='markers', name=f'Cluster {cluster}',
                                 marker=dict(size=10),
                                 text=df_cluster['SerialNumber']))  # Text for hover information

    # Add centroids to the plot
    for i, centroid in enumerate(centroids):
        fig.add_trace(go.Scatter(x=['Centroid'], y=[centroid[0]],
                                 mode='markers+text', name=f'Centroid {i}',
                                 marker=dict(symbol='x', size=12, color='black'),
                                 text=[f'Centroid {i}'], textposition='top center'))

    fig.update_layout(title='Clustering Plot Based on Shipment Duration',
                      xaxis_title='Serial Number',
                      yaxis_title='Trip Length (Hours)',
                      xaxis={'type': 'category'})  # Assuming SerialNumber is categorical

    return fig


# Clustering trips based on departure times
@app.callback(
    Output('departure-time-clustering-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value')])
def update_departure_time_clustering_plot(selected_serials):
    if not selected_serials:
        return go.Figure()

    # Filter dataframe based on selected_serials
    df_for_clustering = shipments[shipments['SerialNumber'].isin(selected_serials)].copy()
    df_for_clustering['ActualDepartureTime'] = pd.to_datetime(df_for_clustering['ActualDepartureTime'])

    # Extract month-year as a string
    df_for_clustering['MonthYear'] = df_for_clustering['ActualDepartureTime'].dt.to_period('M')

    # Label Encoder
    le = LabelEncoder()
    df_for_clustering['MonthYearEncoded'] = le.fit_transform(df_for_clustering['MonthYear'])

    # K-Means Clustering with n clusters (n = number of months in dataset)
    kmeans = KMeans(n_clusters=len(df_for_clustering['MonthYear'].unique()))
    df_for_clustering['Cluster'] = kmeans.fit_predict(df_for_clustering[['MonthYearEncoded']].values.reshape(-1, 1))

    # Getting centroids
    centroids = kmeans.cluster_centers_

    fig = go.Figure()

    # Scatter plot for shipments
    for cluster in df_for_clustering['Cluster'].unique():
        df_cluster = df_for_clustering[df_for_clustering['Cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=df_cluster['SerialNumber'], y=df_cluster['MonthYearEncoded'],
            mode='markers', name=f'Cluster {cluster}',
            marker=dict(size=10),
            text=df_cluster['SerialNumber']))  # Text for hover information

    # Add centroids to the plot
    for i, centroid in enumerate(centroids):
        # Find the closest month-year for the centroid
        centroid_month_year = le.inverse_transform([int(round(centroid[0]))])[0]
        fig.add_trace(go.Scatter(
            x=['Centroid'], y=[centroid[0]],
            mode='markers+text', name=f'Centroid {i} ({centroid_month_year})',
            marker=dict(symbol='x', size=12, color='black'),
            text=[f'Centroid {i} ({centroid_month_year})'], textposition='top center'))

    # Update layout to use the actual month-year labels for y-axis ticks
    month_year_ticks = {str(v): str(k) for k, v in enumerate(le.classes_)}
    fig.update_layout(
        title='Clustering Based on Departure Time by Month-Year',
        yaxis_title='Departure Months',
        yaxis=dict(
            title='Departure Month-Year',
            tickmode='array',
            tickvals=list(range(len(month_year_ticks))),
            ticktext=[str(my) for my in month_year_ticks.keys()]
        ),
        xaxis=dict(
            title='',
            showticklabels=False
        )
    )

    return fig


# Clustering trips based on arrival times
@app.callback(
    Output('arrival-time-clustering-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_arrival_time_clustering_plot(selected_serials):
    if not selected_serials:
        return go.Figure()

    # Filter dataframe based on selected_serials
    df_for_clustering = shipments[shipments['SerialNumber'].isin(selected_serials)].copy()
    df_for_clustering['ActualArrivalTime'] = pd.to_datetime(df_for_clustering['ActualArrivalTime'])

    # Extract month-year as a string
    df_for_clustering['MonthYear'] = df_for_clustering['ActualArrivalTime'].dt.to_period('M')

    # Label Encoder
    le = LabelEncoder()
    df_for_clustering['MonthYearEncoded'] = le.fit_transform(df_for_clustering['MonthYear'])

    # K-Means Clustering with n clusters (n = number of months in dataset)
    kmeans = KMeans(n_clusters=len(df_for_clustering['MonthYear'].unique()))
    df_for_clustering['Cluster'] = kmeans.fit_predict(df_for_clustering[['MonthYearEncoded']].values.reshape(-1, 1))

    # Getting centroids
    centroids = kmeans.cluster_centers_

    fig = go.Figure()

    # Scatter plot for shipments
    for cluster in df_for_clustering['Cluster'].unique():
        df_cluster = df_for_clustering[df_for_clustering['Cluster'] == cluster]
        fig.add_trace(go.Scatter(
            x=df_cluster['SerialNumber'], y=df_cluster['MonthYearEncoded'],
            mode='markers', name=f'Cluster {cluster}',
            marker=dict(size=10),
            text=df_cluster['SerialNumber']))  # Text for hover information

    # Add centroids to the plot
    for i, centroid in enumerate(centroids):
        # Find the closest month-year for the centroid
        centroid_month_year = le.inverse_transform([int(round(centroid[0]))])[0]
        fig.add_trace(go.Scatter(
            x=['Centroid'], y=[centroid[0]],
            mode='markers+text', name=f'Centroid {i} ({centroid_month_year})',
            marker=dict(symbol='x', size=12, color='black'),
            text=[f'Centroid {i} ({centroid_month_year})'], textposition='top center'))

    # Update layout to use the actual month-year labels for y-axis ticks
    month_year_ticks = {str(v): str(k) for k, v in enumerate(le.classes_)}
    fig.update_layout(
        title='Clustering Based on Arrival Time by Month-Year',
        yaxis_title='Arrival Months',
        yaxis=dict(
            title='Arrival Month-Year',
            tickmode='array',
            tickvals=list(range(len(month_year_ticks))),
            ticktext=[str(my) for my in month_year_ticks.keys()]
        ),
        xaxis=dict(
            title='',
            showticklabels=False
        )
    )

    return fig


@app.callback(
    Output('temperature-clustering-plot', 'figure'),
    [Input('dropdown-serial-multi', 'value'),
     Input('temperature-cluster-number', 'value')],

)
def update_temperature_clustering_plot(selected_serials, n_clusters):
    if selected_serials is None or n_clusters is None:
        return go.Figure()  # Return an empty figure

    # Filter dataframe based on selected_serials
    df_for_clustering = shipments[shipments['SerialNumber'].isin(selected_serials)]

    df_avg_temp = df_for_clustering.groupby('SerialNumber')['PointValue'].mean().reset_index()
    df_avg_temp.columns = ['SerialNumber', 'AvgTemperature']

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters)
    df_avg_temp['Cluster'] = kmeans.fit_predict(df_avg_temp[['AvgTemperature']])

    # Save the cluster assignments to a CSV file
    filename = f"results/average_temperature_cluster_{n_clusters}_clusters.csv"
    df_avg_temp.to_csv(filename, index=False)
    print(f"Saved average temperatue cluster assignments to {filename}")

    # Getting centroids
    centroids = kmeans.cluster_centers_

    fig = go.Figure()

    # Scatter plot for shipments
    for cluster in df_avg_temp['Cluster'].unique():
        df_cluster = df_avg_temp[df_avg_temp['Cluster'] == cluster]
        fig.add_trace(go.Scatter(x=df_cluster['SerialNumber'], y=df_cluster['AvgTemperature'],
                                 mode='markers', name=f'Cluster {cluster}',
                                 marker=dict(size=10),
                                 text=df_cluster['SerialNumber']))  # Text for hover information

    # Add centroids to the plot
    for i, centroid in enumerate(centroids):
        fig.add_trace(go.Scatter(x=['Centroid'], y=[centroid[0]],
                                 mode='markers+text', name=f'Centroid {i}',
                                 marker=dict(symbol='x', size=12, color='black'),
                                 text=[f'Centroid {i}'], textposition='top center'))

    fig.update_layout(title='Clustering Plot Based on Average Shipment Temperature',
                      xaxis_title='Serial Number',
                      yaxis_title='Average Temperature',
                      xaxis={'type': 'category'})

    return fig


@app.callback(
    Output('cluster-summary-stats', 'children'),
    [Input('dropdown-serial-multi', 'value'), Input('tripduration-cluster-number', 'value'),
     Input('temperature-cluster-number', 'value')]
)
def update_cluster_summary_stats(selected_serials, n_clusters_trip, n_clusters_temp):
    if not selected_serials or n_clusters_trip is None or n_clusters_temp is None:
        return [html.P(
            'Select n clusters for temperature and trip duration above to see number of shipments in each cluster')]

    # For trip duration
    df_for_clustering_trip = shipments[shipments['SerialNumber'].isin(selected_serials)].copy()
    df_for_clustering_trip['Full Trip Duration'] = pd.to_timedelta(df_for_clustering_trip['Full Trip Duration'])
    df_for_clustering_trip['TripLengthHours'] = df_for_clustering_trip['Full Trip Duration'].dt.total_seconds() / 3600
    df_for_clustering_trip = df_for_clustering_trip[['SerialNumber', 'TripLengthHours']].dropna()

    kmeans_trip = KMeans(n_clusters=int(n_clusters_trip))
    df_for_clustering_trip['Cluster'] = kmeans_trip.fit_predict(df_for_clustering_trip[['TripLengthHours']])
    trip_duration_cluster_counts = df_for_clustering_trip.groupby('Cluster')['SerialNumber'].nunique().reset_index(
        name='counts')

    # For average temperature
    df_for_clustering_temp = shipments[shipments['SerialNumber'].isin(selected_serials)].copy()
    df_avg_temp = df_for_clustering_temp.groupby('SerialNumber')[
        'PointValue'].mean().reset_index()
    df_avg_temp.columns = ['SerialNumber', 'AvgTemperature']

    kmeans_temp = KMeans(n_clusters=int(n_clusters_temp))
    df_avg_temp['Cluster'] = kmeans_temp.fit_predict(df_avg_temp[['AvgTemperature']])
    avg_temp_cluster_counts = df_avg_temp.groupby('Cluster')['SerialNumber'].nunique().reset_index(name='counts')

    # Calculate summary stats
    trip_cluster_summary = df_for_clustering_trip.groupby('Cluster').agg(
        counts=('SerialNumber', 'nunique'),
        centroid=('TripLengthHours', 'mean')  # Assuming you want the mean of TripLengthHours as the centroid
    ).reset_index()

    # Convert the temperature summary DataFrame into a format suitable for DataTable
    temp_cluster_summary = df_avg_temp.groupby('Cluster').agg(
        counts=('SerialNumber', 'nunique'),
        centroid=('AvgTemperature', 'mean')  # Assuming you want the mean of AvgTemperature as the centroid
    ).reset_index()

    # DataTable for Temperature Clusters
    trip_table = dash.dash_table.DataTable(
        columns=[
            {"name": "Cluster Number", "id": "Cluster"},
            {"name": "Counts", "id": "counts"},
            {"name": "Centroids (Trip Duration)", "id": "centroid"},
        ],
        data=trip_cluster_summary.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'fontWeight': 'bold'}
    )

    # DataTable for Temperature Clusters
    temp_table = dash.dash_table.DataTable(
        columns=[
            {"name": "Cluster Number", "id": "Cluster"},
            {"name": "Counts", "id": "counts"},
            {"name": "Centroids (Mean Temperature)", "id": "centroid"},
        ],
        data=temp_cluster_summary.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'},
        style_header={'fontWeight': 'bold'}
    )

    # Combine the two tables with headers into the layout
    children = [
        html.H3('Trip Duration Clusters', style={'textAlign': 'center'}),
        trip_table,  # The table for trip duration clusters already defined in your callback
        html.H3('Temperature Clusters', style={'textAlign': 'center'}),
        temp_table  # The table for temperature clusters
    ]

    return children


# Callback to show summary statistics based on the selected number of clusters
@app.callback(
    Output('temperature-stats', 'children'),
    [Input('dropdown-serial-multi', 'value')]
)
def update_temperature_stats(selected_serials):
    if selected_serials is None:
        children = [
            html.H4('Temperature Statistics'),
            html.P('Select shipments above to see summary stats')
        ]
        return children

    df_for_clustering_temp = shipments[shipments['SerialNumber'].isin(selected_serials)]
    temp_mean = df_for_clustering_temp['PointValue'].mean()
    temp_median = df_for_clustering_temp['PointValue'].median()
    temp_sd = df_for_clustering_temp['PointValue'].std()
    temp_q1 = df_for_clustering_temp['PointValue'].quantile(0.25)
    temp_q3 = df_for_clustering_temp['PointValue'].quantile(0.75)
    temp_iqr = temp_q3 - temp_q1

    # Compute the average num rows for each shipment
    avg_len_list = []

    shipments_copy = shipments.copy()

    groups = shipments_copy.groupby('SerialNumber')

    for name, group in groups:
        avg_len_list.append(len(group))

    avg_shipment_num_rows = np.mean(avg_len_list)

    children = [
        html.H4('Temperature Statistics'),
        html.P(f'Average number of rows per shipment: {avg_shipment_num_rows:.2f}'),
        html.P(f'Mean Temperature: {temp_mean:.2f}°C'),
        html.P(f'Median Temperature: {temp_median:.2f}°C'),
        html.P(f'Standard Deviation of Temperature: {temp_sd:.2f}°C'),
        html.P(f'25th Percentile (Q1): {temp_q1:.2f}°C'),
        html.P(f'75th Percentile (Q3): {temp_q3:.2f}°C'),
        html.P(f'Interquartile Range (IQR): {temp_iqr:.2f}°C')
    ]

    return children


@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('correlation-plot', 'n_clicks')]
)
def update_correlation_heatmap(selected_vars):
    if not selected_vars:
        return go.Figure()

    # Filter the DataFrame to include only the selected variables
    df_filtered = shipments.copy()
    df_filtered = df_filtered[numeric_cols]

    # Label Encoder
    le = LabelEncoder()
    df_filtered['SerialNumber'] = le.fit_transform(df_filtered['SerialNumber'])

    df_filtered['Full Trip Duration'] = pd.to_timedelta(df_filtered['Full Trip Duration'])
    df_filtered['Relative Time'] = pd.to_timedelta(df_filtered['Relative Time'])

    df_filtered['ActualArrivalTime'] = pd.to_datetime(df_filtered['ActualArrivalTime'])
    df_filtered['ActualDepartureTime'] = pd.to_datetime(df_filtered['ActualDepartureTime'])
    # df_filtered['ActualArrivalTime'] = pd.to_timedelta(df_filtered['ActualArrivalTime'])
    # df_filtered['ActualDepartureTime'] = pd.to_timedelta(df_filtered['ActualDepartureTime'])

    df_filtered['Full Trip Duration'] = df_filtered['Full Trip Duration'].dt.total_seconds()
    df_filtered['Relative Time'] = df_filtered['Relative Time'].dt.total_seconds()
    # df_filtered['ActualDepartureTime'] = df_filtered['ActualDepartureTime'].dt.total_seconds()
    # df_filtered['ActualDepartureTime'] = df_filtered['ActualDepartureTime'].dt.total_seconds()

    # Calculate correlation matrix
    corr_matrix = df_filtered.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='Viridis'
    ))
    fig.update_layout(title='Correlation Heatmap', xaxis_title='Variables', yaxis_title='Variables')

    return fig


# Callback for shipment metadata
@app.callback(
    Output('metadata-display', 'children'),
    [Input('dropdown-serial', 'value')]
)
def update_metadata(selected_serial):
    if selected_serial:
        # Filter the DataFrame for the selected serial number
        metadata = shipments[shipments['SerialNumber'] == selected_serial].iloc[0]

        # Format the metadata into a human-readable string or HTML
        metadata_str = [
            html.H3("Metadata Information"),
            html.P(f"Serial Number: {metadata['SerialNumber']}"),
            html.P(f"Departure Time: {metadata['ActualDepartureTime']}"),
            html.P(f"Trip Length: {metadata['Full Trip Duration']}"),
            html.P(f"Arrival Time: {metadata['ActualArrivalTime']}"),
            html.P(f"Product Name: {metadata['ProductName']}"),
            html.P(f"Supplier: {metadata['OriginLocationName']}"),
            html.P(f"Receiver: {metadata['DestinationLocationName']}")
        ]

        return metadata_str
    return "Please select a serial number to display metadata."






###### Single shipment Analysis
# Callback for line plot
@app.callback(
    Output('line-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_line_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        print(filtered_df['SerialNumber'].unique())
        fig = go.Figure(
            data=go.Scatter(x=filtered_df['Date / Time'], y=filtered_df['PointValue'],
                            mode='lines', name='Temperature'))

        # Add boundary lines for the ideal client temperature range
        fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=[2] * len(filtered_df), mode='lines',
                                 name='2°C supplier lower boundary, ideal upper boundary', line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=[6] * len(filtered_df), mode='lines',
                                 name='6°C supplier upper boundary', line=dict(color='red', dash='dash')))

        # Add boundary lines for the ideal berries temperature range
        fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=[-1] * len(filtered_df), mode='lines',
                                 name='-1°C ideal lower boundary', line=dict(color='green', dash='dash')))

        fig.update_layout(title='Temperature Time Series', xaxis_title='DateTime', yaxis_title='Temperature')
        return fig
    return go.Figure()


@app.callback(
    Output('line-plot-relative', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_line_plot_with_relative_time(selected_serial):
    if selected_serial:
        # Filter the dataframe for the selected serial number
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]

        # Ensure that the date column is in datetime
        filtered_df['Date / Time'] = pd.to_datetime(filtered_df['Date / Time'])

        # Convert 'Date / Time' column to pandas Date / Time
        filtered_df['Date / Time'] = pd.to_datetime(filtered_df['Date / Time'])

        # Calculate the time difference in hours relative to the first timestamp
        relative_time_hours = (filtered_df['Date / Time'] - filtered_df['Date / Time'].iloc[0]).dt.total_seconds() / 3600

        # Create the figure
        fig = go.Figure(
            data=go.Scatter(x=relative_time_hours, y=filtered_df['PointValue'],
                            mode='lines', name='Temperature'))

        # Calculate the length of relative_time_hours for the boundary lines
        time_length = len(relative_time_hours)

        # Add boundary lines for the ideal client temperature range
        fig.add_trace(go.Scatter(x=relative_time_hours, y=[2] * time_length, mode='lines',
                                 name='2°C supplier lower boundary, ideal upper boundary',
                                 line=dict(color='blue', dash='dash')))
        fig.add_trace(go.Scatter(x=relative_time_hours, y=[6] * time_length, mode='lines',
                                 name='6°C supplier upper boundary', line=dict(color='red', dash='dash')))

        # Add boundary lines for the ideal berries temperature range
        fig.add_trace(go.Scatter(x=relative_time_hours, y=[-1] * time_length, mode='lines',
                                 name='-1°C ideal lower boundary', line=dict(color='green', dash='dash')))

        # Update layout with the new x-axis title
        fig.update_layout(title='Temperature Time Series (Relative time)', xaxis_title='Relative Time (hours)',
                          yaxis_title='Temperature')

        return fig
    return go.Figure()


@app.callback(
    Output('stft-line-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_stft_multi_line_plot(selected_serial):
    if selected_serial:
        # Create a figure with subplots
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Segment 1', 'Segment 2'))

        # Determine the length of the shipment
        ship_length = len(shipments[shipments['SerialNumber'] == selected_serial])

        # Samples per segment based on the longest shipment
        samples_per_segment = ship_length // 2

        # Process each serial number
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
        sensor_data = filtered_df['PointValue'].values

        # Remove the mean of the data
        sensor_data_detrended = sensor_data - np.mean(sensor_data)
        fs = 1/900  # Sampling frequency

        # Perform STFT on data without the mean for each segment
        for segment in range(1, 3):
            start_index = (segment - 1) * samples_per_segment
            end_index = start_index + samples_per_segment if segment < 2 else ship_length
            segment_data = sensor_data_detrended[start_index:end_index]

            f, t, Zxx = stft(segment_data, fs=fs, window='hann', nperseg=samples_per_segment, noverlap=samples_per_segment // 2)
            Zxx_magnitude = np.abs(Zxx)
            selected_time_segment = Zxx_magnitude[:, 0]  # First column for STFT result of this segment

            # Plot the line for this segment
            fig.add_trace(
                go.Scatter(x=f, y=selected_time_segment, mode='lines', name=f'{selected_serial} Segment {segment}'),
                row=segment, col=1
            )

        # Update plot layout
        fig.update_layout(title='STFT Magnitude of Temperature Frequency Components',
                          xaxis_title='Frequency (Hz)',
                          yaxis_title='Amplitude',
                          height=1200)  # Increase the height to accommodate 3 subplots

        return fig
    return go.Figure()

@app.callback(
    Output('stft-histogram-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_stft_histogram_plot(selected_serial):
    if selected_serial:
        # Create a figure with subplots
        fig = make_subplots(rows=3, cols=1, subplot_titles=('Segment 1', 'Segment 2', 'Segment 3'))

        # Filter the dataframe for the selected serial number
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
        sensor_data = filtered_df['PointValue'].values

        samples_per_shipment = len(sensor_data)
        samples_per_segment = samples_per_shipment // 3

        # Remove the mean of the data to detrend it
        sensor_data_detrended = sensor_data - np.mean(sensor_data)
        fs = 1 / 900  # Sampling frequency in Hz

        # Compute the STFT for each segment
        for segment in range(1, 4):
            start_index = (segment - 1) * samples_per_segment
            end_index = start_index + samples_per_segment if segment < 3 else len(sensor_data)
            segment_data = sensor_data_detrended[start_index:end_index]

            f, t, Zxx = stft(segment_data, fs=fs, window='hann', nperseg=samples_per_segment, noverlap=2)
            Zxx_magnitude = np.abs(Zxx)

            # Convert frequency to micro hertz and filter for frequencies <= 100 micro hertz
            micro_hertz_conversion = 1e6
            micro_hertz_frequencies = f * micro_hertz_conversion
            frequency_mask = micro_hertz_frequencies <= 100
            filtered_frequencies = micro_hertz_frequencies[frequency_mask]
            amplitude = np.max(Zxx_magnitude[frequency_mask, :], axis=1)  # Max amplitude for each frequency

            # Create individual bars for each frequency
            for i, freq in enumerate(filtered_frequencies):
                time_period_hr = str(round((1 / (freq * 0.000001)) / 3600, 2))
                time_period_hr = f"{time_period_hr} hrs"
                fig.add_trace(
                    go.Bar(x=[freq], y=[amplitude[i]], name=f'{freq} µHz', showlegend=False, text=time_period_hr),
                    row=segment, col=1
                )


        # Update plot layout
        fig.update_layout(title='STFT Histogram of Temperature Frequency Components (0 to 100 µHz)',
                          xaxis_title='Frequency (µHz)',
                          yaxis_title='Amplitude',
                          height=1200)  # Increase the height to accommodate 3 subplots

        return fig
    return go.Figure()


# @app.callback(
#     Output('stft-histogram-plot', 'figure'),
#     [Input('dropdown-serial', 'value')]
# )
# def update_stft_histogram_plot(selected_serial):
#     if selected_serial:
#         # Create a figure with subplots
#         fig = make_subplots(rows=3, cols=1, subplot_titles=('Segment 1', 'Segment 2', 'Segment 3'))
#
#         # Filter the dataframe for the selected serial number
#         filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
#         filtered_df['Date'] = pd.to_datetime(filtered_df['Date / Time'])
#         sensor_data = filtered_df['PointValue'].values
#
#         samples_per_shipment = len(sensor_data)
#         samples_per_segment = samples_per_shipment // 3
#
#         # Remove the mean of the data to detrend it
#         sensor_data_detrended = sensor_data - np.mean(sensor_data)
#         fs = 1 / 900  # Sampling frequency in Hz
#
#         # Compute the STFT for each segment
#         for segment in range(1, 4):
#             start_index = (segment - 1) * samples_per_segment
#             end_index = start_index + samples_per_segment if segment < 3 else len(sensor_data)
#             segment_data = sensor_data_detrended[start_index:end_index]
#
#             f, t, Zxx = stft(segment_data, fs=fs, window='hann', nperseg=samples_per_segment, noverlap=2)
#             Zxx_magnitude = np.abs(Zxx)
#
#             # Find the index of frequencies <= 100 micro hertz
#             micro_hertz_conversion = 1e6
#             micro_hertz_frequencies = f * micro_hertz_conversion
#             frequency_mask = micro_hertz_frequencies <= 100
#             time_bins = t[frequency_mask]
#             amplitude = np.max(Zxx_magnitude[frequency_mask, :], axis=0)  # Max amplitude for each time bin
#
#             # Create individual bars for each time bin
#             for i, time_bin in enumerate(time_bins):
#                 # Convert time bin from seconds to a readable format
#                 time_period_start = pd.to_timedelta(time_bin, unit='s')
#                 time_period_end = pd.to_timedelta(time_bin + fs * samples_per_segment, unit='s')
#                 annotation_text = f'{time_period_start} - {time_period_end}'
#
#                 # Add the bar for this time bin
#                 fig.add_trace(
#                     go.Bar(x=[time_bin], y=[amplitude[i]], name=f'Time bin {i}', showlegend=False),
#                     row=segment, col=1
#                 )
#
#                 # Add annotation for the time period
#                 fig.add_annotation(
#                     x=time_bin, y=amplitude[i], text=annotation_text,
#                     showarrow=True, arrowhead=1, row=segment, col=1
#                 )
#
#         # Update plot layout
#         fig.update_layout(
#             title='STFT Histogram of Temperature Frequency Components (0 to 100 µHz)',
#             xaxis_title='Time Period',
#             yaxis_title='Amplitude',
#             height=1200  # Increase the height to accommodate 3 subplots
#         )
#
#         return fig
#     return go.Figure()

# Callback for scatter plot with rolling mean
@app.callback(
    Output('rolling-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_rolling_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        rolling_mean = filtered_df['PointValue'].rolling(window=24).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=filtered_df['Date / Time'], y=filtered_df['PointValue'], mode='lines',
                       name='Temperature'))
        fig.add_trace(go.Scatter(x=filtered_df['Date / Time'], y=rolling_mean, mode='lines', name='Rolling Mean'))
        fig.update_layout(title='Temperature with Rolling Mean', xaxis_title='DateTime', yaxis_title='Temperature')
        return fig
    return go.Figure()


# Callback for histogram
@app.callback(
    Output('histogram-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_histogram_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        fig = go.Figure(data=go.Histogram(x=filtered_df['PointValue'], nbinsx=50))
        fig.update_layout(title='Temperature Distribution', xaxis_title='Temperature', yaxis_title='Count')
        return fig
    return go.Figure()


# Callback for box plot
@app.callback(
    Output('box-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_box_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        fig = go.Figure(
            data=go.Box(y=filtered_df['PointValue'], boxpoints='all', name="Temperature"))
        fig.update_layout(title='Box Plot of Temperature', yaxis_title='Temperature')
        return fig
    return go.Figure()


# Callback for heat map
@app.callback(
    Output('heatmap-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_heatmap_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]

        fig = go.Figure(data=go.Heatmap(
            z=filtered_df['PointValue'],
            x=filtered_df['Date / Time'].dt.date,
            y=filtered_df['Date / Time'].dt.hour,
            colorscale='Viridis'))
        fig.update_layout(title='Temperature Heatmap', xaxis_nticks=36)
        return fig
    return go.Figure()


# Callback for seasonal decompostion
@app.callback(
    Output('decomposition-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_decomposition_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        filtered_df = filtered_df[['Date / Time', 'PointValue']]
        filtered_df.set_index('Date / Time', inplace=True)
        # filtered_df.drop(columns=[])
        filtered_df = filtered_df.resample(
            'h').mean().dropna()
        decomposition = seasonal_decompose(filtered_df['PointValue'], model='additive')

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual'))

        fig.add_trace(go.Scatter(x=filtered_df.index, y=decomposition.observed, mode='lines', name='Observed'), row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=filtered_df.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=filtered_df.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3,
                      col=1)
        fig.add_trace(go.Scatter(x=filtered_df.index, y=decomposition.resid, mode='lines', name='Residual'), row=4,
                      col=1)

        fig.update_layout(height=600, title_text="Seasonal Decompose")
        return fig
    return go.Figure()


# Callback for auto-correlation plot
@app.callback(
    Output('autocorrelation-plot', 'figure'),
    [Input('dropdown-serial', 'value')]
)
def update_autocorrelation_plot(selected_serial):
    if selected_serial:
        filtered_df = shipments[shipments['SerialNumber'] == selected_serial]
        autocorr = [filtered_df['PointValue'].autocorr(lag=i) for i in
                    range(len(filtered_df) // 2)]

        fig = go.Figure(data=go.Bar(x=list(range(len(autocorr))), y=autocorr))
        fig.update_layout(title='Autocorrelation Plot', xaxis_title='Lag', yaxis_title='Autocorrelation')
        return fig
    return go.Figure()


if __name__ == '__main__':
    app.run_server(debug=True)
