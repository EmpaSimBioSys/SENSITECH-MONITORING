import configparser 
import dash
from dash import dcc, html, Input, Output, State
from datasets.coop import CoopData
from datasets.data_merger import ShipmentDataMerger
from preprocessing.trim import TimeSeriesTrimmer
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def z_normalize(ts):
    ts_mean = np.mean(ts)
    ts_std = np.std(ts)
    return (ts - ts_mean) / ts_std

# Load configuration items
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
coop_features = config["dataset"]["coop_features"].split(", ")

data_features = ['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown', 'Relative Time']

# Load and process datasets
def load_and_process_data(filepath, data_features):
    coop_shipments = CoopData(filepath, feature_list=data_features, dependent_var='Temperature')
    trimmer = TimeSeriesTrimmer(coop_shipments.data, temperature_column='Temperature')
    trimmed_shipments = trimmer.trim_time_series()
    grouped_shipments = trimmed_shipments.groupby("Serial no")
    serials = list(coop_shipments.data['Serial no'].unique())
    return grouped_shipments, serials

grouped_shipments_coop_spike, spike_serials = load_and_process_data("data/data_spike.csv", data_features)
grouped_shipments_coop_cool_defrost, cool_defrost_serials = load_and_process_data("data/data_cool_defrost.csv", data_features)
grouped_shipments_coop_excursion, excursion_serials = load_and_process_data("data/data_excursion.csv", data_features)
grouped_shipments_coop_not_precooled, not_precooled_serials = load_and_process_data("data/data_not_precooled.csv", data_features)
grouped_shipments_coop_norm, norm_serials = load_and_process_data("data/data_norm.csv", data_features)
grouped_shipments_coop_initial_ramp, initial_ramp_serials = load_and_process_data("data/data_initial_ramp.csv", data_features)
grouped_shipments_coop_top_freezing, top_freezing_serials = load_and_process_data("data/data_chilling_injury.csv", data_features)
grouped_shipments_coop_extended_drift, extended_drift_serials = load_and_process_data("data/data_extended_drift.csv", data_features)

# Merge all the different classes
merged_shipments = pd.concat([
    grouped_shipments_coop_spike.obj,
    grouped_shipments_coop_cool_defrost.obj,
    grouped_shipments_coop_excursion.obj,
    grouped_shipments_coop_not_precooled.obj,
    grouped_shipments_coop_norm.obj,
    grouped_shipments_coop_initial_ramp.obj,
    grouped_shipments_coop_top_freezing.obj,
    grouped_shipments_coop_extended_drift.obj
])

grouped_shipments_all = merged_shipments.groupby("Serial no")

# Load the complete dataset
config_path = 'config.ini'
coop_path = "data/all_data_combined_meta.csv"
bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"
shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
data = shipment_merger.merged_dataframe
data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()
shipment_groups = data.groupby('Serial no')

# Prepare data for DTW
shipment_time_series = {name: group['Temperature'].values for name, group in grouped_shipments_all}

# Create a list of serial numbers to track index in the distance matrix
serial_numbers = list(shipment_time_series.keys())

distance_matrix = np.load("/Users/divinefavourodion/results/dtw_distance_matrix.npy")

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
reordered_affinity_matrix = affinity_matrix[np.ix_(ordered_indices, ordered_indices)]
sorted_affinity_indices = np.argsort(-affinity_matrix.sum(axis=1))
highest_affinity_matrix = affinity_matrix[np.ix_(sorted_affinity_indices, sorted_affinity_indices)]

# Define class boundaries
class_boundaries = [
    (0, len(norm_serials)),
    (len(norm_serials), len(norm_serials) + len(spike_serials)),
    (len(norm_serials) + len(spike_serials), len(norm_serials) + len(spike_serials) + len(cool_defrost_serials)),
    (len(norm_serials) + len(spike_serials) + len(cool_defrost_serials), len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials)),
    (len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials), len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials)),
    (len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials), len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials) + len(initial_ramp_serials)),
    (len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials) + len(initial_ramp_serials), len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials) + len(initial_ramp_serials) + len(top_freezing_serials)),
    (len(norm_serials) + len(spike_serials) + len(cool_defrost_serials) + len(excursion_serials) + len(not_precooled_serials) + len(initial_ramp_serials) + len(top_freezing_serials), len(ordered_serials))
]

class_names = [
    "Normal",
    "Spikey",
    "Cool Defrost",
    "Excursion",
    "Not Precooled",
    "Initial Ramp",
    "Top Freezing",
    "Extended Drift"
]

def generate_heatmap_figure(matrix, serial_numbers, class_boundaries, class_names):
    fig = px.imshow(matrix, labels=dict(x="Serial no", y="Serial no"), x=serial_numbers, y=serial_numbers)
    fig.update_layout(coloraxis_showscale=False, width=1000, height=1000)

    shapes = []
    annotations = []
    for class_name, (start, end) in zip(class_names, class_boundaries):
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=start,
                y0=start,
                x1=end,
                y1=end,
                line=dict(color="green", width=2),
                fillcolor="rgba(0,0,0,0)"
            )
        )
        annotations.append(
            dict(
                x=(start + end) / 2,
                y=end,
                xref="x",
                yref="y",
                text=class_name,
                showarrow=False,
                font=dict(color="green", size=12)
            )
        )

    fig.update_layout(shapes=shapes, annotations=annotations)
    return fig

original_heatmap_fig = generate_heatmap_figure(affinity_matrix, serial_numbers, [], [])
reordered_heatmap_fig = generate_heatmap_figure(reordered_affinity_matrix, ordered_serials, class_boundaries, class_names)
highest_affinity_heatmap_fig = generate_heatmap_figure(highest_affinity_matrix, np.array(serial_numbers)[sorted_affinity_indices], [], [])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Original Affinity Matrix', children=[
            html.Div([
                dcc.Graph(id='original-heatmap', figure=original_heatmap_fig),
                dcc.Graph(id='original-time-series')
            ])
        ]),
        dcc.Tab(label='Reordered Affinity Matrix', children=[
            html.Div([
                dcc.Graph(id='reordered-heatmap', figure=reordered_heatmap_fig),
                dcc.Graph(id='reordered-time-series')
            ])
        ]),
        dcc.Tab(label='Highest Affinity Matrix', children=[
            html.Div([
                dcc.Graph(id='highest-affinity-heatmap', figure=highest_affinity_heatmap_fig),
                dcc.Graph(id='highest-affinity-time-series')
            ])
        ])
    ])
])

@app.callback(
    Output('original-time-series', 'figure'),
    Output('reordered-time-series', 'figure'),
    Output('highest-affinity-time-series', 'figure'),
    Input('original-heatmap', 'hoverData'),
    Input('reordered-heatmap', 'hoverData'),
    Input('highest-affinity-heatmap', 'hoverData'),
    State('original-time-series', 'figure'),
    State('reordered-time-series', 'figure'),
    State('highest-affinity-time-series', 'figure')
)
def update_time_series(original_hover, reordered_hover, highest_affinity_hover, original_fig, reordered_fig, highest_affinity_fig):
    ctx = dash.callback_context

    if not ctx.triggered:
        return original_fig, reordered_fig, highest_affinity_fig

    hovered_elem_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if hovered_elem_id == 'original-heatmap' and original_hover:
        point = original_hover['points'][0]
        x, y = point['x'], point['y']
        serial_x, serial_y = x, y
        group_x = shipment_groups.get_group(serial_x)
        group_y = shipment_groups.get_group(serial_y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_x['Date / Time'], y=group_x['Temperature'], mode='lines', name=serial_x))
        fig.add_trace(go.Scatter(x=group_y['Date / Time'], y=group_y['Temperature'], mode='lines', name=serial_y))
        return fig, reordered_fig, highest_affinity_fig

    elif hovered_elem_id == 'reordered-heatmap' and reordered_hover:
        point = reordered_hover['points'][0]
        x, y = point['x'], point['y']
        serial_x, serial_y = x, y
        group_x = shipment_groups.get_group(serial_x)
        group_y = shipment_groups.get_group(serial_y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_x['Date / Time'], y=group_x['Temperature'], mode='lines', name=serial_x))
        fig.add_trace(go.Scatter(x=group_y['Date / Time'], y=group_y['Temperature'], mode='lines', name=serial_y))
        return original_fig, fig, highest_affinity_fig

    elif hovered_elem_id == 'highest-affinity-heatmap' and highest_affinity_hover:
        point = highest_affinity_hover['points'][0]
        x, y = point['x'], point['y']
        serial_x, serial_y = x, y
        group_x = shipment_groups.get_group(serial_x)
        group_y = shipment_groups.get_group(serial_y)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=group_x['Date / Time'], y=group_x['Temperature'], mode='lines', name=serial_x))
        fig.add_trace(go.Scatter(x=group_y['Date / Time'], y=group_y['Temperature'], mode='lines', name=serial_y))
        return original_fig, reordered_fig, fig

    return original_fig, reordered_fig, highest_affinity_fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8090)
