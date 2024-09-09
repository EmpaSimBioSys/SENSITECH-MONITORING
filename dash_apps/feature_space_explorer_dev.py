import dash
import json
import ast
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import linregress, kurtosis, skew
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids
from preprocessing.feature_generation import FeatureGenerator 
from datasets.data_merger import ShipmentDataMerger


def str_to_float_list(s):
    try:
        return list(map(float, ast.literal_eval(s)))
    except:
        return []
    
def pad_or_clip(series, length):
    if len(series) > length:
        return series[:length]
    elif len(series) < length:
        return np.pad(series, (0, length - len(series)), 'constant')
    else:
        return series

# Function to load and merge data from different CSV files
def load_and_merge_data(file_paths, class_names):
    df_list = []
    for file_path, class_name in zip(file_paths, class_names):
        df = pd.read_csv(file_path)
        df['AnomalyClass'] = class_name
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Label data based on criteria with adjustable threshold
def label_based_on_threshold(dataframe, column_name, threshold):
    """
    Function to label data points as 'Anomaly' or 'Normal' based on a threshold in a given column.
    
    Args:
    dataframe (pd.DataFrame): The dataframe containing the data.
    column_name (str): The column name on which the threshold is applied.
    threshold (float): The threshold value for labeling.
    
    Returns:
    dict: A dictionary with serial numbers as keys and labels ('Anomaly' or 'Normal') as values.
    """
    labels = dataframe[column_name].apply(lambda x: 'Anomaly' if x > threshold else 'Normal')
    return dict(zip(dataframe['Serial no'], labels))


def compute_fft_magnitudes(series):
    n = len(series)
    yf = fft(series)
    xf = fftfreq(n, 1 / (4 * 60))  # assuming 15 min intervals, 4 samples per hour

    # Focus on positive frequencies
    positive_frequencies = xf[xf > 0]
    magnitudes = np.abs(yf[xf > 0])

    # Normalizing the magnitudes
    normalized_magnitudes = magnitudes / np.max(magnitudes) if len(magnitudes) > 0 else [0, 0, 0, 0]

    return normalized_magnitudes

def label_based_on_criteria(dataframe, criteria, threshold):
    def high_initial_temperature(series, threshold, period):
        period_length = int(period / 15)
        return criteria if np.mean(series[:period_length] > threshold) > 0 else 'normal'

    def cyclical_components(series, threshold):
        magnitudes = compute_fft_magnitudes(series)

        mean_amp = np.mean(magnitudes) # 75th percentile, 0.11 for median 
        if mean_amp > threshold:
            return criteria
        return 'normal'

    def peakiness(series):
        peaks, _ = find_peaks(series)
        return criteria if len(peaks) > 10 else 'normal'

    def degree_minutes(series, threshold, below=False):
        if below:
            return criteria if np.mean(series < threshold) > 0 else 'normal'
        return criteria if np.mean(series > threshold) > 0 else 'normal'

    def high_initial_positive_trend(time_series, temp_series, period):
        period_length = int(period / 15)
        slope, intercept, r_value, p_value, std_err = linregress(time_series[:period_length], temp_series[:period_length])
        return criteria if slope > 0 else 'normal'

    def overall_positive_trend(time_series, temp_series):
        slope, intercept, r_value, p_value, std_err = linregress(time_series, temp_series)
        return criteria if slope > 0 else 'normal'

    def average_variation(series, interval):
        interval_length = int(interval / 15)  # Convert interval to the number of data points
        if interval_length >= len(series):
            return 'normal'  # If interval is longer than the series, we return 'normal' as it cannot be computed

        rolling_gradients = []
        for i in range(len(series) - interval_length):
            window = series[i:i + interval_length]
            gradient = np.mean(np.abs(np.diff(window)))
            rolling_gradients.append(gradient)
        
        avg_gradient = np.mean(rolling_gradients)
        return criteria if avg_gradient > 0.5 else 'normal' 

    criteria_func_map = {
        'High_init_temp_2hr': lambda g: high_initial_temperature(g['Temperature'].values, 6, 2 * 60),
        'High_init_temp_4hr': lambda g: high_initial_temperature(g['Temperature'].values, 6, 4 * 60),
        'High_init_temp_6hr': lambda g: high_initial_temperature(g['Temperature'].values, 6, 6 * 60),
        'Cyclical_median': lambda g: cyclical_components(g['Temperature'].values, threshold),
        'Peakiness': lambda g: peakiness(g['Temperature'].values),
        'Degree_minutes_above_2C': lambda g: degree_minutes(g['Temperature'].values, 2),
        'Degree_minutes_above_6C': lambda g: degree_minutes(g['Temperature'].values, 6),
        'Degree_minutes_below_-1C': lambda g: degree_minutes(g['Temperature'].values, -1, below=True),
        'High_init_pos_trend_2hr': lambda g: high_initial_positive_trend(g['Relative Time'].values, g['Temperature'].values, 2 * 60),
        'High_init_pos_trend_4hr': lambda g: high_initial_positive_trend(g['Relative Time'].values, g['Temperature'].values, 4 * 60),
        'High_init_pos_trend_6hr': lambda g: high_initial_positive_trend(g['Relative Time'].values, g['Temperature'].values, 6 * 60),
        'Overall_pos_trend': lambda g: overall_positive_trend(g['Relative Time'].values, g['Temperature'].values),
        'Avg_var_30min': lambda g: average_variation(g['Temperature'].values, 30),
        'Avg_var_1hr': lambda g: average_variation(g['Temperature'].values, 60),
        'Avg_var_2hr': lambda g: average_variation(g['Temperature'].values, 120),
        'Avg_var_3hr': lambda g: average_variation(g['Temperature'].values, 180)
    }

    labels_dict = {}
    grouped_data = dataframe.groupby('Serial no')

    for name, group in grouped_data:
        label = criteria_func_map[criteria](group)
        labels_dict[name] = label if label != 'normal' else 'normal'

    return labels_dict


# Load manually labelled dataset 
# File paths and class names
file_paths = [
    "data/classified/7_class/data_spikes.csv",
    "data/classified/7_class/data_cyclical_events.csv",
    "data/classified/7_class/data_excursion.csv",
    "data/classified/7_class/data_not_precooled.csv",
    "data/classified/7_class/data_normal.csv",
    "data/classified/7_class/data_initial_ramp.csv",
    "data/classified/7_class/data_below_freezing.csv",
    "data/classified/7_class/data_extended_drift.csv"
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

# Columns for criteria labels
criteria_columns = [
    'High_init_temp_2hr', 'High_init_temp_4hr', 'High_init_temp_6hr',
    'Cyclical_25th_percentile', 'Cyclical_median', 'Cyclical_mean', 'Cyclical_75th_percentile',
    'Peakiness', 'Degree_minutes_above_2C', 'Degree_minutes_above_6C', 'Degree_minutes_below_-1C',
    'High_init_pos_trend_2hr', 'High_init_pos_trend_4hr', 'High_init_pos_trend_6hr',
    'Overall_pos_trend', 'Avg_var_30min', 'Avg_var_1hr', 'Avg_var_2hr', 'Avg_var_3hr'
]

multi_label_column = ['multi_label']

# Load distance matrix ordered serial numbers
with open('data/dtw_distance_matrix/serial_numbers.json', 'r') as json_file:
    serial_numbers = json.load(json_file)

# Load and merge data
merged_data = load_and_merge_data(file_paths, class_names)

# Create a mapping from Serial no to AnomalyClass
serial_to_anomaly_class = dict(zip(merged_data['Serial no'], merged_data['AnomalyClass']))

# Load the classified dataset with soft criteria labels
classified_df = pd.read_csv('data/classified/2_class/classified_data_2_classes_relaxed.csv')
classified_df['Date / Time'] = pd.to_datetime(classified_df['Date / Time'])

# Create a mapping from Serial no to 2-class problem labels
serial_to_2class_label = dict(zip(classified_df['Serial no'], classified_df['Label']))


# Load the classified dataset with hard criteria labels
classified_df_strong = pd.read_csv('data/classified/classified_data_criteria.csv')
classified_df_strong['Date / Time'] = pd.to_datetime(classified_df_strong['Date / Time'])
classified_df_strong['prevalent_anomaly'] = classified_df_strong[criteria_columns].idxmax(axis=1)
classified_df_strong['multi_label'] = classified_df_strong['multi_label'].apply(str_to_float_list)
classified_df_strong['average_score'] = classified_df_strong['multi_label'].apply(lambda x: np.mean(x))
classified_df_strong['avg_multi_label'] = ""
classified_df_strong.loc[classified_df_strong['average_score'] > 0.5, 'avg_multi_label'] = "Anomaly"
classified_df_strong.loc[classified_df_strong['average_score'] <= 0.5, 'avg_multi_label'] = "Normal"

# Create a mapping from Serial no to prevalent criteria labels
serial_to_prevalent_label = dict(zip(classified_df_strong['Serial no'], classified_df_strong['prevalent_anomaly']))
serial_to_average_label = dict(zip(classified_df_strong['Serial no'], classified_df_strong['avg_multi_label']))


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Initialize FeatureGenerator and extract all featureconfig_path = 'config.ini'
coop_path = "data/all_data_combined_meta.csv"
bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"

# Initialize FeatureGenerator and extract all features
config_path = 'config.ini'
coop_path = "data/all_data_combined_meta.csv"
bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"

shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
data = shipment_merger.resample_time_series()
data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
feature_generator = FeatureGenerator(data, config_path, coop_path, bama_path)
feature_generator.data = feature_generator.load_data(df=False)

all_features, _, _, _ = feature_generator.extract_all_features()
all_features = feature_generator.normalize_features(all_features)


# Load and process distance matrix
distance_matrix = np.load("data/dtw_distance_matrix/dtw_distance_matrix.npy")

# Substitute inf values with a large number (4 times the maximum finite value in the matrix)
max_finite_value =  np.nanmax(distance_matrix[np.isfinite(distance_matrix)])
large_number = max_finite_value * 2
distance_matrix[np.isinf(distance_matrix)] = large_number

# Handle NaN values if present, substituting them with the large number
distance_matrix[np.isnan(distance_matrix)] = large_number

# Convert distance matrix to affinity matrix
beta = 1
distance_matrix_std = np.nanstd(distance_matrix)
if distance_matrix_std == 0:
    distance_matrix_std = 1
affinity_matrix = np.exp(-beta * distance_matrix / distance_matrix_std)

serial_to_spectral_label = {}

# Get the list of features in the dictionary
feature_keys = list(next(iter(all_features.values())).keys())

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='feature-set-dropdown',
                options=[{'label': key, 'value': key} for key in feature_keys],
                multi=True,
                placeholder='Select feature sets'
            ),
        ], width=6),
    ]),
    dcc.Slider(
        id='perplexity-slider',
        min=10,
        max=500,
        step=10,
        marks={i: str(i) for i in range(10, 501, 10)},
        value=30,
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    dcc.Input(
        id='n-clusters-input',
        type='number',
        value=3,
        min=2,
        max=50,
        step=1,
        placeholder='Number of clusters'
    ),
    dcc.Dropdown(
        id='label-type-dropdown',
        options=[
            {'label': '2-Class Problem', 'value': '2class'},
            {'label': '8-Class Problem', 'value': '8class'},
            {'label': 'Spectral Clustering', 'value': 'spectral'},
            {'label': 'Prevalent Anomaly', 'value': 'prevalent_anomaly'},
            {'label': 'Average Multi-label', 'value': 'avg_multi_label'},
            {'label': 'Criteria-based', 'value': 'criteria-based'},
            {'label': 'Criteria-based-2', 'value': 'criteria-based-2'} 
        ],
        value='2class',
        placeholder='Select label type'
    ),
    dcc.Dropdown(
        id='criteria-label-dropdown',
        options=[{'label': col, 'value': col} for col in criteria_columns],
        placeholder='Select criteria label'
    ),
    dcc.Slider(
        id='criteria-threshold-slider',
        min=0,
        max=1,
        step=0.1,
        marks={i/10: str(i/10) for i in range(0, 11)},
        value=0.5,
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    dbc.Tabs([
        dbc.Tab(label='PCA', tab_id='tab-pca'),
        dbc.Tab(label='t-SNE', tab_id='tab-tsne'),
        dbc.Tab(label='K-Means', tab_id='tab-kmeans'),
        dbc.Tab(label='K-Medoids', tab_id='tab-kmedoids'),
    ], id='tabs', active_tab='tab-pca'),
    html.Div(id='tab-content'),
    html.Div(id='cluster-statistics', style={'margin-top': '20px'}),
    html.Div(id='cluster-properties', style={'margin-top': '20px'}),
    html.Div(id='average-trip-length', style={'margin-top': '20px', 'font-weight': 'bold'})
])

@app.callback(Output('tab-content', 'children'), 
              [Input('tabs', 'active_tab'), 
               Input('feature-set-dropdown', 'value'), 
               Input('perplexity-slider', 'value'), 
               Input('n-clusters-input', 'value'),
               Input('label-type-dropdown', 'value'),
               Input('criteria-label-dropdown', 'value'),
               Input('criteria-threshold-slider', 'value')])
def render_content(tab, selected_feature_sets, perplexity, n_clusters, label_type, criteria_label, criteria_threshold):
    if not selected_feature_sets:
        return html.Div("Please select at least one feature set.")
    
    # Combine selected features
    combined_features = []
    serial_nos = list(all_features.keys())
    for serial_no in serial_nos:
        combined_vector = []
        for feature_set in selected_feature_sets:
            combined_vector.extend(all_features[serial_no][feature_set])
        combined_features.append(combined_vector)
    combined_features = np.array(combined_features)

    if "combined" not in selected_feature_sets and len(selected_feature_sets) > 1:
        combined_features = (combined_features - np.mean(combined_features)) / np.median(combined_features)

    # Generate labels based on the selected label type
    if label_type == 'spectral':
        labels = [serial_to_anomaly_class.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif label_type == '8class':
        labels = [serial_to_anomaly_class.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif label_type == '2class':
        labels = [serial_to_2class_label.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif label_type == 'prevalent_anomaly':
        labels = [serial_to_prevalent_label.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif label_type == 'avg_multi_label':
        labels = [serial_to_average_label.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif criteria_label != "" and label_type == 'criteria-based':
        criteria_labels = label_based_on_threshold(classified_df_strong, criteria_label, criteria_threshold)
        labels = [criteria_labels.get(serial_no, 'Unknown') for serial_no in serial_nos] 
    elif criteria_label is None and label_type == 'criteria-based':
        labels = [serial_to_spectral_label.get(serial_no, 'Unknown') for serial_no in serial_nos]
    elif criteria_label != "" and label_type == 'criteria-based-2':
        criteria_labels = label_based_on_criteria(data, criteria_label, threshold=criteria_threshold)
        labels = [criteria_labels.get(serial_no, 'Unknown') for serial_no in serial_nos] 
    else:
        labels = [serial_to_anomaly_class.get(serial_no, 'Unknown') for serial_no in serial_nos]

     # Add trip length calculation
    # trip_lengths = [[data[data['Serial no'] == sn].shape[0] * 15 / 60, data[data['Serial no'] == sn].shape[0] * 15 / 1440] for sn in serial_nos]
    trip_lengths = [data[data['Serial no'] == sn].shape[0] * 15 / 60 for sn in serial_nos]

    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'x': combined_features[:, 0],
        'y': combined_features[:, 1],
        'serial_no': serial_nos,
        'label': labels,
        'trip_length': trip_lengths
    })


    if tab == 'tab-pca':
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_features)
        plot_data['x'] = pca_result[:, 0]
        plot_data['y'] = pca_result[:, 1]
        fig = px.scatter(
            plot_data,
            x='x',
            y='y',
            hover_name='serial_no',
            labels={'x': 'PCA Axis 1', 'y': 'PCA Axis 2'},
            color='label',
            custom_data=['trip_length']
        )
        fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
        fig.update_layout(title='PCA Visualization', dragmode='select')
        return html.Div([
            dcc.Graph(id='pca-plot', figure=fig),
            dcc.Graph(id='time-series-plot-pca', style={'height': '400px'})
        ])
    elif tab == 'tab-tsne':
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_result = tsne.fit_transform(combined_features)
        plot_data['x'] = tsne_result[:, 0]
        plot_data['y'] = tsne_result[:, 1]
        fig = px.scatter(
            plot_data,
            x='x',
            y='y',
            hover_name='serial_no',
            labels={'x': 't-SNE Axis 1', 'y': 't-SNE Axis 2'},
            color='label',
            custom_data=['trip_length']
        )
        fig.update_traces(marker=dict(size=8), selector=dict(mode='markers'))
        fig.update_layout(title=f't-SNE Visualization (perplexity={perplexity})', dragmode='select')
        return html.Div([
            dcc.Graph(id='tsne-plot', figure=fig),
            dcc.Graph(id='time-series-plot-tsne', style={'height': '400px'})
        ])
    elif tab == 'tab-kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(combined_features)
        cluster_statistics = calculate_cluster_statistics(kmeans_labels, n_clusters, combined_features)
        return html.Div([
            generate_cluster_plots('K-Means', kmeans_labels, n_clusters, label_type, kmeans.cluster_centers_, combined_features, criteria_label, criteria_threshold),
            generate_cluster_statistics_table(cluster_statistics),
            generate_cluster_properties_table(kmeans_labels, n_clusters)
        ])
    elif tab == 'tab-kmedoids':
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, metric='euclidean')
        kmedoids_labels = kmedoids.fit_predict(combined_features)
        cluster_statistics = calculate_cluster_statistics(kmedoids_labels, n_clusters, combined_features)
        return html.Div([
            generate_cluster_plots('K-Medoids', kmedoids_labels, n_clusters, label_type, kmedoids.cluster_centers_, combined_features, criteria_label, criteria_threshold),
            generate_cluster_statistics_table(cluster_statistics),
            generate_cluster_properties_table(kmedoids_labels, n_clusters)
        ])
    
def generate_cluster_plots(algorithm, labels, n_clusters, label_type, cluster_centers, combined_features, criteria_label, criteria_threshold):
    
    # Perform spectral clustering using the affinity matrix
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    spectral_labels = spectral_clustering.fit_predict(affinity_matrix)

    # Map spectral clustering labels to serial numbers
    serial_to_spectral_label = dict(zip(serial_numbers, spectral_labels))
    
    cluster_figs = []
    for cluster_id in range(n_clusters):
        cluster_serial_nos = [serial_no for serial_no, label in zip(all_features.keys(), labels) if label == cluster_id]
        cluster_data = data[data['Serial no'].isin(cluster_serial_nos)]

        # Determine the longest sequence in the cluster
        max_length = max(cluster_data.groupby('Serial no').size())

        # Pad or clip all time series to the maximum length
        padded_series = []
        for serial_no in cluster_serial_nos:
            series = cluster_data[cluster_data['Serial no'] == serial_no]['Temperature'].values
            padded_series.append(pad_or_clip(series, max_length))

        # Convert to numpy array for easier manipulation
        padded_series = np.array(padded_series)

        # Compute mean and median across the stacked time series
        mean_series = np.mean(padded_series, axis=0)
        median_series = np.median(padded_series, axis=0)

        fig = go.Figure()

        for series, serial_no in zip(padded_series, cluster_serial_nos):

            if label_type == 'spectral':
                label = serial_to_spectral_label.get(serial_no, 'Unknown')
            elif label_type == '8class':
                label = serial_to_anomaly_class.get(serial_no, 'Unknown')
            elif label_type == '2class':
                label = serial_to_2class_label.get(serial_no, 'Unknown')
            elif label_type == 'prevalent_anomaly':
                label = serial_to_prevalent_label.get(serial_no, 'Unknown')
            elif label_type == 'avg_multi_label':
                label = serial_to_average_label.get(serial_no, 'Unknown')
            elif criteria_label and label_type == 'criteria_based':
                criteria_labels_1 = label_based_on_threshold(classified_df_strong, criteria_label, criteria_threshold)
                label = criteria_labels_1.get(serial_no, 'Unknown')
            elif criteria_label is None and label_type == 'criteria-based':
                label = serial_to_spectral_label.get(serial_no, 'Unknown')
            elif criteria_label != "" and label_type == 'criteria-based-2':
                criteria_labels_2 = label_based_on_criteria(data, criteria_label, threshold=criteria_threshold)
                label = criteria_labels_2.get(serial_no, 'Unknown')
            else:
                print("No labelling configuration found")
                label = serial_to_spectral_label.get(serial_no, 'Unknown')

            fig.add_trace(go.Scatter(x=np.arange(max_length), y=series, mode='lines', line=dict(color='grey', width=1), name=f"{serial_no}{label}"))
        fig.add_trace(go.Scatter(x=np.arange(max_length), y=mean_series, mode='lines', line=dict(color='red', width=2), name='Mean'))
        fig.add_trace(go.Scatter(x=np.arange(max_length), y=median_series, mode='lines', line=dict(color='blue', width=2), name='Median'))

        # Calculate Euclidean distances between cluster centers
        distances = np.linalg.norm(cluster_centers - cluster_centers[cluster_id], axis=1)
        distance_annotations = [f"Distance to Cluster {i + 1}: {dist:.2f}" for i, dist in enumerate(distances) if i != cluster_id]

        # Add cluster quality measures
        silhouette_avg = silhouette_score(combined_features, labels) if len(set(labels)) > 1 else np.nan
        wcss = np.sum((combined_features - cluster_centers[labels]) ** 2)

        # Add annotations for distances and quality measures
        annotations = [f"Silhouette Score: {silhouette_avg:.2f}", f"WCSS: {wcss:.2f}"] + distance_annotations
        fig.add_annotation(text="<br>".join(annotations), showarrow=False, xref="paper", yref="paper", x=-0.1, y=1, bordercolor="black", borderwidth=1)

        fig.update_layout(title=f'Cluster {cluster_id + 1}', xaxis_title='Data Point Index', yaxis_title='Temperature')

        cluster_figs.append(dcc.Graph(figure=fig, style={'height': '400px'}))

    return html.Div(cluster_figs)

def calculate_cluster_statistics(labels, n_clusters, combined_features):
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_features = combined_features[cluster_indices]

        num_shipments = len(cluster_indices)
        avg_temperature = np.mean([data[data['Serial no'] == serial_numbers[i]]['Temperature'].mean() for i in cluster_indices])
        std_temperature = np.std([data[data['Serial no'] == serial_numbers[i]]['Temperature'].mean() for i in cluster_indices])
        avg_length = np.mean([len(data[data['Serial no'] == serial_numbers[i]]) for i in cluster_indices])
        avg_gradient = np.mean([np.gradient(data[data['Serial no'] == serial_numbers[i]]['Temperature']).mean() for i in cluster_indices])
        avg_initial_temp = np.mean([data[data['Serial no'] == serial_numbers[i]]['Temperature'].iloc[0] for i in cluster_indices])
        avg_trend = np.mean([np.polyfit(np.arange(len(data[data['Serial no'] == serial_numbers[i]])), data[data['Serial no'] == serial_numbers[i]]['Temperature'], 1)[0] for i in cluster_indices])
        avg_peakiness = np.mean([len(find_peaks(data[data['Serial no'] == serial_numbers[i]]['Temperature'])[0]) for i in cluster_indices])

        # freq_features = [all_features[serial_numbers[i]]['frequency_200'] for i in cluster_indices]
        # avg_freq_vector = np.mean(freq_features, axis=0)
        # power_cyclical = np.mean(avg_freq_vector)

        cluster_stats.append({
            'Cluster': cluster_id + 1,
            'Num Shipments': num_shipments,
            'Avg Temperature': avg_temperature,
            'Std Temperature': std_temperature,
            'Avg Length': avg_length,
            'Avg Gradient': avg_gradient,
            'Avg Initial Temp': avg_initial_temp,
            'Avg Trend': avg_trend,
            'Avg Peakiness': avg_peakiness
            # 'Power Cyclical': power_cyclical
        })

    return cluster_stats

def generate_cluster_statistics_table(cluster_statistics):
    columns = ['Cluster', 'Num Shipments', 'Avg Temperature', 'Std Temperature', 'Avg Length', 'Avg Gradient', 'Avg Initial Temp', 'Avg Trend', 'Avg Peakiness']
    table_header = [html.Thead(html.Tr([html.Th(col) for col in columns]))]
    table_body = [html.Tbody([html.Tr([html.Td(cluster_stats[col]) for col in columns]) for cluster_stats in cluster_statistics])]
    return dbc.Table(table_header + table_body, bordered=True, dark=True, hover=True, responsive=True)


def generate_cluster_properties_table(labels, n_clusters):
    cluster_props = []
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_data = data[data['Serial no'].isin([serial_numbers[i] for i in cluster_indices])]
        
        trip_lengths = cluster_data.groupby('Serial no').size() * 15 / 60  # assuming 15 min intervals, convert to hours
        avg_temps = cluster_data.groupby('Serial no')['Temperature'].mean()
        min_temps = cluster_data.groupby('Serial no')['Temperature'].min()
        max_temps = cluster_data.groupby('Serial no')['Temperature'].max()
        kurtosis_vals = cluster_data.groupby('Serial no')['Temperature'].apply(kurtosis)
        skew_vals = cluster_data.groupby('Serial no')['Temperature'].apply(skew)

        cluster_props.append({
            'Cluster': cluster_id + 1,
            'Avg Trip Length (hrs)': trip_lengths.mean(),
            'Avg Temperature': avg_temps.mean(),
            'Min Temperature': min_temps.mean(),
            'Max Temperature': max_temps.mean(),
            'Kurtosis': kurtosis_vals.mean(),
            'Skewness': skew_vals.mean()
        })

    columns = ['Cluster', 'Avg Trip Length (hrs)', 'Avg Temperature', 'Min Temperature', 'Max Temperature', 'Kurtosis', 'Skewness']
    table_header = [html.Thead(html.Tr([html.Th(col) for col in columns]))]
    table_body = [html.Tbody([html.Tr([html.Td(cluster_prop[col]) for col in columns]) for cluster_prop in cluster_props])]
    return dbc.Table(table_header + table_body, bordered=True, dark=True, hover=True, responsive=True)


@app.callback(
    Output('average-trip-length', 'children'),
    [Input('pca-plot', 'selectedData'),
     Input('tsne-plot', 'selectedData')]
)
def display_average_trip_length(pca_selected_data, tsne_selected_data):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "Average Trip Length: N/A"

    selected_data = None
    if ctx.triggered[0]['prop_id'].startswith('pca-plot'):
        selected_data = pca_selected_data
    elif ctx.triggered[0]['prop_id'].startswith('tsne-plot'):
        selected_data = tsne_selected_data

    if selected_data is None or not selected_data['points']:
        return "Average Trip Length: N/A"

    selected_trip_lengths = [point['customdata'][0] for point in selected_data['points']]
    average_trip_length = np.mean(selected_trip_lengths) if selected_trip_lengths else 0

    return f"Average Trip Length: {average_trip_length:.2f} hrs"

@app.callback(
    Output('time-series-plot-pca', 'figure'),
    [Input('pca-plot', 'hoverData'), Input('label-type-dropdown', 'value')]
)
def update_time_series_plot_pca(hover_data, label_type):
    if hover_data is None:
        return go.Figure()

    point_data = hover_data['points'][0]
    serial_no = point_data['hovertext'].split("_")[0]
    trip_length_hours = point_data['customdata'][0]  # Access the first element

    group = data[data["Serial no"] == serial_no]
    temperature_data = group['Temperature'].values
    time_data = group['Date / Time'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=temperature_data, mode='lines', name='Temperature', line=dict(color='blue')))
    fig.update_layout(title=f'Time Series for Serial no: {serial_no} (Trip Length: {trip_length_hours:.2f} hrs)', xaxis_title='Date / Time', yaxis_title='Temperature')

    return fig

@app.callback(
    Output('time-series-plot-tsne', 'figure'),
    [Input('tsne-plot', 'hoverData'), Input('label-type-dropdown', 'value')]
)
def update_time_series_plot_tsne(hover_data, label_type):
    if hover_data is None:
        return go.Figure()

    point_data = hover_data['points'][0]
    serial_no = point_data['hovertext'].split("_")[0]
    trip_length_hours = point_data['customdata'][0]  # Access the first element

    group = data[data["Serial no"] == serial_no]
    temperature_data = group['Temperature'].values
    time_data = group['Date / Time'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=temperature_data, mode='lines', name='Temperature', line=dict(color='blue')))
    fig.update_layout(title=f'Time Series for Serial no: {serial_no} (Trip Length: {trip_length_hours:.2f} hrs)', xaxis_title='Date / Time', yaxis_title='Temperature')

    return fig

def update_time_series_plot(hover_data, plot_id, label_type):
    if hover_data is None:
        return go.Figure()

    serial_no = hover_data['points'][0]['hovertext'].split("_")[0]
    group = data[data["Serial no"] == serial_no]
    temperature_data = group['Temperature'].values
    time_data = group['Date / Time'].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_data, y=temperature_data, mode='lines', name='Temperature', line=dict(color='blue')))
    fig.update_layout(title=f'Time Series for Serial no: {serial_no}', xaxis_title='Date / Time', yaxis_title='Temperature')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8090)