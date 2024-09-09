import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Load augmented data from JSON files
def load_augmented_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_augmented_data(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# File paths
file_paths = {
    'Spike': 'augmented_spike.json',
    'Cool Defrost': 'augmented_cool_defrost.json',
    'Excursion': 'augmented_excursion.json',
    'Not Precooled': 'augmented_not_precooled.json',
    'Initial Ramp': 'augmented_initial_ramp.json',
    'Top Freezing': 'augmented_top_freezing.json',
    'Extended Drift': 'augmented_extended_drift.json'
}

# Load all data
augmented_spike = load_augmented_data(file_paths['Spike'])
augmented_cool_defrost = load_augmented_data(file_paths['Cool Defrost'])
augmented_excursion = load_augmented_data(file_paths['Excursion'])
augmented_not_precooled = load_augmented_data(file_paths['Not Precooled'])
augmented_initial_ramp = load_augmented_data(file_paths['Initial Ramp'])
augmented_top_freezing = load_augmented_data(file_paths['Top Freezing'])
augmented_extended_drift = load_augmented_data(file_paths['Extended Drift'])

# Class mapping
class_mapping = {
    'Spike': augmented_spike,
    'Cool Defrost': augmented_cool_defrost,
    'Excursion': augmented_excursion,
    'Not Precooled': augmented_not_precooled,
    'Initial Ramp': augmented_initial_ramp,
    'Top Freezing': augmented_top_freezing,
    'Extended Drift': augmented_extended_drift
}

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='View Time Series', children=[
            html.Div([
                html.H1("Time Series Augmentation Viewer"),
                html.Label("Select Class:"),
                dcc.Dropdown(
                    id='class-dropdown',
                    options=[{'label': key, 'value': key} for key in class_mapping.keys()],
                    value='Spike'
                ),
                html.Label("Select Serial Number:"),
                dcc.Dropdown(id='serial-dropdown'),
                dcc.Graph(id='time-series-plot')
            ])
        ]),
        dcc.Tab(label='Move Signal', children=[
            html.Div([
                html.H1("Move Signal to Another Class"),
                html.Label("Select Source Class:"),
                dcc.Dropdown(
                    id='source-class-dropdown',
                    options=[{'label': key, 'value': key} for key in class_mapping.keys()],
                    value='Spike'
                ),
                html.Label("Select Serial Number:"),
                dcc.Dropdown(id='source-serial-dropdown'),
                html.Label("Select Target Class:"),
                dcc.Dropdown(
                    id='target-class-dropdown',
                    options=[{'label': key, 'value': key} for key in class_mapping.keys()],
                    value='Cool Defrost'
                ),
                dcc.Graph(id='move-time-series-plot'),
                html.Button('Move Signal', id='move-signal-button'),
                html.Div(id='move-signal-output')
            ])
        ])
    ])
])

@app.callback(
    Output('serial-dropdown', 'options'),
    Output('serial-dropdown', 'value'),
    Input('class-dropdown', 'value')
)
def update_serial_dropdown(selected_class):
    serials = list(class_mapping[selected_class].keys())
    options = [{'label': serial, 'value': serial} for serial in serials]
    value = serials[0] if serials else None
    return options, value

@app.callback(
    Output('time-series-plot', 'figure'),
    Input('class-dropdown', 'value'),
    Input('serial-dropdown', 'value')
)
def update_time_series_plot(selected_class, selected_serial):
    data_dict = class_mapping[selected_class]
    original_ts = data_dict[selected_serial]['original']
    augmented_ts = data_dict[selected_serial]['augmented']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Time Series", "Augmented Time Series"))
    
    fig.add_trace(go.Scatter(y=original_ts, mode='lines', name='Original'), row=1, col=1)
    
    for i, aug_ts in enumerate(augmented_ts):
        fig.add_trace(go.Scatter(y=aug_ts, mode='lines', name=f'Augmented {i+1}'), row=1, col=2)
    
    fig.update_layout(title_text=f"Time Series for Serial {selected_serial} in Class {selected_class}")
    
    return fig

@app.callback(
    Output('source-serial-dropdown', 'options'),
    Output('source-serial-dropdown', 'value'),
    Input('source-class-dropdown', 'value')
)
def update_source_serial_dropdown(selected_class):
    serials = list(class_mapping[selected_class].keys())
    options = [{'label': serial, 'value': serial} for serial in serials]
    value = serials[0] if serials else None
    return options, value

@app.callback(
    Output('move-time-series-plot', 'figure'),
    Input('source-class-dropdown', 'value'),
    Input('source-serial-dropdown', 'value')
)
def update_move_time_series_plot(source_class, source_serial):
    data_dict = class_mapping[source_class]
    original_ts = data_dict[source_serial]['original']
    augmented_ts = data_dict[source_serial]['augmented']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Time Series", "Augmented Time Series"))
    
    fig.add_trace(go.Scatter(y=original_ts, mode='lines', name='Original'), row=1, col=1)
    
    for i, aug_ts in enumerate(augmented_ts):
        fig.add_trace(go.Scatter(y=aug_ts, mode='lines', name=f'Augmented {i+1}'), row=1, col=2)
    
    fig.update_layout(title_text=f"Time Series for Serial {source_serial} in Class {source_class}")
    
    return fig

@app.callback(
    Output('move-signal-output', 'children'),
    Input('move-signal-button', 'n_clicks'),
    State('source-class-dropdown', 'value'),
    State('source-serial-dropdown', 'value'),
    State('target-class-dropdown', 'value')
)
def move_signal(n_clicks, source_class, source_serial, target_class):
    if n_clicks is None:
        return ''
    if source_class == target_class:
        return 'Source and target classes must be different.'
    
    # Move the signal
    signal_data = class_mapping[source_class].pop(source_serial, None)
    if not signal_data:
        return 'Serial number not found in source class.'
    
    class_mapping[target_class][source_serial] = signal_data
    
    # Save updated JSON files
    save_augmented_data(file_paths[source_class], class_mapping[source_class])
    save_augmented_data(file_paths[target_class], class_mapping[target_class])
    
    return f'Successfully moved serial {source_serial} from {source_class} to {target_class}.'

if __name__ == '__main__':
    app.run_server(debug=True)
