import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask import Flask, request, render_template, send_file, jsonify
from datasets.data_merger import ShipmentDataMerger 
from preprocessing.trim import TimeSeriesTrimmer

matplotlib.use('Agg')

app = Flask(__name__)

# Function to check and load or initialize dataframe
def load_or_initialize_csv(file_path, columns):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame(columns=columns)

# Define file paths
file_paths = {
    "norm": "data/data_norm.csv",
    "spike": "data/data_spike.csv",
    "not_precooled": "data/data_not_precooled.csv",
    "initial_ramp": "data/data_initial_ramp.csv",
    "cool_defrost": "data/data_cool_defrost.csv",
    "excursion": "data/data_excursion.csv",
    "extended_drift": "data/data_extended_drift.csv",
    "chilling injury": "data/data_chilling_injury.csv",
    "questionable_start_stop": "data/data_questionable_start_stop.csv",
}

# Load existing or initialize new DataFrames
df_dict = {key: load_or_initialize_csv(path, ['Date / Time', 'Serial no', 'Temperature', 'H_ShipmentId', 'OriginCityorTown', 'DestinationCityorTown'])
           for key, path in file_paths.items()}

# Get all existing serial numbers from the loaded DataFrames
existing_serials = pd.concat([df['Serial no'] for df in df_dict.values()]).unique()

# Load the data
config_path = 'config.ini'
coop_path = "data/all_data_combined_meta.csv"
bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"
shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
data = shipment_merger.merged_dataframe
data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()

# Filter new shipments not in existing data
new_shipments = data[~data['Serial no'].isin(existing_serials)]
grouped_new_shipments = new_shipments.groupby("Serial no")

# Total number of shipments
total_shipments = 1256

@app.route('/')
def index():
    counts = {key: df['Serial no'].nunique() for key, df in df_dict.items()}
    total_classified = sum(counts.values())
    return render_template('index.html', serials=grouped_new_shipments.groups.keys(), counts=counts, total_classified=total_classified, total_shipments=total_shipments)

@app.route('/plot/<serial_no>')
def plot(serial_no):
    group = grouped_new_shipments.get_group(serial_no)

    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(group["Date / Time"], group["Temperature"])

    # Calculate error margins
    upper_bound = group["Temperature"] + 0.5
    lower_bound = group["Temperature"] - 0.5
    ax.fill_between(group["Date / Time"], lower_bound, upper_bound, color='blue', alpha=0.2, label='±0.5°C Error Margin')

    # Draw horizontal lines at specific temperatures
    ax.axhline(y=-1, color='violet', linestyle='--', linewidth=3, label='-1°C Boundary')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=3, label='0°C Boundary')
    ax.axhline(y=2, color='g', linestyle='--', linewidth=3, label='2°C Boundary')
    ax.axhline(y=6, color='b', linestyle='--', linewidth=3, label='6°C Boundary')

    ax.legend()  # Show the legend to explain the lines
    ax.grid(True)  # Optionally, turn on the grid for better readability
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (° C)")
    ax.set_title(f"Shipment sensor {serial_no}", fontsize=10)

    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    plt.close(fig)
    return send_file(BytesIO(output.getvalue()), mimetype='image/png')

@app.route('/classify', methods=['POST'])
def classify():
    serial_no = request.form['serial_no']
    classification = int(request.form['classification'])

    group = grouped_new_shipments.get_group(serial_no)
    category = ["spike", "excursion", "not_precooled", "initial_ramp", "cool_defrost", "extended_drift", "norm", "chilling injury", "questionable_start_stop"][classification]
    df_dict[category] = df_dict[category]._append(group)
    df_dict[category].to_csv(file_paths[category], index=False)

    counts = {key: df['Serial no'].nunique() for key, df in df_dict.items()}
    total_classified = sum(counts.values())
    return jsonify({"message": f"Shipment {serial_no} classified as {category}.", "counts": counts, "total_classified": total_classified})

if __name__ == '__main__':
    app.run(debug=True)
