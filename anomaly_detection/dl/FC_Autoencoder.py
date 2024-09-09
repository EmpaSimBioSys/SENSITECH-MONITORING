import os
import argparse
import yaml
import torch
import torch.nn.init as init
import torch.nn.functional as F
import copy
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from datasets.coop import CoopData
from datasets.data_merger import ShipmentDataMerger
from preprocessing.trim import TimeSeriesTrimmer
from preprocessing.reshape import TimeSeriesReshaper

# Define the device to run training on between CPU and GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_yaml_config(config_path: str) -> dict:
    """Read and parse a YAML configuration file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train an autoencoder model.')
    parser.add_argument('--config', type=str, default= 'anomaly_detection/dl/config.yaml', help='Path to the configuration file')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--seq_len', type=int, default=300, help='Sequence length for the input data')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension for the encoder')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--input_dim', type=int, default=300, help='Input dimension for the data')
    return parser.parse_args()


def get_model_filename(base_dir: str, base_name: str, params: dict) -> str:
    """Generate a unique filename for the model based on the run parameters."""
    current_date = datetime.now().strftime("%Y-%m-%d")
    param_str = "_".join([f"{key}-{val}" for key, val in params.items()])
    filename = f"{base_name}_{current_date}_{param_str}.pth"
    full_path = os.path.join(base_dir, filename)
    
    i = 1
    while os.path.exists(full_path):
        filename = f"{base_name}_{current_date}_{param_str}_{i}.pth"
        full_path = os.path.join(base_dir, filename)
        i += 1
    
    return full_path


class CNNEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * (input_dim // 2), embedding_dim)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int = 64, output_dim: int = 4):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, output_dim)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for the model."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super(Autoencoder, self).__init__()
        self.encoder = CNNEncoder(input_dim, embedding_dim).to(device)
        self.decoder = Decoder(embedding_dim, input_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    
def resample_time_series(df: pd.DataFrame, interval: str = '15min', fill_method: str = 'ffill') -> pd.DataFrame:
    """Resample each time series to a specified interval."""
    resampled_data = []

    def resample_group(group: pd.DataFrame):
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
        
        resampled_data.append(resampled_group.reset_index())

    df.groupby('Serial no').apply(resample_group)
    resampled_df = pd.concat(resampled_data, ignore_index=True).infer_objects()
    
    return resampled_df


def create_dataset(df: pd.DataFrame, serial_no: str) -> TensorDataset:
    """Create a PyTorch dataset from a DataFrame."""
    df.drop(columns=serial_no, inplace=True)
    sequences = df.astype(np.float32).to_numpy()
    dataset = torch.tensor(sequences).float()
    return TensorDataset(dataset)


def plot_actual_vs_predicted_wandb(seq_true: np.ndarray, seq_pred: np.ndarray, epoch: int, idx: int, batch_idx: int, scaler: StandardScaler):
    """Plot and log actual vs predicted sequences to WandB."""
    seq_true_original = scaler.inverse_transform(seq_true.reshape(-1, 1)).flatten()
    seq_pred_original = scaler.inverse_transform(seq_pred.reshape(-1, 1)).flatten()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(seq_true_original))), y=seq_true_original, mode='lines', name='True', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(range(len(seq_pred_original))), y=seq_pred_original, mode='lines', name='Predicted', line=dict(color='red')))
    
    fig.update_layout(
        title=f'Epoch {epoch} - Sequence {idx} - Batch {batch_idx}',
        xaxis_title='Time Steps',
        yaxis_title='Temperature (°C)',
        legend=dict(x=0, y=1, traceorder='normal', font=dict(size=12)),
        font=dict(size=10),
        width=800,
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    wandb.log({f"Validation Plot Epoch {epoch} Seq {idx} Batch {batch_idx}": fig}, step=epoch)


def train_model(model: nn.Module, train_dataset: TensorDataset, val_dataset: TensorDataset, n_epochs: int, scaler: StandardScaler, batch_size: int) -> tuple:
    """Train the autoencoder model."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(wandb.config["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    criterion = nn.MSELoss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            seq_true = batch[0].to(device)
            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx == 2:
                    break

                seq_true = batch[0].to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())
                
                for i in range(seq_true.size(0)):
                    plot_actual_vs_predicted_wandb(
                        seq_true[i].cpu().numpy(), 
                        seq_pred[i].cpu().numpy(), 
                        epoch, 
                        i, 
                        batch_idx, 
                        scaler
                    )

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})
        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        scheduler.step()

    model.load_state_dict(best_model_wts)
    return model.eval(), history


def normalize_dataset(df: pd.DataFrame, serial_no_col: str, feature_col: str) -> tuple:
    """Normalize the dataset."""
    df_copy = df.copy()
    scaler = StandardScaler()
    df_copy[feature_col] = scaler.fit_transform(df_copy[[feature_col]])
    return df_copy, scaler


def main():
    args = parse_args()
    config = read_yaml_config(args.config)
    
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.seq_len is not None:
        config['seq_len'] = args.seq_len
    if args.embedding_dim is not None:
        config['embedding_dim'] = args.embedding_dim
    if args.input_dim is not None:
        config['input_dim'] = args.input_dim

    wandb.login(force=True)
    wandb.init(project='Autoencoder', entity='divineod')
    wandb.config.update(config)

    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU Index: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    data_features = ['Date / Time', 'Serial no', 'Temperature']
    coop_shipments_norm = CoopData("data/classified/7_class/data_normal.csv", feature_list=data_features, dependent_var='Temperature')
    trimmer = TimeSeriesTrimmer(coop_shipments_norm.data, temperature_column='Temperature')
    trimmed_coop_shipments_norm = trimmer.trim_time_series()
    resampled_coop_shipments_norm = resample_time_series(trimmed_coop_shipments_norm, interval='15min', fill_method='ffill')
    resampled_coop_shipments_norm['Relative Time'] = resampled_coop_shipments_norm.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    resampled_coop_shipments_norm["Relative Time"] = pd.to_timedelta(resampled_coop_shipments_norm["Relative Time"]).dt.total_seconds()
    grouped_shipments_coop_norm = resampled_coop_shipments_norm.groupby("Serial no")

    config_path = 'config.ini'
    coop_path = "data/all_data_combined_meta.csv"
    bama_path = "data/SWP_BAMA_Sensor_Shipment_berries.csv"
    shipment_merger = ShipmentDataMerger(coop_path, bama_path, config_path)
    data = shipment_merger.resample_time_series()
    data["Date / Time"] = pd.to_datetime(data["Date / Time"], utc=True)
    data['Relative Time'] = data.groupby('Serial no')['Date / Time'].transform(lambda x: (x - x.min()).dt.total_seconds())
    data["Relative Time"] = pd.to_timedelta(data["Relative Time"]).dt.total_seconds()
    shipment_groups = data.groupby('Serial no')

    normalized_trimmed_coop_shipments_norm, scaler_norm = normalize_dataset(trimmed_coop_shipments_norm, 'Serial no', 'Temperature')
    normalized_data, scaler_anom = normalize_dataset(data, 'Serial no', 'Temperature')

    reshaper_norm = TimeSeriesReshaper(normalized_trimmed_coop_shipments_norm, 'Date / Time', 'Serial no', 'Temperature')
    reshaped_normal_df = reshaper_norm.reshape(num_points=wandb.config["seq_len"])

    reshaper_anom = TimeSeriesReshaper(normalized_data, 'Date / Time', 'Serial no', 'Temperature')
    reshaped_anom_df = reshaper_anom.reshape(num_points=wandb.config["seq_len"])

    train_df, train_val_df = train_test_split(reshaped_normal_df, test_size=0.15, random_state=42)
    val_df, test_df = train_test_split(reshaped_anom_df, test_size=0.33, random_state=42)

    train_dataset = create_dataset(train_df, serial_no="Serial no")
    val_dataset = create_dataset(val_df, serial_no="Serial no")
    test_normal_dataset = create_dataset(train_val_df, serial_no="Serial no")
    test_anomaly_dataset = create_dataset(test_df, serial_no="Serial no")

    model = Autoencoder(input_dim=wandb.config["input_dim"], embedding_dim=wandb.config["embedding_dim"])
    print(f"The number of model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device)

    model, history = train_model(model, val_dataset, test_anomaly_dataset, n_epochs=wandb.config["epochs"], scaler=scaler_anom, batch_size=wandb.config["batch_size"])

    model_dir = "models/checkpoints"
    run_params = {
        "lr": wandb.config["learning_rate"],
        "bs": wandb.config["batch_size"],
        "epochs": wandb.config["epochs"],
        "seq_len": wandb.config["seq_len"],
        "embedding_dim": wandb.config["embedding_dim"],
        "input_dim": wandb.config["input_dim"]
    }

    model_filename = get_model_filename(model_dir, "autoencoder_model", run_params)
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")


if __name__ == "__main__":
    main()
