
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv("data/modelado/ds_modelado.csv")
df = df[df["estacion_del_anio"] == 1].copy()

# Definir target y features
target = "N_arribos_intervalo"
cols_a_excluir = [
    "N_arribos_intervalo", "id_recorrido", "id_usuario",
    "id_estacion_destino", "barrio_destino", "zona_destino_cluster",
    "cantidad_estaciones_cercanas_destino",
    "año_destino", "mes_destino", "dia_destino",
    "hora_destino", "minuto_destino", "segundo_destino",
    "duracion_recorrido"
]
features = [col for col in df.columns if col not in cols_a_excluir]


# Split estratificado
df_train, df_val = train_test_split(
    df,
    test_size=0.2,
    stratify=df["año_intervalo"],
    random_state=42
)

# Escalar
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(df_train[features])
X_val_scaled = scaler_X.transform(df_val[features])

y_train = df_train[target].values
y_val = df_val[target].values

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# Crear ventanas temporales
def crear_ventanas_temporales(X, y, ventana):
    X_seq, y_seq = [], []
    for i in range(ventana, len(X)):
        X_seq.append(X[i - ventana:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

ventana = 6
X_train_seq, y_train_seq = crear_ventanas_temporales(X_train_scaled, y_train_scaled, ventana)
X_val_seq, y_val_seq = crear_ventanas_temporales(X_val_scaled, y_val_scaled, ventana)


# Dataset PyTorch
class GRUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(GRUDataset(X_train_seq, y_train_seq), batch_size=128, shuffle=True)
val_loader = DataLoader(GRUDataset(X_val_seq, y_val_seq), batch_size=128, shuffle=False)


# Modelo GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])

model = GRUModel(input_dim=X_train_seq.shape[2]).to("cuda" if torch.cuda.is_available() else "cpu")


# Entrenamiento
def train_gru(model, train_loader, val_loader, scaler_y, epochs=5, lr=0.001):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds, val_targets = [], []
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch).squeeze().cpu().numpy()
                val_preds.append(output)
                val_targets.append(y_batch.numpy())

            y_pred = scaler_y.inverse_transform(np.concatenate(val_preds).reshape(-1, 1)).flatten()
            y_true = scaler_y.inverse_transform(np.concatenate(val_targets).reshape(-1, 1)).flatten()

            print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f} - "
                  f"Val MAE: {mean_absolute_error(y_true, y_pred):.2f} - "
                  f"R2: {r2_score(y_true, y_pred):.4f}")

train_gru(model, train_loader, val_loader, scaler_y, epochs=5)
