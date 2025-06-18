import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.gru import MLPModel
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_loader, val_loader, epochs, lr, scaler_y):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Usando dispositivo: {device}")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"\n🔄 Epoch {epoch+1}/{epochs} - Entrenando...")

        for i, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Entrenamiento Epoch {epoch+1}")):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 0:
                print(f"Batch {i}/{len(train_loader)} procesado - Loss actual: {loss.item():.4f} - Dispositivo: {X_batch.device}")

        print(f"📉 Loss de entrenamiento promedio: {running_loss / len(train_loader):.4f}")

        model.eval()
        val_preds_scaled, val_trues_scaled = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).squeeze().cpu().numpy()
                val_preds_scaled.extend(preds)
                val_trues_scaled.extend(y_batch.numpy())

        # Desescalar predicciones y ground truth
        val_preds = scaler_y.inverse_transform(np.array(val_preds_scaled).reshape(-1,1)).flatten()
        val_trues = scaler_y.inverse_transform(np.array(val_trues_scaled).reshape(-1,1)).flatten()

        mae = mean_absolute_error(val_trues, val_preds)
        rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        r2 = r2_score(val_trues, val_preds)

        print(f"✅ Epoch {epoch+1} terminada - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # Gráfica comparativa
    plt.figure(figsize=(10,5))
    plt.plot(val_trues[:100], label="Real", marker='o')
    plt.plot(val_preds[:100], label="Predicho", marker='x')
    baseline = [np.mean(val_trues)] * len(val_trues)
    plt.plot(baseline[:100], label="Baseline (media)", linestyle='--')
    plt.title("Predicciones vs Valores reales")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Primeros 10 valores reales y predichos:")
    for i in range(10):
        print(f"Real: {val_trues[i]:.2f} - Predicho: {val_preds[i]:.2f}")

    print("Varianza de y_val:", np.var(val_trues))
    print("Media de y_val:", np.mean(val_trues))

def train_model_earlystop(model, train_loader, val_loader, epochs, lr, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}")):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Imprimir cada 200 batches (o menos si tienes pocos batches)
            if i % 5000 == 0:
                print(f"Epoch {epoch+1}, Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # Validación
        model.eval()
        val_preds_scaled, val_trues_scaled = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).squeeze().cpu().numpy()
                val_preds_scaled.extend(preds)
                val_trues_scaled.extend(y_batch.numpy())
        val_trues = np.array(val_trues_scaled)
        val_preds = np.array(val_preds_scaled)
        val_loss = mean_squared_error(val_trues, val_preds)

        print(f"Epoch {epoch+1} completo - Train Loss promedio: {avg_train_loss:.4f}, Val MSE: {val_loss:.4f}, R2: {r2_score(val_trues, val_preds):.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# Función para hacer secuencias
def crear_ventanas_temporales(X, y, ventana=6):
    # Crea secuencias temporales de datos para entrenamiento
    secuencias, targets = [], []
    for i in range(len(X) - ventana):
        secuencia = X[i:i+ventana]
        objetivo = y[i+ventana]
        secuencias.append(secuencia)
        targets.append(objetivo)
    return np.array(secuencias), np.array(targets)



#MLP UITLS
def crear_features_con_lags(X, y, ventana=6):
    X_seq, y_seq = [], []
    for i in range(ventana, len(X)):
        X_seq.append(X[i-ventana:i].flatten())
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def train_mlp_model(X_train, y_train, X_val, y_val, input_dim, scaler_y, epochs=10, batch_size=128, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Usando dispositivo: {device}")

    model = MLPModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Convertir a tensores y mover al dispositivo
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_val_tensor).squeeze().cpu().numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"📉 Epoch {epoch+1} - Loss: {running_loss/len(train_loader):.4f} - MAE: {mae:.4f} - R²: {r2:.4f}")

    return model
