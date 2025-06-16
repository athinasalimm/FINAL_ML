import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.MSELoss() # Usamos MSELoss para regresión
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # usamos adam ! (dsp optimizamos el lr)

    for epoch in range(epochs): 
        model.train()
        for X_batch, y_batch in train_loader: #baches de entrenamiento
            output = model(X_batch) # fprediccion 
            loss = criterion(output.squeeze(), y_batch) # calcula el error 
            optimizer.zero_grad() # limpia los gradientes acumulados
            loss.backward() # backpropagation
            optimizer.step() # actualiza los pesos del modelo

        model.eval()
        with torch.no_grad(): #no calculamos gradientes en la validación
            val_preds, val_trues = [], []
            for X_batch, y_batch in val_loader:
                preds = model(X_batch).squeeze().detach().numpy()
                val_preds.extend(preds) # extendemos la lista de predicciones
                val_trues.extend(y_batch.numpy())

        mae = mean_absolute_error(val_trues, val_preds)
        rmse = np.sqrt(mean_squared_error(val_trues, val_preds))
        print(f"Epoch {epoch+1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

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
