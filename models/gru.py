import torch
import torch.nn as nn
from torch.utils.data import Dataset

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h[-1])


class GRUDataset(Dataset):
    #lo que pasa con GRU es que trabaja con secuencias de datos, por lo que necesitamos un Dataset que maneje secuencias
    #entonces en vez de mandarle filas individuales, le mandamos secuencias de datos
    def __init__(self, sequences, targets):
        self.sequences = sequences # sequences es una lista de listas, donde cada lista es una secuencia de datos
        self.targets = targets # targets son las etiquetas o valores que queremos predecir

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

