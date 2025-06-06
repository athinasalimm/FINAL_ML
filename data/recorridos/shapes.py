import pandas as pd
trips2020 = pd.read_csv('data/recorridos/raw/trips_2020.csv')
trips2021 = pd.read_csv('data/recorridos/raw/trips_2021.csv')
trips2022 = pd.read_csv('data/recorridos/raw/trips_2022.csv')
trips2023 = pd.read_csv('data/recorridos/raw/trips_2023.csv')
trips2024 = pd.read_csv('data/recorridos/raw/trips_2024.csv')

print(trips2020.shape)
print(trips2021.shape)
print(trips2022.shape)
print(trips2023.shape)  
print(trips2024.shape)