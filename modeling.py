import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from math import radians, cos, sin, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def clusterizar_estaciones(df_viajes, n_clusters=8):
    estaciones = df_viajes[["id_estacion_destino", "lat_estacion_destino", "long_estacion_destino"]].drop_duplicates()
    estaciones = estaciones.rename(columns={
        "id_estacion_destino": "id_estacion", 
        "lat_estacion_destino": "lat", 
        "long_estacion_destino": "lon"
    })

    coords = estaciones[["lat", "lon"]].to_numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    estaciones["zona_estacion"] = kmeans.labels_

    plt.figure(figsize=(8, 6))
    for z in range(n_clusters):
        cluster = estaciones[estaciones["zona_estacion"] == z]
        plt.scatter(cluster["lon"], cluster["lat"], label=f"Zona {z}", s=10)
    plt.title("Clustering de estaciones por zona")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True)
    plt.show()

    return dict(zip(estaciones["id_estacion"], estaciones["zona_estacion"]))

def agregar_intervalo_30m(df, col_fecha):
    df = df.copy()
    df["intervalo_30m"] = pd.to_datetime(df[col_fecha]).dt.floor("30min")
    return df

def agregar_features_temporales(df):
    df["dia_semana"] = df["intervalo_30m"].dt.dayofweek
    df["es_finde"] = df["dia_semana"].isin([5, 6])
    df["hora_dia"] = df["intervalo_30m"].dt.hour
    return df

def construir_features_por_estacion_intervalo(df_viajes, df_usuarios, mapping_zona):
    df = df_viajes.copy()
    df = agregar_intervalo_30m(df, "fecha_origen_recorrido")
    df["zona_estacion_origen"] = df["id_estacion_origen"].map(mapping_zona)
    
    df = df.merge(df_usuarios.rename(columns={"ID_usuario": "id_usuario"}), on="id_usuario", how="left")

    group_cols = ["intervalo_30m", "id_estacion_origen"]
    agg = df.groupby(group_cols).agg(
        cantidad_salidas=("id_recorrido", "count"),
        promedio_duracion=("duracion_recorrido", "mean"),
        promedio_edad=("edad_usuario", "mean"),
        porcentaje_mujeres=("genero", lambda x: (x == "FEMALE").mean()),
        porcentaje_hombres=("genero", lambda x: (x == "MALE").mean()),
        porcentaje_otro=("genero", lambda x: ((x != "FEMALE") & (x != "MALE")).mean()),
        modelo_mas_usado=("modelo_bicicleta", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        cantidad_estaciones_destino_distintas=("id_estacion_destino", pd.Series.nunique),
    ).reset_index()

    df_top = df.groupby(group_cols + ["id_estacion_destino"]).size().reset_index(name="frecuencia")
    top1 = df_top.sort_values("frecuencia", ascending=False).groupby(group_cols).first().reset_index()
    top1 = top1.rename(columns={"id_estacion_destino": "top_origen_1"})

    features = pd.merge(agg, top1, on=group_cols, how="left")
    features["zona_estacion"] = features["id_estacion_origen"].map(mapping_zona)
    features = agregar_features_temporales(features)

    features = features.rename(columns={"id_estacion_origen": "id_estacion"})
    return features

def calcular_target_arribos(df_viajes, df_intervals):
    df = df_viajes.copy()
    df = agregar_intervalo_30m(df, "fecha_destino_recorrido")
    df_target = df.groupby(["intervalo_30m", "id_estacion_destino"]).size().reset_index(name="N_arribos_intervalo")
    df_target = df_target.rename(columns={"id_estacion_destino": "id_estacion", "intervalo_30m": "intervalo_30m_futuro"})

    df_intervals["intervalo_30m_futuro"] = df_intervals["intervalo_30m"] + pd.Timedelta(minutes=30)
    df_full = pd.merge(df_intervals, df_target, on=["intervalo_30m_futuro", "id_estacion"], how="left")
    df_full["N_arribos_intervalo"] = df_full["N_arribos_intervalo"].fillna(0).astype(int)
    return df_full.drop(columns=["intervalo_30m_futuro"])

def construir_dataset_modelado(df_viajes, df_usuarios, n_clusters=8):
    zona_mapping = clusterizar_estaciones(df_viajes, n_clusters=n_clusters)
    features = construir_features_por_estacion_intervalo(df_viajes, df_usuarios, zona_mapping)
    dataset = calcular_target_arribos(df_viajes, features)
    return dataset