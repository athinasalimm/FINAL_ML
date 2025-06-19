import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from tqdm import tqdm
import os

def cargar_usuarios_todos_los_anios():
    rutas = []
    for anio in range(2015, 2025):
        path_1 = f"data/usuarios/processed/usuarios_ecobici_{anio}_limpio.csv"
        path_2 = f"data/new_data/processed/usuarios_ecobici_{anio}_limpio.csv"
        if os.path.exists(path_1):
            rutas.append(path_1)
        if os.path.exists(path_2):
            rutas.append(path_2)

    usuarios = []
    for ruta in rutas:
        try:
            df = pd.read_csv(ruta)
            if "ID_usuario" in df.columns:
                df = df.rename(columns={"ID_usuario": "id_usuario"})
            usuarios.append(df)
        except Exception as e:
            print(f"⚠️ Error al leer {ruta}: {e}")

    if usuarios:
        df_todos = pd.concat(usuarios, ignore_index=True)
        df_todos = df_todos.drop_duplicates(subset="id_usuario")
        return df_todos
    else:
        print("❌ No se encontraron archivos de usuarios.")
        return pd.DataFrame(columns=["id_usuario"])

def agregar_intervalo_tiempo(df, col_fecha='fecha_origen_recorrido', freq='30min'):
    df[col_fecha] = pd.to_datetime(df[col_fecha])
    df['fecha_intervalo'] = df[col_fecha].dt.floor(freq)
    df['hora_dia'] = df['fecha_intervalo'].dt.hour
    df['dia_semana'] = df['fecha_intervalo'].dt.weekday
    df['es_finde'] = df['dia_semana'].isin([5, 6]).astype(int)
    return df

def calcular_estacion_del_anio(fecha):
    mes = fecha.month
    dia = fecha.day
    if (mes == 12 and dia >= 21) or mes in [1, 2] or (mes == 3 and dia < 21):
        return 'verano'
    elif (mes == 3 and dia >= 21) or mes in [4, 5] or (mes == 6 and dia < 21):
        return 'otono'
    elif (mes == 6 and dia >= 21) or mes in [7, 8] or (mes == 9 and dia < 21):
        return 'invierno'
    else:
        return 'primavera'

def agregar_estacion_del_anio(df):
    df['estacion_del_anio'] = df['fecha_intervalo'].apply(calcular_estacion_del_anio)
    return df

def clusterizar_barrios(df_estaciones, k=15):
    df_centros = df_estaciones.groupby('barrio')[['lat', 'lon']].mean().reset_index()
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_centros[['lat', 'lon']])
    df_centros['zona_cluster'] = kmeans.labels_
    return df_centros[['barrio', 'zona_cluster']].set_index('barrio').to_dict()['zona_cluster']

def contar_estaciones_cercanas(df_estaciones, radio_m=500):
    estaciones = df_estaciones[['id_estacion', 'lat', 'lon']].to_dict('records')
    cercanas = {}
    for est in tqdm(estaciones, desc="Contando estaciones cercanas"):
        count = 0
        for otra in estaciones:
            if est['id_estacion'] != otra['id_estacion']:
                dist = geodesic((est['lat'], est['lon']), (otra['lat'], otra['lon'])).meters
                if dist <= radio_m:
                    count += 1
        cercanas[est['id_estacion']] = count
    return cercanas

def construir_dataset_modelado_v2(df_viajes, df_usuarios, df_estaciones):
    df = df_viajes.copy()
    df = agregar_intervalo_tiempo(df)
    df = agregar_estacion_del_anio(df)

    df_usuarios_global = cargar_usuarios_todos_los_anios()
    columnas_usuario = [col for col in df_usuarios_global.columns if col != "id_usuario"]
    df = df.merge(df_usuarios_global, how="left", on="id_usuario")
    df["usuario_registrado"] = df["id_usuario"].isin(df_usuarios_global["id_usuario"]).astype(int)

    columnas_str_usuario = [col for col in columnas_usuario if df[col].dtype == "O"]
    columnas_num_usuario = [col for col in columnas_usuario if np.issubdtype(df[col].dtype, np.number)]
    for col in columnas_str_usuario:
        df[col] = df[col].fillna("desconocido")
    for col in columnas_num_usuario:
        df[col] = df[col].fillna(0)

    # Clustering y estaciones cercanas para ORIGEN y DESTINO
    dict_cluster = clusterizar_barrios(df_estaciones)
    df['zona_destino_cluster'] = df['barrio_destino'].map(dict_cluster)
    df['zona_origen_cluster'] = df['barrio_origen'].map(dict_cluster)

    estaciones_cercanas = contar_estaciones_cercanas(df_estaciones)
    df['cantidad_estaciones_cercanas_destino'] = df['id_estacion_destino'].map(estaciones_cercanas)
    df['cantidad_estaciones_cercanas_origen'] = df['id_estacion_origen'].map(estaciones_cercanas)

    agg = df.groupby(['id_estacion_destino', 'fecha_intervalo']).size().reset_index(name='N_arribos_intervalo')
    df = df.merge(agg, on=['id_estacion_destino', 'fecha_intervalo'], how='left')
    agg2 = df.groupby(['id_estacion_origen', 'fecha_intervalo']).size().reset_index(name='N_salidas_intervalo')
    df = df.merge(agg2, on=['id_estacion_origen', 'fecha_intervalo'], how='left')

    # Columnas innecesarias
    columnas_a_eliminar = [
        'lat_estacion_destino', 'lat_estacion_origen',
        'long_estacion_destino', 'long_estacion_origen',
        'nombre_estacion_destino', 'nombre_estacion_origen',
        'direccion_estacion_destino', 'direccion_estacion_origen',
        'genero', 'genero_other', 'hora_alta_h', 'hora_alta_m', 'hora_alta_s', 'dia_alta'
    ]
    df.drop(columns=[col for col in columnas_a_eliminar if col in df.columns], inplace=True)

    return df