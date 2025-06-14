import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from geopy.distance import geodesic
from tqdm import tqdm

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

def clusterizar_barrios(df_estaciones, k=8):
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

import pandas as pd
from tqdm import tqdm

def construir_dataset_modelado_v2(df_viajes, df_usuarios, df_estaciones):
    df = df_viajes.copy()

    # Agregar columnas temporales
    df = agregar_intervalo_tiempo(df)
    df = agregar_estacion_del_anio(df)

    # Merge con usuarios (pueden ser de años anteriores)
    df_usuarios = df_usuarios.drop_duplicates(subset="id_usuario")
    df = df.merge(df_usuarios, how="left", on="id_usuario")

    # Reportar usuarios no encontrados
    df["usuario_encontrado"] = df["ID_usuario"].notna()
    no_encontrados = df[~df["usuario_encontrado"]]
    conteo_no_encontrados = no_encontrados.groupby(df["fecha_origen_recorrido"].str[:4]).size()
    print("📌 Usuarios sin datos por año de viaje:")
    print(conteo_no_encontrados)

    # Clustering de barrios
    dict_cluster = clusterizar_barrios(df_estaciones)
    df['zona_destino_cluster'] = df['barrio_destino'].map(dict_cluster)
    df['zona_origen_cluster'] = df['barrio_origen'].map(dict_cluster)

    # Estaciones cercanas
    estaciones_cercanas = contar_estaciones_cercanas(df_estaciones)
    df['cantidad_estaciones_cercanas_destino'] = df['id_estacion_destino'].map(estaciones_cercanas)
    df['cantidad_estaciones_cercanas_origen'] = df['id_estacion_origen'].map(estaciones_cercanas)

    # Target de arribos y salidas por intervalo
    agg = df.groupby(['id_estacion_destino', 'fecha_intervalo']).size().reset_index(name='N_arribos_intervalo')
    df = df.merge(agg, on=['id_estacion_destino', 'fecha_intervalo'], how='left')

    agg2 = df.groupby(['id_estacion_origen', 'fecha_intervalo']).size().reset_index(name='N_salidas_intervalo')
    df = df.merge(agg2, on=['id_estacion_origen', 'fecha_intervalo'], how='left')

    # Marcar tipo de movimiento
    df_llegadas = df.copy()
    df_llegadas['tipo_movimiento'] = 'llegada'

    df_salidas = df.copy()
    df_salidas['tipo_movimiento'] = 'salida'
    df_salidas = df_salidas.rename(columns={
        'id_estacion_origen': 'id_estacion_destino',
        'barrio_origen': 'barrio_destino',
        'zona_origen_cluster': 'zona_destino_cluster',
        'cantidad_estaciones_cercanas_origen': 'cantidad_estaciones_cercanas_destino'
    })

    # Eliminar columnas viejas de salida
    columnas_a_eliminar = ['id_estacion_origen', 'barrio_origen', 'zona_origen_cluster', 'cantidad_estaciones_cercanas_origen']
    df_salidas.drop(columns=[col for col in columnas_a_eliminar if col in df_salidas.columns], inplace=True)

    # Igualar columnas para concatenar
    columnas_comunes = sorted(set(df_llegadas.columns) & set(df_salidas.columns))
    df_llegadas = df_llegadas.reindex(columns=columnas_comunes)
    df_salidas = df_salidas.loc[:, ~df_salidas.columns.duplicated()]
    df_salidas = df_salidas.reindex(columns=columnas_comunes)

    # Concatenar filas
    df_final = pd.concat([df_llegadas, df_salidas], ignore_index=True)

    # Completar id_usuario si hace falta
    if 'ID_usuario' in df_final.columns and 'id_usuario' in df_final.columns:
        df_final['id_usuario'] = df_final['id_usuario'].fillna(df_final['ID_usuario'])

    df_final.drop(columns=['usuario_encontrado'], errors='ignore', inplace=True)
    return df_final