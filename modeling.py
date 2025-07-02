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

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json
import os

def clusterizar_barrios(df_estaciones, k=15, output_json="data/modelado/mapa_cluster_barrios.json", save_plot="data/modelado/cluster_barrios.png"):
    df_centros = df_estaciones.groupby('barrio')[['lat', 'lon']].mean().reset_index()

    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_centros[['lat', 'lon']])
    df_centros['zona_cluster'] = kmeans.labels_

    # 🔍 Mostrar y guardar gráfico
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_centros["lon"], df_centros["lat"], c=df_centros["zona_cluster"], cmap="tab20", s=100, edgecolor="black")
    for _, row in df_centros.iterrows():
        plt.text(row["lon"], row["lat"], row["barrio"], fontsize=8)
    plt.title("Clustering de Barrios (entrenamiento)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(True)

    os.makedirs(os.path.dirname(save_plot), exist_ok=True)
    plt.savefig(save_plot, bbox_inches="tight")
    plt.show()

    # 💾 Guardar JSON
    dict_cluster = df_centros.set_index("barrio")["zona_cluster"].to_dict()
    with open(output_json, "w") as f:
        json.dump(dict_cluster, f, ensure_ascii=False)
    print(f"✅ Clustering guardado en {output_json} y gráfico en {save_plot}")

    return dict_cluster

import json

def leer_clusters(path_json="data/modelado/mapa_cluster_barrios.json"):
    with open(path_json, "r") as f:
        dict_cluster = json.load(f)
    return dict_cluster

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

def construir_dataset_modelado_v2(df_viajes, df_usuarios, df_estaciones, test = False, path_json_cluster="data/mapa_cluster_barrios.json"):
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

    if test:
        dict_cluster = leer_clusters(path_json_cluster)
    else:
        dict_cluster = clusterizar_barrios(df_estaciones, output_json=path_json_cluster)

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


import pandas as pd
import os

def insumos_modelado(test=False, path_test=None):
    """
    Crea el dataset modelado, funcionando para modo completo (2020–2024) o test individual.
    """
    # === CARGA DE VIAJES
    if test:
        if not path_test or not os.path.exists(path_test):
            raise FileNotFoundError("❌ Path de test inválido.")
        print(f"🧪 Modo test: usando archivo {path_test}")
        df_viajes = pd.read_csv(path_test)
    else:
        print("📥 Cargando viajes completos 2020–2024...")
        recorridos_dir = "data/recorridos/processed"
        df_viajes = pd.concat([
            pd.read_csv(os.path.join(recorridos_dir, f))
            for f in sorted(os.listdir(recorridos_dir))
            if f.endswith(".csv")
        ], ignore_index=True)

    # === CARGA DE USUARIOS (siempre todos)
    print("👤 Cargando todos los usuarios (2015–2024)...")
    usuarios_dir_1 = "data/usuarios/processed"
    usuarios_dir_2 = "data/new_data/processed"
    archivos_usuarios = []

    for carpeta in [usuarios_dir_1, usuarios_dir_2]:
        archivos_usuarios += [
            os.path.join(carpeta, f)
            for f in sorted(os.listdir(carpeta))
            if f.endswith(".csv")
        ]

    df_usuarios = pd.concat([
        pd.read_csv(f) for f in archivos_usuarios
    ], ignore_index=True)

    # === CARGA DE ESTACIONES (con barrios ya asignados)
    df_estaciones = pd.read_csv("data/estaciones_con_barrios.csv")

    # === LLAMADA A LA FUNCIÓN PRINCIPAL
    from modeling import construir_dataset_modelado_v2
    df_modelado = construir_dataset_modelado_v2(df_viajes, df_usuarios, df_estaciones, test=test)

    return df_modelado

def procesar_columnas_modelado(df_modelado):
    # === Estación del año → int
    mapa_estaciones = {"verano": 1, "otono": 2, "invierno": 3, "primavera": 4}
    df_modelado["estacion_del_anio"] = df_modelado["estacion_del_anio"].map(mapa_estaciones).astype(int)

    # === Conversiones de fecha
    df_modelado["fecha_origen_recorrido"] = pd.to_datetime(df_modelado["fecha_origen_recorrido"], errors="coerce")
    df_modelado["fecha_destino_recorrido"] = pd.to_datetime(df_modelado["fecha_destino_recorrido"], errors="coerce")
    df_modelado["fecha_intervalo"] = pd.to_datetime(df_modelado["fecha_intervalo"], errors="coerce")

    # === Columnas temporales origen
    df_modelado["año_origen"] = df_modelado["fecha_origen_recorrido"].dt.year
    df_modelado["mes_origen"] = df_modelado["fecha_origen_recorrido"].dt.month
    df_modelado["dia_origen"] = df_modelado["fecha_origen_recorrido"].dt.day
    df_modelado["hora_origen"] = df_modelado["fecha_origen_recorrido"].dt.hour
    df_modelado["minuto_origen"] = df_modelado["fecha_origen_recorrido"].dt.minute
    df_modelado["segundo_origen"] = df_modelado["fecha_origen_recorrido"].dt.second

    # === Columnas temporales destino
    df_modelado["año_destino"] = df_modelado["fecha_destino_recorrido"].dt.year
    df_modelado["mes_destino"] = df_modelado["fecha_destino_recorrido"].dt.month
    df_modelado["dia_destino"] = df_modelado["fecha_destino_recorrido"].dt.day
    df_modelado["hora_destino"] = df_modelado["fecha_destino_recorrido"].dt.hour
    df_modelado["minuto_destino"] = df_modelado["fecha_destino_recorrido"].dt.minute
    df_modelado["segundo_destino"] = df_modelado["fecha_destino_recorrido"].dt.second

    # === Columnas temporales intervalo
    df_modelado["año_intervalo"] = df_modelado["fecha_intervalo"].dt.year
    df_modelado["mes_intervalo"] = df_modelado["fecha_intervalo"].dt.month
    df_modelado["dia_intervalo"] = df_modelado["fecha_intervalo"].dt.day
    df_modelado["hora_intervalo"] = df_modelado["fecha_intervalo"].dt.hour
    df_modelado["minuto_intervalo"] = df_modelado["fecha_intervalo"].dt.minute

    # === Edad
    df_modelado["edad_usuario"] = pd.to_numeric(df_modelado["edad_usuario"], errors="coerce").fillna(-1).astype(int)

    # === Mapeo de barrios únicos
    barrios_unicos = pd.unique(df_modelado[["barrio_origen", "barrio_destino"]].values.ravel())
    mapa_barrio = {barrio: i for i, barrio in enumerate(barrios_unicos, start=1)}

    # Guardar el mapa
    with open("data/modelado/mapa_barrio.json", "w") as f:
        json.dump(mapa_barrio, f, ensure_ascii=False)

    df_modelado["barrio_origen"] = df_modelado["barrio_origen"].map(mapa_barrio)
    df_modelado["barrio_destino"] = df_modelado["barrio_destino"].map(mapa_barrio)

    # === Modelo de bicicleta
    df_modelado["modelo_bicicleta"] = df_modelado["modelo_bicicleta"].map({"ICONIC": 1, "FIT": 0}).astype(int)

    # === Eliminar columnas no necesarias
    df_modelado.drop(columns=[
        "hora_dia",
        "fecha_origen_recorrido",
        "fecha_destino_recorrido",
        "fecha_intervalo"
    ], errors="ignore", inplace=True)

    return df_modelado

def generar_atributos_estaciones(
    path_estaciones: str = "data/new_data/raw/nuevas-estaciones-bicicletas-publicas.csv",
    path_ciclovias: str = "data/new_data/raw/ciclovias.geojson",
    output_path: str = "data/new_data/processed/atributos_estaciones.csv"
) -> None:
    import geopandas as gpd
    import pandas as pd
    import os
    from shapely.geometry import Point, LineString, MultiLineString

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar estaciones y ciclovías
    df_est = pd.read_csv(path_estaciones)
    gdf_ciclovias = gpd.read_file(path_ciclovias)

    # Convertir estaciones a GeoDataFrame
    gdf_est = gpd.GeoDataFrame(
        df_est,
        geometry=gpd.points_from_xy(df_est["longitud"], df_est["latitud"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Ciclovías a misma proyección
    gdf_ciclovias = gdf_ciclovias.to_crs(epsg=3857)
    ciclovias_list = list(gdf_ciclovias.geometry)
    ciclovias_union = gdf_ciclovias.unary_union

    # Distancia a la ciclovía más cercana
    gdf_est["dist_ciclovia_m"] = gdf_est.geometry.apply(lambda p: p.distance(ciclovias_union))

    # Longitud de ciclovías en un radio de 200m
    longitudes = []
    for est_geom in gdf_est.geometry:
        buffer = est_geom.buffer(200)
        total_length = 0.0
        for ciclovia in ciclovias_list:
            try:
                inter = ciclovia.intersection(buffer)
                if inter.is_empty:
                    continue
                if isinstance(inter, (LineString, MultiLineString)):
                    total_length += inter.length
            except Exception:
                continue
        longitudes.append(total_length)
    gdf_est["ciclo_len_200m"] = longitudes

    # Dejar solo columnas útiles
    df_out = gdf_est[[
        "id", "dist_ciclovia_m", "ciclo_len_200m"
    ]].rename(columns={"id": "id_estacion_origen"})

    df_out.to_csv(output_path, index=False)
    print(f"✅ Atributos de estaciones guardados en: {output_path}")

import pandas as pd
import json
import os

def final_prep(
    path_modelado: str,
    path_output: str = "data/modelado/ds_modelado_FUNCIONA.csv",
    path_atributos_estacion: str = "data/new_data/processed/atributos_estaciones.csv",
    path_barrios: str = "data/estaciones_con_barrios.csv",
    path_mapa_barrio: str = "data/modelado/mapa_barrio.json"
) -> None:
    df = pd.read_csv(path_modelado)

    df["fecha_intervalo"] = pd.to_datetime(dict(
        year=df["año_intervalo"],
        month=df["mes_intervalo"],
        day=df["dia_intervalo"],
        hour=df["hora_intervalo"],
        minute=df["minuto_intervalo"]
    ))

    fecha_min = df["fecha_intervalo"].min()
    fecha_max = df["fecha_intervalo"].max()
    intervalos = pd.date_range(start=fecha_min, end=fecha_max, freq="30min")
    estaciones = df["id_estacion_origen"].unique()

    df_regular = pd.MultiIndex.from_product(
        [estaciones, intervalos],
        names=["id_estacion_origen", "fecha_intervalo"]
    ).to_frame(index=False)

    df_lags = df[[
        "id_estacion_origen", "fecha_intervalo",
        "N_arribos_intervalo", "N_salidas_intervalo"
    ]]
    df_regular = df_regular.merge(df_lags, on=["id_estacion_origen", "fecha_intervalo"], how="left")
    df_regular[["N_arribos_intervalo", "N_salidas_intervalo"]] = df_regular[["N_arribos_intervalo", "N_salidas_intervalo"]].fillna(0)

    for lag in [1, 2, 3]:
        df_regular[f"arribos_lag{lag}"] = df_regular.groupby("id_estacion_origen")["N_arribos_intervalo"].shift(lag)
    for lag in [1, 2]:
        df_regular[f"salidas_lag{lag}"] = df_regular.groupby("id_estacion_origen")["N_salidas_intervalo"].shift(lag)

    df_regular["arribos_rolling7"] = (
        df_regular.groupby("id_estacion_origen")["N_arribos_intervalo"]
            .transform(lambda x: x.shift(1).rolling(window=7).mean())
    )
    df_regular["arribos_ultima_hora"] = (
        df_regular.groupby("id_estacion_origen")["N_arribos_intervalo"]
            .transform(lambda x: x.shift(1).rolling(window=2).sum())
    )

    lag_cols = [col for col in df_regular.columns if "lag" in col or "rolling" in col]
    df_regular[lag_cols] = df_regular[lag_cols].fillna(-1)

    df = df.merge(
        df_regular[["id_estacion_origen", "fecha_intervalo"] + lag_cols],
        on=["id_estacion_origen", "fecha_intervalo"],
        how="left"
    )

    df["fecha_origen"] = pd.to_datetime(dict(
        year=df["año_origen"],
        month=df["mes_origen"],
        day=df["dia_origen"],
        hour=df["hora_origen"],
        minute=df["minuto_origen"]
    ))
    df = df.sort_values(["id_usuario", "fecha_origen"]).copy()

    df["ultima_estacion_origen_usuario"] = df.groupby("id_usuario")["id_estacion_origen"].shift(1).fillna(-1).astype(int)
    df["ultima_estacion_destino_usuario"] = df.groupby("id_usuario")["id_estacion_destino"].shift(1).fillna(-1).astype(int)
    df["fecha_ultimo_viaje_usuario"] = df.groupby("id_usuario")["fecha_origen"].shift(1)
    df["tiempo_desde_ultimo_viaje_usuario"] = (
        (df["fecha_origen"] - df["fecha_ultimo_viaje_usuario"]).dt.total_seconds() / 60
    ).fillna(-1)

    df_barrios = pd.read_csv(path_barrios)
    with open(path_mapa_barrio, "r") as f:
        mapa_barrio = json.load(f)

    df = df.merge(
        df_barrios.rename(columns={
            "id_estacion": "ultima_estacion_origen_usuario",
            "barrio": "barrio_ultima_estacion_origen_usuario"
        }),
        on="ultima_estacion_origen_usuario", how="left"
    )
    df = df.merge(
        df_barrios.rename(columns={
            "id_estacion": "ultima_estacion_destino_usuario",
            "barrio": "barrio_ultima_estacion_destino_usuario"
        }),
        on="ultima_estacion_destino_usuario", how="left"
    )

    df["barrio_ultima_estacion_origen_usuario"] = df["barrio_ultima_estacion_origen_usuario"].map(mapa_barrio).fillna(-1).astype(int)
    df["barrio_ultima_estacion_destino_usuario"] = df["barrio_ultima_estacion_destino_usuario"].map(mapa_barrio).fillna(-1).astype(int)

    for lag in [1, 2, 3]:
        df[f"estacion_origen_viejo_us_lag{lag}"] = df.groupby("id_usuario")["id_estacion_origen"].shift(lag).fillna(-1).astype(int)
        df[f"estacion_destino_viejo_us_lag{lag}"] = df.groupby("id_usuario")["id_estacion_destino"].shift(lag).fillna(-1).astype(int)

    df["estacion_origen_viejo_us"] = df["estacion_origen_viejo_us_lag1"]
    df["estacion_destino_viejo_us"] = df["estacion_destino_viejo_us_lag1"]

    df_atributos = pd.read_csv(path_atributos_estacion)
    df = df.merge(df_atributos, on="id_estacion_origen", how="left")

    df.to_csv(path_output, index=False)
    print(f"✅ Dataset final guardado en {path_output}")

import pandas as pd
import os

def limpiar_y_guardar_parquet(
    path_csv: str,
    path_parquet: str
) -> None:
    """
    Limpia columnas innecesarias y guarda el dataset en formato Parquet.
    
    Parámetros:
    - path_csv: ruta al CSV de entrada
    - path_parquet: ruta de salida en formato .parquet
    """
    if not os.path.exists(path_csv):
        print(f"❌ No se encontró el archivo: {path_csv}")
        return

    df = pd.read_csv(path_csv)

    # Eliminar columnas innecesarias si existen
    df = df.drop(columns=[
        "lat_x", "lon_x", "lat_y", "lon_y", "fecha_ultimo_viaje_usuario"
    ], errors="ignore")

    # Rellenar columnas de ciclovías
    df["ciclo_len_200m"] = df["ciclo_len_200m"].fillna(0.0)
    df["dist_ciclovia_m"] = df["dist_ciclovia_m"].fillna(9999.0)

    # Guardar
    df.to_parquet(path_parquet, index=False)
    print(f"✅ Parquet guardado en: {path_parquet}")


def armar_dataset_modelado(test=False, path_test=None):
    if test:
        if not path_test or not os.path.exists(path_test):
            print("❌ Path de test inválido o no existe.")
            return

        print("\n🧪 Generando dataset de TEST...")
        df_test = insumos_modelado(test=True, path_test=path_test)
        df_modelado_test = procesar_columnas_modelado(df_test)

        path_csv = "data/test/ds_modelado_test.csv"
        path_csv_final = "data/test/ds_modelado_test_final.csv"
        path_parquet = "data/test/ds_modelado_test.parquet"
    
        df_modelado_test.to_csv(path_csv, index=False)

        final_prep(
            path_modelado=path_csv,
            path_output=path_csv_final
        )

        limpiar_y_guardar_parquet(
            path_csv=path_csv_final,
            path_parquet=path_parquet
        )
        print("✅ Dataset de test generado correctamente.")

    else:
        print("\n🧱 Generando dataset de TRAIN (años completos)...")
        df_train = insumos_modelado()
        df_modelado_train = procesar_columnas_modelado(df_train)

        path_csv = "data/modelado/ds_modelado.csv"
        path_csv_final = "data/modelado/ds_modelado_FUNCIONA.csv"
        path_parquet = "data/modelado/ds_modelado_FUNCIONA.parquet"

        df_modelado_train.to_csv(path_csv, index=False)

        final_prep(
            path_modelado=path_csv,
            path_output=path_csv_final
        )

        limpiar_y_guardar_parquet(
            path_csv=path_csv_final,
            path_parquet=path_parquet
        )
        print("✅ Dataset de entrenamiento generado correctamente.")