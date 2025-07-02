import pandas as pd
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os


def filtrar_meses(dataset, columna_fecha, meses_a_excluir):
    dataset[columna_fecha] = pd.to_datetime(dataset[columna_fecha], errors='coerce')
    filas_originales = len(dataset)
    df_filtrado = dataset[~dataset[columna_fecha].dt.month.isin(meses_a_excluir)].copy()
    eliminadas = filas_originales - len(df_filtrado)
    return df_filtrado, eliminadas

def verificar_columnas_y_tipos(archivos, archivo_referencia=None):
    info_archivos = {}
    for archivo in archivos:
        df = pd.read_csv(archivo)
        columnas = df.columns.tolist()
        tipos = df.dtypes.to_dict()
        info_archivos[archivo] = {
            'columnas': columnas,
            'tipos': tipos
        }
    if archivo_referencia is None:
        archivo_referencia = archivos[-1]
    columnas_ref = info_archivos[archivo_referencia]['columnas']
    tipos_ref = info_archivos[archivo_referencia]['tipos']
    print(f"\nUsando '{archivo_referencia}' como referencia.\n")
    for archivo in archivos:
        print(f"Verificando: {archivo}")
        columnas_actual = info_archivos[archivo]['columnas']
        if columnas_actual != columnas_ref:
            print("Columnas diferentes.")
            print("Diferencias:", set(columnas_ref).symmetric_difference(columnas_actual))
        else:
            print("Columnas iguales.")

        tipos_actual = info_archivos[archivo]['tipos']
        diferencias_tipos = {
            col: (tipos_actual[col], tipos_ref[col])
            for col in columnas_ref
            if col in tipos_actual and tipos_actual[col] != tipos_ref[col]
        }
        if diferencias_tipos:
            print("Tipos diferentes en algunas columnas:")
            for col, (tipo_act, tipo_ref) in diferencias_tipos.items():
                print(f" - {col}: {tipo_act} (vs {tipo_ref})")
        else:
            print("Tipos de datos iguales.")
        print("-" * 40)

def estandarizar_nombres_columnas(archivos):
    for archivo in archivos:
        df = pd.read_csv(archivo)
        columnas_renombradas = {}
        for col in df.columns:
            if col.lower() == 'género':
                columnas_renombradas[col] = 'genero'
            elif col == 'Id_recorrido':
                columnas_renombradas[col] = 'id_recorrido'

        if columnas_renombradas:
            df = df.rename(columns=columnas_renombradas)
            df.to_csv(archivo, index=False)
        else:
            print(f"No se encontraron columnas para renombrar en {archivo}.")

def limpieza_202x(df):
    df = df.copy()
    for col in ["id_recorrido", "id_estacion_origen", "id_estacion_destino", "id_usuario"]:
        df[col] = df[col].astype(str).str.replace("BAEcobici", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["duracion_recorrido"] = df["duracion_recorrido"].astype(str).str.replace(",", "", regex=False)
    df["duracion_recorrido"] = pd.to_numeric(df["duracion_recorrido"], errors="coerce")
    for col in ["lat_estacion_destino", "long_estacion_destino"]:
        df[col] = df[col].astype(str).str.extract(r"(-?\d+\.\d+)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "id_estacion_destino" in df.columns:
        df["id_estacion_destino"] = pd.to_numeric(df["id_estacion_destino"], errors="coerce").astype("Int64")
    if "id_usuario" in df.columns:
        df["id_usuario"] = pd.to_numeric(df["id_usuario"], errors="coerce")  # queda como float64
    return df


def limpieza_2023(df):
    df = df.copy()
    for col in ["id_recorrido", "id_estacion_origen", "id_estacion_destino", "id_usuario"]:
        df[col] = df[col].astype(str).str.replace("BAEcobici", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["duracion_recorrido"] = df["duracion_recorrido"].astype(str).str.replace(",", "", regex=False)
    df["duracion_recorrido"] = pd.to_numeric(df["duracion_recorrido"], errors="coerce")
    for col in ["lat_estacion_origen", "long_estacion_origen", "lat_estacion_destino", "long_estacion_destino"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "id_estacion_destino" in df.columns:
        df["id_estacion_destino"] = pd.to_numeric(df["id_estacion_destino"], errors="coerce").astype("Int64")
    if "id_usuario" in df.columns:
        df["id_usuario"] = pd.to_numeric(df["id_usuario"], errors="coerce")  # float64
    return df


funciones = {
    "2020": limpieza_202x,
    "2021": limpieza_202x,
    "2022": limpieza_202x,
    "2023": limpieza_2023,
}



def verificar_consistencia_coordenadas(path_csv: str, anio: int = None, test: bool = False):
    if not test:
        print(f"\nVerificando consistencia de coordenadas en {anio or path_csv}...")
    else:
        print(f"\nVerificando consistencia en TEST")

    try:
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f" No se encontró el archivo: {path_csv}")
        return

    coordenadas_por_estacion = {}

    for _, row in df.iterrows():
        for rol in ['origen', 'destino']:
            id_col = f"id_estacion_{rol}"
            lat_col = f"lat_estacion_{rol}"
            lon_col = f"long_estacion_{rol}"

            est_id = row[id_col]
            lat = row[lat_col]
            lon = row[lon_col]

            if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                if est_id not in coordenadas_por_estacion:
                    coordenadas_por_estacion[est_id] = {"lat": set(), "lon": set()}
                coordenadas_por_estacion[est_id]["lat"].add(lat)
                coordenadas_por_estacion[est_id]["lon"].add(lon)

    hay_errores = False
    for est_id, coords in sorted(coordenadas_por_estacion.items()):
        n_lat = len(coords["lat"])
        n_lon = len(coords["lon"])
        if n_lat > 1 or n_lon > 1:
            hay_errores = True
            if n_lat > 1:
                print(f" - Estación {est_id}: difiere en lat ({n_lat} latitudes distintas)")
            if n_lon > 1:
                print(f" - Estación {est_id}: difiere en long ({n_lon} longitudes distintas)")

    if not hay_errores:
        print("Todas las estaciones tienen coordenadas consistentes.")



def corregir_longitudes_inconsistentes(path_csv: str):
    try:
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f" Archivo no encontrado: {path_csv}")
        return

    estaciones_con_errores = set()
    longitudes_por_estacion = {}

    for _, row in df.iterrows():
        for rol in ["origen", "destino"]:
            est_id = row[f"id_estacion_{rol}"]
            lon = row[f"long_estacion_{rol}"]
            if pd.notna(est_id) and pd.notna(lon):
                if est_id not in longitudes_por_estacion:
                    longitudes_por_estacion[est_id] = set()
                longitudes_por_estacion[est_id].add(lon)

    for est_id, longs in longitudes_por_estacion.items():
        if len(longs) > 1:
            estaciones_con_errores.add(est_id)

    if not estaciones_con_errores:
        print(f" No se encontraron inconsistencias de longitud en {path_csv}")
        return

    longitudes_correctas = {}
    for est_id in estaciones_con_errores:
        longitudes = df.loc[df["id_estacion_origen"] == est_id, "long_estacion_origen"].dropna()
        if not longitudes.empty:
            longitudes_correctas[est_id] = longitudes.mode().iloc[0]
        else:
            print(f"Estación {est_id} no aparece como origen en {path_csv}, no se corrige.")

    def corregir_longitud(row):
        for rol in ["origen", "destino"]:
            est_id = row[f"id_estacion_{rol}"]
            lon_col = f"long_estacion_{rol}"
            if est_id in longitudes_correctas:
                long_buena = longitudes_correctas[est_id]
                if pd.notna(row[lon_col]) and row[lon_col] != long_buena:
                    row[lon_col] = long_buena
        return row

    df = df.apply(corregir_longitud, axis=1)
    df.to_csv(path_csv, index=False)
    print(f"Longitudes corregidas automáticamente en: {path_csv}")


def corregir_latitud_estacion(df: pd.DataFrame, est_id: int = 240) -> pd.DataFrame:
    """
    Corrige la latitud de una estación específica (por defecto, la 240)
    usando como referencia la latitud más frecuente cuando aparece como origen.

    Aplica la corrección tanto en origen como destino si la latitud es distinta.
    """
    latitudes = df.loc[df["id_estacion_origen"] == est_id, "lat_estacion_origen"].dropna()
    
    if latitudes.empty:
        print(f" No se encontró la estación {est_id} como origen.")
        return df

    lat_correcta = latitudes.mode().iloc[0]
    print(f"Latitud de referencia para estación {est_id}: {lat_correcta}")

    def corregir_lat(row):
        for rol in ["origen", "destino"]:
            if row[f"id_estacion_{rol}"] == est_id:
                lat_col = f"lat_estacion_{rol}"
                if pd.notna(row[lat_col]) and row[lat_col] != lat_correcta:
                    row[lat_col] = lat_correcta
        return row

    df = df.apply(corregir_lat, axis=1)
    return df

def procesar_recorridos_anio(ruta_csv_entrada, ruta_csv_salida, meses_a_eliminar, col_origen="fecha_origen_recorrido", col_destino="fecha_destino_recorrido"):

    df = pd.read_csv(ruta_csv_entrada)

    df_filtrado, filas_eliminadas_origen = filtrar_meses(df, col_origen, meses_a_eliminar)
    print(f"Se eliminaron {filas_eliminadas_origen} filas por fecha de origen.")

    df_filtrado, filas_eliminadas_destino = filtrar_meses(df_filtrado, col_destino, meses_a_eliminar)
    print(f"Se eliminaron {filas_eliminadas_destino} filas por fecha de destino.")

    total_eliminadas = filas_eliminadas_origen + filas_eliminadas_destino

    df_filtrado.to_csv(ruta_csv_salida, index=False)

    return df_filtrado, total_eliminadas

def limpiar_y_guardar_recorridos():
    años_columnas_a_dropear = {
        2020: [],
        2021: ["Género"],
        2022: ["X"],
        2023: []
    }

    for año, columnas in años_columnas_a_dropear.items():
        ruta_entrada = f"data/recorridos/raw/trips_{año}.csv"
        ruta_salida = f"data/recorridos/processed/trips_{año}_pr.csv"
        
        df = pd.read_csv(ruta_entrada, index_col=0)
        
        if columnas:
            df = df.drop(columns=[col for col in columnas if col in df.columns])
        
        df = df.reset_index(drop=True)
        df.to_csv(ruta_salida, index=False)


def forzar_tipos(df, columnas, tipo):
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].astype(tipo)
    return df

def limpiar_y_normalizar_archivos(archivos, funciones):
    anios = [2020, 2021, 2022, 2023]

    estandarizar_nombres_columnas(archivos)

    for anio in [2020, 2023]:
        ruta = f"data/recorridos/processed/trips_{anio}_pr.csv"
        df = pd.read_csv(ruta)
        df = df.dropna()
        df.to_csv(ruta, index=False)

    for anio in anios:
        print(f"Procesando año {anio}...")
        ruta = f"data/recorridos/processed/trips_{anio}_pr.csv"
        df = pd.read_csv(ruta)
        df_limpio = funciones[str(anio)](df)
        df_limpio = forzar_tipos(df_limpio, ["id_usuario"], 'float64')
        df_limpio.to_csv(ruta, index=False)

    verificar_columnas_y_tipos(archivos)

def corregir_coordenadas_con_base_2024(path_2024, archivos_a_corregir):
    df_2024 = pd.read_csv(path_2024)
    coords_2024 = {}

    for rol in ["origen", "destino"]:
        id_col = f"id_estacion_{rol}"
        lat_col = f"lat_estacion_{rol}"
        lon_col = f"long_estacion_{rol}"

        for _, row in df_2024.iterrows():
            est_id = row[id_col]
            lat = row[lat_col]
            lon = row[lon_col]
            if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                coords_2024[est_id] = (lat, lon)

    for anio, path in archivos_a_corregir.items():
        if not os.path.exists(path):
            print(f"No se encontró el archivo de {anio}")
            continue

        df = pd.read_csv(path)

        for rol in ["origen", "destino"]:
            id_col = f"id_estacion_{rol}"
            lat_col = f"lat_estacion_{rol}"
            lon_col = f"long_estacion_{rol}"

            def corregir_coord(row):
                est_id = row[id_col]
                if est_id in coords_2024:
                    lat, lon = coords_2024[est_id]
                    row[lat_col] = lat
                    row[lon_col] = lon
                return row

            df = df.apply(corregir_coord, axis=1)

        df.to_csv(path, index=False)
        print(f"Archivo actualizado y sobrescrito: {path}")

def corregir_estaciones_2020():
    path_2020 = "data/recorridos/processed/trips_2020_pr.csv"
    df_2020 = pd.read_csv(path_2020)
    df_2020 = corregir_latitud_estacion(df_2020, est_id=240)
    df_2020.to_csv(path_2020, index=False)
    corregir_longitudes_inconsistentes(path_2020)

def verificar_coordenadas_todos_los_anios(anios = [], test: bool = False, path_csv: str = None):
    if not test: 
        for anio in anios:
            path = f"data/recorridos/processed/trips_{anio}_pr.csv"
            verificar_consistencia_coordenadas(path, anio=anio)
    if test: 
        verificar_consistencia_coordenadas(path_csv, test=True)


def construir_diccionario_estaciones(paths_descendentes):
    estaciones_dict = {}

    for path in paths_descendentes:
        if not os.path.exists(path):
            print(f"Archivo no encontrado: {path}")
            continue

        df = pd.read_csv(path)

        for rol in ["origen", "destino"]:
            id_col = f"id_estacion_{rol}"
            lat_col = f"lat_estacion_{rol}"
            lon_col = f"long_estacion_{rol}"

            for _, row in df.iterrows():
                est_id = row[id_col]
                lat = row[lat_col]
                lon = row[lon_col]

                if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                    if est_id not in estaciones_dict:
                        estaciones_dict[est_id] = (lat, lon)

    return estaciones_dict


def asignar_barrios_a_datasets(df_estaciones, anios=None, test=False, path_test=None, path_shapefile="data/barrios/barrios.shp"):
    """
    Enriquece datasets de viajes agregando los barrios de origen y destino según las estaciones.

    Parámetros:
    - df_estaciones: DataFrame con columnas ['id_estacion', 'lat', 'lon']
    - anios: lista de años a procesar (si test=False)
    - test: si es True, se aplica solo al archivo en path_test
    - path_test: path a un CSV de viajes para modo test
    - path_shapefile: ruta al archivo de polígonos de barrios
    """
    barrios_gdf = gpd.read_file(path_shapefile)
    if "nombre" in barrios_gdf.columns and "barrio" not in barrios_gdf.columns:
        barrios_gdf = barrios_gdf.rename(columns={"nombre": "barrio"})

    df_estaciones["geometry"] = df_estaciones.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    estaciones_gdf = gpd.GeoDataFrame(df_estaciones, geometry="geometry", crs=barrios_gdf.crs)

    estaciones_con_barrios = gpd.sjoin(estaciones_gdf, barrios_gdf, how="left", predicate="within")

    estaciones_con_barrios[["id_estacion", "lat", "lon", "barrio"]].to_csv("data/estaciones_con_barrios.csv", index=False)

    mapa_barrio = estaciones_con_barrios.set_index("id_estacion")["barrio"].to_dict()

    mapa_barrio[111] = "PUERTO MADERO"
    mapa_barrio[541] = "PALERMO"

    if test:
        if path_test is None or not os.path.exists(path_test):
            print("⚠️ Path inválido o no encontrado para test.")
            return
        df = pd.read_csv(path_test)
        df["barrio_origen"] = df["id_estacion_origen"].map(mapa_barrio)
        df["barrio_destino"] = df["id_estacion_destino"].map(mapa_barrio)
        df.to_csv(path_test, index=False)
        print(f"🧪 Archivo enriquecido (modo test): {path_test}")
        return

    if anios is None:
        print("⚠️ No se especificaron años para procesar.")
        return

    for anio in anios:
        path = f"data/recorridos/processed/trips_{anio}_pr.csv"
        if not os.path.exists(path):
            print(f"⚠️ Archivo no encontrado: {path}")
            continue

        df = pd.read_csv(path)
        df["barrio_origen"] = df["id_estacion_origen"].map(mapa_barrio)
        df["barrio_destino"] = df["id_estacion_destino"].map(mapa_barrio)
        df.to_csv(path, index=False)
        print(f" Enriquecido con barrios: {anio}")



def obtener_o_construir_df_estaciones(lista_paths, path_guardado="data/estaciones_unico.csv"):
    if os.path.exists(path_guardado):
        print("Usando df_estaciones ya guardado.")
        return pd.read_csv(path_guardado)

    print("Construyendo df_estaciones desde archivos...")
    estaciones_dict = {}

    for path in lista_paths:
        if not os.path.exists(path):
            print(f"Archivo no encontrado: {path}")
            continue

        df = pd.read_csv(path)

        for rol in ["origen", "destino"]:
            id_col = f"id_estacion_{rol}"
            lat_col = f"lat_estacion_{rol}"
            lon_col = f"long_estacion_{rol}"

            for _, row in df.iterrows():
                est_id = row[id_col]
                lat = row[lat_col]
                lon = row[lon_col]

                if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                    if est_id not in estaciones_dict:
                        estaciones_dict[est_id] = (lat, lon)

    df_estaciones = pd.DataFrame([
        {"id_estacion": est_id, "lat": lat, "lon": lon}
        for est_id, (lat, lon) in estaciones_dict.items()
    ])

    os.makedirs(os.path.dirname(path_guardado), exist_ok=True)
    df_estaciones.to_csv(path_guardado, index=False)
    print(f"df_estaciones guardado en {path_guardado}")

    return df_estaciones

def analyze_barrios(path_shapefile="data/barrios/barrios.shp"):
    """
    Analiza y muestra información general del shapefile de barrios.
    
    Parámetros:
    - path_shapefile: ruta al archivo .shp
    
    Salida:
    - Imprime columnas, cantidad de barrios únicos y sus nombres.
    """
    try:
        barrios_gdf = gpd.read_file(path_shapefile)
    except Exception as e:
        print(f"Error al leer el shapefile: {e}")
        return

    print("🧩 Columnas disponibles en el shapefile:")
    print(barrios_gdf.columns.tolist())

    if "nombre" not in barrios_gdf.columns:
        print("La columna 'nombre' no existe en el shapefile.")
        return

    barrios_unicos = sorted(barrios_gdf["nombre"].dropna().unique())
    total = len(barrios_unicos)

    print(f"\nCantidad de barrios únicos en el shapefile: {total}")
    print("\nBarrios:")
    print(barrios_unicos)
    print(f"\nTotal de barrios: {total}")


def analizar_presencia_estaciones(archivos, anios):
    estaciones_por_anio = {}

    for archivo, anio in zip(archivos, anios):
        if not os.path.exists(archivo):
            print(f"⚠️ Archivo no encontrado para el año {anio}")
            continue

        df = pd.read_csv(archivo)
        estaciones_origen = df['id_estacion_origen'].dropna().unique()
        estaciones_destino = df['id_estacion_destino'].dropna().unique()
        estaciones = set(estaciones_origen).union(set(estaciones_destino))
        estaciones_por_anio[anio] = estaciones

    todas_las_estaciones = sorted(set.union(*estaciones_por_anio.values()))
    tabla_presencia = pd.DataFrame(index=todas_las_estaciones)

    for anio in anios:
        estaciones = estaciones_por_anio.get(anio, set())
        tabla_presencia[anio] = ["✓" if est in estaciones else "" for est in tabla_presencia.index]

    tabla_presencia["Anios_presente"] = tabla_presencia[anios].apply(lambda row: sum(cell == "✓" for cell in row), axis=1)
    tabla_presencia = tabla_presencia.sort_index()

    print("\nPresencia de estaciones por año (por ID, ordenadas por cantidad de años presentes):")
    print(tabla_presencia)

    print("\nCantidad de estaciones por año:")
    for anio in anios:
        cantidad = len(estaciones_por_anio.get(anio, set()))
        print(f"  - {anio}: {cantidad} estaciones")

    return tabla_presencia, estaciones_por_anio


def mostrar_estaciones_faltantes_en_2024(estaciones_por_anio):
    print("\nEstaciones que aparecían en años anteriores pero NO en 2024:")

    estaciones_2024 = estaciones_por_anio.get(2024, set())

    for anio in [2020, 2021, 2022, 2023]:
        estaciones_anio = estaciones_por_anio.get(anio, set())
        solo_en_anio = sorted(estaciones_anio - estaciones_2024)
        print(f"\n Estaciones en {anio} pero NO en 2024 ({len(solo_en_anio)}):")
        print(solo_en_anio)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from matplotlib.lines import Line2D
import os

def estaciones_graf():

    colores_barrios = [
        "red", "green", "blue", "darkorange", "purple", "cyan", "magenta", "gold",
        "orchid", "tomato", "lightpink", "deepskyblue", "chartreuse", "firebrick", "sienna", "dodgerblue",
        "plum", "turquoise", "slateblue", "darkgreen", "hotpink", "indigo", "salmon", "navy",
        "darkred", "chocolate", "crimson", "teal", "coral", "darkviolet", "mediumorchid", "mediumseagreen",
        "darkmagenta", "mediumblue", "olivedrab", "slategray", "seagreen", "deeppink", "steelblue", "brown",
        "orangered", "forestgreen", "darkcyan", "violet", "palevioletred", "blueviolet", "darkslateblue", "limegreen"
    ]

    barrios_gdf = gpd.read_file("data/barrios/barrios.shp").to_crs(epsg=4326)
    barrios_unicos = sorted(barrios_gdf["nombre"].unique())
    assert len(colores_barrios) == len(barrios_unicos), "La cantidad de colores no coincide con la de barrios"
    color_dict = {barrio: colores_barrios[i] for i, barrio in enumerate(barrios_unicos)}

    archivos = [
        "data/recorridos/processed/trips_2024_pr.csv",
        "data/recorridos/processed/trips_2023_pr.csv",
        "data/recorridos/processed/trips_2022_pr.csv",
        "data/recorridos/processed/trips_2021_pr.csv",
        "data/recorridos/processed/trips_2020_pr.csv"
    ]

    estaciones_dict = {}
    for path in archivos:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        for rol in ["origen", "destino"]:
            for _, row in df.iterrows():
                est_id = row[f"id_estacion_{rol}"]
                lat = row[f"lat_estacion_{rol}"]
                lon = row[f"long_estacion_{rol}"]
                if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                    if est_id not in estaciones_dict:
                        estaciones_dict[est_id] = (lat, lon)

    df_estaciones_todas = pd.DataFrame([
        {"id_estacion": est_id, "lat": lat, "lon": lon}
        for est_id, (lat, lon) in estaciones_dict.items()
    ])
    df_estaciones_todas["geometry"] = df_estaciones_todas.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    gdf_estaciones_todas = gpd.GeoDataFrame(df_estaciones_todas, geometry="geometry", crs="EPSG:4326")


    estaciones_2024 = {}
    df_2024 = pd.read_csv("data/recorridos/processed/trips_2024_pr.csv")
    for rol in ["origen", "destino"]:
        for _, row in df_2024.iterrows():
            est_id = row[f"id_estacion_{rol}"]
            lat = row[f"lat_estacion_{rol}"]
            lon = row[f"long_estacion_{rol}"]
            if pd.notna(est_id) and pd.notna(lat) and pd.notna(lon):
                if est_id not in estaciones_2024:
                    estaciones_2024[est_id] = (lat, lon)

    df_estaciones_2024 = pd.DataFrame([
        {"id_estacion": est_id, "lat": lat, "lon": lon}
        for est_id, (lat, lon) in estaciones_2024.items()
    ])
    df_estaciones_2024["geometry"] = df_estaciones_2024.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)
    gdf_estaciones_2024 = gpd.GeoDataFrame(df_estaciones_2024, geometry="geometry", crs="EPSG:4326")
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    ax1, ax2 = axes

    for barrio, shape in barrios_gdf.groupby("nombre"):
        shape.plot(ax=ax1, color=color_dict[barrio], edgecolor="black", linewidth=0.5)
    gdf_estaciones_todas.plot(ax=ax1, color="black", edgecolor="white", markersize=40, alpha=0.9)
    ax1.set_title("📍 Todas las estaciones (2020–2024)", fontsize=18)
    ax1.set_xlabel("Longitud", fontsize=13)
    ax1.set_ylabel("Latitud", fontsize=13)
    ax1.axis("equal")
    ax1.grid(True)

    for barrio, shape in barrios_gdf.groupby("nombre"):
        shape.plot(ax=ax2, color=color_dict[barrio], edgecolor="black", linewidth=0.5)
    gdf_estaciones_2024.plot(ax=ax2, color="black", edgecolor="white", markersize=40, alpha=0.9)
    ax2.set_title("📍 Estaciones activas en 2024", fontsize=18)
    ax2.set_xlabel("Longitud", fontsize=13)
    ax2.set_ylabel("Latitud", fontsize=13)
    ax2.axis("equal")
    ax2.grid(True)

    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[barrio],
               markeredgecolor='black', markersize=10, label=barrio)
        for barrio in barrios_unicos
    ]
    fig.legend(handles=handles, title="Barrios", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize="small")

    plt.tight_layout()
    plt.show()


def analisis_usuarios(test=False, path_test=None):
    anios_recorridos = [2020, 2021, 2022, 2023, 2024]
    anios_usuarios_nuevos = [2015, 2016, 2017, 2018, 2019]

    if test:
        if not path_test or not os.path.exists(path_test):
            print("Path de test inválido o no encontrado.")
            return
        print(f"Modo test: analizando usuarios en {path_test}")
        df_usuarios_recorridos = pd.read_csv(path_test, usecols=["id_usuario"])
        df_usuarios_recorridos = df_usuarios_recorridos.dropna().astype({"id_usuario": int})
        df_usuarios_recorridos["año_recorrido"] = "TEST"
    else:
        usuarios_recorridos = []
        for anio in anios_recorridos:
            path = f"data/recorridos/processed/trips_{anio}_pr.csv"
            if os.path.exists(path):
                df = pd.read_csv(path, usecols=["id_usuario"])
                df = df.dropna().astype({"id_usuario": int})
                df["año_recorrido"] = anio
                usuarios_recorridos.append(df)
            else:
                print(f"No se encontró archivo de recorridos {anio}")
        df_usuarios_recorridos = pd.concat(usuarios_recorridos, ignore_index=True)

    usuarios_registrados = []
    for anio in anios_recorridos:
        path = f"data/usuarios/processed/usuarios_ecobici_{anio}_limpio.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["ID_usuario"])
            df = df.dropna().astype({"ID_usuario": int})
            usuarios_registrados.append(df)
        else:
            print(f" No se encontró archivo de usuarios {anio}")
    df_usuarios_registrados = pd.concat(usuarios_registrados, ignore_index=True)
    usuarios_2020_2024_set = set(df_usuarios_registrados["ID_usuario"].unique())

    df_no_registrados = df_usuarios_recorridos[
        ~df_usuarios_recorridos["id_usuario"].isin(usuarios_2020_2024_set)
    ].copy()

    usuarios_extra = []
    for anio in anios_usuarios_nuevos:
        path = f"data/new_data/processed/usuarios_ecobici_{anio}_limpio.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, dtype=str)
            df.columns = [col.replace('"', '') for col in df.columns]
            df = df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
            df["ID_usuario"] = df["ID_usuario"].astype(int)
            df["año_archivo"] = anio
            usuarios_extra.append(df[["ID_usuario", "año_archivo"]])
        else:
            print(f"No se encontró archivo de usuarios nuevos {anio}")
    df_usuarios_extra = pd.concat(usuarios_extra, ignore_index=True)
    usuarios_2015_2019_set = set(df_usuarios_extra["ID_usuario"].unique())

    if test:
        usuarios_en_test = df_no_registrados["id_usuario"].unique()
        total = len(usuarios_en_test)
        encontrados = sum(uid in usuarios_2015_2019_set for uid in usuarios_en_test)
        no_encontrados = total - encontrados
        print("\n🔍 Resultados (modo test):")
        print(f"Usuarios en test no registrados en 2020–2024: {total}")
        print(f"Encontrados en 2015–2019: {encontrados}")
        print(f"Siguen sin aparecer: {no_encontrados}")
    else:
        resultados = []
        for anio in sorted(df_no_registrados["año_recorrido"].unique()):
            usuarios_en_anio = df_no_registrados[df_no_registrados["año_recorrido"] == anio]["id_usuario"].unique()
            total = len(usuarios_en_anio)
            encontrados = sum(uid in usuarios_2015_2019_set for uid in usuarios_en_anio)
            no_encontrados = total - encontrados
            resultados.append({
                "año_recorrido": anio,
                "usuarios_faltantes_2020_2024": total,
                "encontrados_en_2015_2019": encontrados,
                "siguen_sin_aparecer": no_encontrados
            })

        df_resultado = pd.DataFrame(resultados)
        print(" Usuarios de recorridos no encontrados en 2020–2024, pero sí en 2015–2019:")
        print(df_resultado)
    usuarios_totalmente_desconocidos = set(df_no_registrados["id_usuario"]) - usuarios_2015_2019_set

    df_filas_desconocidas = df_usuarios_recorridos[
        df_usuarios_recorridos["id_usuario"].isin(usuarios_totalmente_desconocidos)
    ]

    if test:
        print(f"\nFilas generadas por usuarios totalmente desconocidos (ni 2015–2019 ni 2020–2024): {len(df_filas_desconocidas)}")
    else:
        conteo_filas_por_anio = (
            df_filas_desconocidas.groupby("año_recorrido")
            .size()
            .reset_index(name="filas_totales_usuarios_desconocidos")
        )
        print("\n Filas generadas por usuarios que no aparecen en ningún archivo (ni 2015–2019 ni 2020–2024):")
        print(conteo_filas_por_anio)


def preprocesamiento_completo(test=False, path_test=None):
    if test:
        print(" Preprocesamiento en modo TEST")
        if not path_test or not os.path.exists(path_test):
            raise FileNotFoundError(" Debés especificar un `path_test` válido.")

        verificar_coordenadas_todos_los_anios(test=True, path_csv=path_test)
        df_estaciones = obtener_o_construir_df_estaciones([])  # no necesita archivos si ya existe
        asignar_barrios_a_datasets(df_estaciones, test=True, path_test=path_test)
        analisis_usuarios(test=True, path_test=path_test)

    else:
        print(" Preprocesamiento en modo TRAIN (años 2020–2024)")

        ruta_entrada = "data/recorridos/raw/trips_2024.csv"
        ruta_salida = "data/recorridos/processed/trips_2024_pr.csv"
        meses = [9, 10, 11, 12]
        procesar_recorridos_anio(ruta_entrada, ruta_salida, meses)

        limpiar_y_guardar_recorridos()

        archivos = [
            "data/recorridos/processed/trips_2020_pr.csv",
            "data/recorridos/processed/trips_2021_pr.csv",
            "data/recorridos/processed/trips_2022_pr.csv",
            "data/recorridos/processed/trips_2023_pr.csv",
            "data/recorridos/processed/trips_2024_pr.csv",
        ]

        limpiar_y_normalizar_archivos(archivos, funciones)

        archivos_corregir = {
            2020: archivos[0],
            2021: archivos[1],
            2022: archivos[2],
            2023: archivos[3],
        }

        tabla, estaciones_por_anio = analizar_presencia_estaciones(archivos, [2020, 2021, 2022, 2023, 2024])
        mostrar_estaciones_faltantes_en_2024(estaciones_por_anio)

        verificar_consistencia_coordenadas("data/recorridos/processed/trips_2024_pr.csv", anio=2024)
        corregir_coordenadas_con_base_2024("data/recorridos/processed/trips_2024_pr.csv", archivos_corregir)
        corregir_longitudes_inconsistentes("data/recorridos/processed/trips_2021_pr.csv")
        corregir_estaciones_2020()
        verificar_coordenadas_todos_los_anios([2020, 2021, 2022, 2023])

        analyze_barrios()
        df_estaciones = obtener_o_construir_df_estaciones(archivos)
        asignar_barrios_a_datasets(df_estaciones, anios=[2020, 2021, 2022, 2023, 2024])
        estaciones_graf()
        analisis_usuarios()

    print(" Preprocesamiento completo.")