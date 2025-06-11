import pandas as pd
import numpy as np

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
            print("❌ Columnas diferentes.")
            print("Diferencias:", set(columnas_ref).symmetric_difference(columnas_actual))
        else:
            print("✅ Columnas iguales.")

        tipos_actual = info_archivos[archivo]['tipos']
        diferencias_tipos = {
            col: (tipos_actual[col], tipos_ref[col])
            for col in columnas_ref
            if col in tipos_actual and tipos_actual[col] != tipos_ref[col]
        }

        if diferencias_tipos:
            print("❌ Tipos diferentes en algunas columnas:")
            for col, (tipo_act, tipo_ref) in diferencias_tipos.items():
                print(f" - {col}: {tipo_act} (vs {tipo_ref})")
        else:
            print("✅ Tipos de datos iguales.")
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

    # IDs
    for col in ["id_recorrido", "id_estacion_origen", "id_estacion_destino", "id_usuario"]:
        df[col] = df[col].astype(str).str.replace("BAEcobici", "", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Duración
    df["duracion_recorrido"] = df["duracion_recorrido"].astype(str).str.replace(",", "", regex=False)
    df["duracion_recorrido"] = pd.to_numeric(df["duracion_recorrido"], errors="coerce")

    # Coordenadas destino mezcladas
    for col in ["lat_estacion_destino", "long_estacion_destino"]:
        df[col] = df[col].astype(str).str.extract(r"(-?\d+\.\d+)")[0]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tipos explícitos
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

    # Coordenadas ya vienen bien, solo convertimos
    for col in ["lat_estacion_origen", "long_estacion_origen", "lat_estacion_destino", "long_estacion_destino"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Tipos explícitos
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



def verificar_consistencia_coordenadas(path_csv: str, anio: int = None):
    try:
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f"⚠️ No se encontró el archivo: {path_csv}")
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

    print(f"\n🔍 Verificando consistencia de coordenadas en {anio or path_csv}...")
    hay_errores = False
    for est_id, coords in sorted(coordenadas_por_estacion.items()):
        n_lat = len(coords["lat"])
        n_lon = len(coords["lon"])
        if n_lat > 1 or n_lon > 1:
            hay_errores = True
            diffs = []
            if n_lat > 1:
                diffs.append("lat")
                print(f" - Estación {est_id}: difiere en lat ({n_lat} latitudes distintas)")
            if n_lon > 1:
                diffs.append("long")
                print(f" - Estación {est_id}: difiere en long ({n_lon} longitudes distintas)")

    if not hay_errores:
        print("✅ Todas las estaciones tienen coordenadas consistentes.")


def corregir_longitudes_inconsistentes(path_csv: str):
    try:
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        print(f"⚠️ Archivo no encontrado: {path_csv}")
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
        print(f"✅ No se encontraron inconsistencias de longitud en {path_csv}")
        return

    longitudes_correctas = {}
    for est_id in estaciones_con_errores:
        longitudes = df.loc[df["id_estacion_origen"] == est_id, "long_estacion_origen"].dropna()
        if not longitudes.empty:
            longitudes_correctas[est_id] = longitudes.mode().iloc[0]
        else:
            print(f"⚠️ Estación {est_id} no aparece como origen en {path_csv}, no se corrige.")

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
    print(f"✅ Longitudes corregidas automáticamente en: {path_csv}")


def corregir_latitud_estacion(df: pd.DataFrame, est_id: int = 240) -> pd.DataFrame:
    """
    Corrige la latitud de una estación específica (por defecto, la 240)
    usando como referencia la latitud más frecuente cuando aparece como origen.

    Aplica la corrección tanto en origen como destino si la latitud es distinta.
    """
    latitudes = df.loc[df["id_estacion_origen"] == est_id, "lat_estacion_origen"].dropna()
    
    if latitudes.empty:
        print(f"⚠️ No se encontró la estación {est_id} como origen.")
        return df

    lat_correcta = latitudes.mode().iloc[0]
    print(f"ℹ️ Latitud de referencia para estación {est_id}: {lat_correcta}")

    def corregir_lat(row):
        for rol in ["origen", "destino"]:
            if row[f"id_estacion_{rol}"] == est_id:
                lat_col = f"lat_estacion_{rol}"
                if pd.notna(row[lat_col]) and row[lat_col] != lat_correcta:
                    row[lat_col] = lat_correcta
        return row

    df = df.apply(corregir_lat, axis=1)
    return df