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