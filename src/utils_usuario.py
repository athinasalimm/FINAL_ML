import pandas as pd

def limpiar_y_codificar_usuarios(df):
    df = df.copy()

    # Asegurarse de que fecha y hora están en formato string
    df["fecha_alta"] = df["fecha_alta"].astype(str)
    df["hora_alta"] = df["hora_alta"].astype(str)

    # Separar fecha
    fecha_split = df["fecha_alta"].str.split("-", expand=True)
    df["año_alta"] = pd.to_numeric(fecha_split[0], errors='coerce')
    df["mes_alta"] = pd.to_numeric(fecha_split[1], errors='coerce')
    df["dia_alta"] = pd.to_numeric(fecha_split[2], errors='coerce')

    # Separar hora
    hora_split = df["hora_alta"].str.split(":", expand=True)
    df["hora_alta_h"] = pd.to_numeric(hora_split[0], errors='coerce')
    df["hora_alta_m"] = pd.to_numeric(hora_split[1], errors='coerce')
    df["hora_alta_s"] = pd.to_numeric(hora_split[2], errors='coerce')

    # One-hot encoding del género con 0 y 1 explícitos
    if "genero_usuario" in df.columns:
        dummies = pd.get_dummies(df["genero_usuario"], prefix="genero", drop_first=False).astype(int)
        df = pd.concat([df, dummies], axis=1)

    # Eliminar columnas originales
    df.drop(columns=["fecha_alta", "hora_alta", "genero_usuario"], inplace=True)

    return df