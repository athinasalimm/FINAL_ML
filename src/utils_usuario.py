import pandas as pd
from datetime import datetime

def limpiar_y_codificar_usuarios(df, es=False):
    df = df.copy()

    # Asegurarse de que fecha y hora están en formato string
    df["fecha_alta"] = df["fecha_alta"].astype(str)
    df["hora_alta"] = df["hora_alta"].astype(str)

    # Separar fecha
    fecha_split = df["fecha_alta"].str.split("-", expand=True)
    df["año_alta"] = pd.to_numeric(fecha_split[0], errors='coerce')
    df["mes_alta"] = pd.to_numeric(fecha_split[1], errors='coerce')
    df["dia_alta"] = pd.to_numeric(fecha_split[2], errors='coerce')

    if es:
        # Detectar filas con AM o PM
        df["hora_alta"] = df["hora_alta"].str.strip()
        mask_am_pm = df["hora_alta"].str.contains("AM|PM", na=False)

        def convertir_hora_am_pm(hora_str):
            try:
                return datetime.strptime(hora_str, "%I:%M:%S %p").strftime("%H:%M:%S")
            except:
                try:
                    return datetime.strptime(hora_str, "%I:%M %p").strftime("%H:%M:%S")
                except:
                    print(f"FALLÓ al convertir: {hora_str}")
                    return None

        df.loc[mask_am_pm, "hora_alta"] = df.loc[mask_am_pm, "hora_alta"].apply(convertir_hora_am_pm)

    hora_split = df["hora_alta"].str.split(":", expand=True)
    df["hora_alta_h"] = pd.to_numeric(hora_split[0], errors='coerce')
    df["hora_alta_m"] = pd.to_numeric(hora_split[1], errors='coerce')
    df["hora_alta_s"] = pd.to_numeric(hora_split[2], errors='coerce')

    # One-hot encoding robusto del género
    if "genero_usuario" in df.columns:
        categorias_genero = ["FEMALE", "MALE", "OTHER"]
        dummies = pd.get_dummies(df["genero_usuario"], prefix="genero").astype(int)

        # Asegurar todas las columnas deseadas
        for genero in categorias_genero:
            col = f"genero_{genero}"
            if col not in dummies.columns:
                dummies[col] = 0

        # Ordenar consistentemente
        dummies = dummies[[f"genero_{g}" for g in categorias_genero]]
        df = pd.concat([df, dummies], axis=1)

    # Eliminar columnas originales
    df.drop(columns=["fecha_alta", "hora_alta", "genero_usuario"], inplace=True)

    return df
