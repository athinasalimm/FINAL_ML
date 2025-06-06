import pandas as pd

import pandas as pd

import pandas as pd

def filtrar_meses(dataset, columna_fecha, meses_a_excluir):
    dataset[columna_fecha] = pd.to_datetime(dataset[columna_fecha], errors='coerce')
    filas_originales = len(dataset)
    df_filtrado = dataset[~dataset[columna_fecha].dt.month.isin(meses_a_excluir)].copy()
    eliminadas = filas_originales - len(df_filtrado)
    return df_filtrado, eliminadas