import geopandas as gpd
from shapely.geometry import Point

barrios = gpd.read_file("data/barrios/barrios.geojson")

print("Columnas en barrios:", barrios.columns.tolist())  # <-- Esto muestra todas las columnas

lat = -34.6020389
lon = -58.3838762

punto = Point(lon, lat)  # (lon, lat)

barrio = barrios[barrios.contains(punto)]

if not barrio.empty:
    # Usá el nombre correcto que viste en el print de columnas
    nombre_columna = barrios.columns[0]  # Solo como ejemplo, cambia si es otro
    print("Barrio encontrado:", barrio.iloc[0][nombre_columna])
else:
    print("Barrio no encontrado")
