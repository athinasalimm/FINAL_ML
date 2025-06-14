from flask import Flask, request, jsonify, render_template
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import random
import pandas as pd
import plotly.express as px
import os

app = Flask(__name__)

data_dir = r"data\usuarios\processed"

def generar_estaciones_cercanas(lat, lon, radio_km):
    estaciones = []
    for i in range(5):
        estaciones.append({
            "nombre": f"Estación {i+1}",
            "lat": lat + random.uniform(-0.01, 0.01),
            "lon": lon + random.uniform(-0.01, 0.01),
            "bicis": random.randint(0, 10)
        })
    return estaciones

def cargar_datos_usuario(anio):
    ruta = os.path.join(data_dir, f"usuarios_ecobici_{anio}_limpio.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta)
    return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mapa")
def mapa():
    return render_template("mapa.html")

@app.route("/perfil")
def perfil():
    return render_template("perfil.html")

@app.route("/estadisticas")
def estadisticas():
    # Esta es la página que muestra el select y los gráficos
    return render_template("estadisticas.html")

@app.route("/usuarios/<int:anio>")
def estadisticas_usuarios(anio):
    df = cargar_datos_usuario(anio)
    print(f"Cargando datos para el año {anio}, df is None? {df is None}")
    if df is None:
        return jsonify({"html": f"<p>No hay datos para el año {anio}</p>"})

    # Verificamos columnas
    required_cols = ["ID_usuario", "genero_FEMALE", "genero_MALE", "genero_OTHER", "edad_usuario"]
    for col in required_cols:
        if col not in df.columns:
            return jsonify({"html": f"<p>Falta columna '{col}' en los datos del año {anio}</p>"})

    # Convertir edad_usuario a numérico para evitar errores
    df["edad_usuario"] = pd.to_numeric(df["edad_usuario"], errors='coerce')

    usuarios_unicos = df["ID_usuario"].nunique()
    usuarios_unicos_fmt = f"{usuarios_unicos:,}"

    edad_promedio = df["edad_usuario"].mean()

    # Convertir columnas de género a numérico y llenar NaN con 0
    for col in ["genero_FEMALE", "genero_MALE", "genero_OTHER"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    total_generos = df["genero_FEMALE"].sum() + df["genero_MALE"].sum() + df["genero_OTHER"].sum()
    if total_generos == 0:
        porc_female = porc_male = porc_other = 0
    else:
        porc_female = df["genero_FEMALE"].sum() / total_generos * 100
        porc_male = df["genero_MALE"].sum() / total_generos * 100
        porc_other = df["genero_OTHER"].sum() / total_generos * 100

    fig_edades = px.histogram(
        df, x="edad_usuario", nbins=20,
        title=f"Distribución de edades en {anio}",
        labels={"edad_usuario": "Edad"},
        color_discrete_sequence=["#a8dadc"]
    )
    html_grafico_edades = fig_edades.to_html(include_plotlyjs='cdn', full_html=False)

    html = f"""
        <p><strong>Usuarios únicos en {anio}:</strong> {usuarios_unicos_fmt}</p>
        <p><strong>Edad promedio:</strong> {edad_promedio:.1f} años</p>
        <p><strong>Porcentaje de mujeres:</strong> {porc_female:.1f}%</p>
        <p><strong>Porcentaje de hombres:</strong> {porc_male:.1f}%</p>
        <p><strong>Porcentaje de otros géneros:</strong> {porc_other:.1f}%</p>
        {html_grafico_edades}
    """
    return jsonify({"html": html})



@app.route("/api/altas_por_anio")
def altas_por_anio():
    conteo = {}
    for anio in range(2020, 2025):
        df = cargar_datos_usuario(anio)
        if df is not None:
            conteo[str(anio)] = df["ID_usuario"].nunique()
        else:
            conteo[str(anio)] = 0
    return jsonify(conteo)

@app.route("/buscar_estaciones", methods=["POST"])
def buscar_estaciones():
    data = request.get_json()
    direccion = data.get("direccion", "")
    minutos = int(data.get("minutos", 10))
    radio = float(data.get("radio", 1.5))

    geolocator = Nominatim(user_agent="bike-flask-app")
    ubicacion = geolocator.geocode(direccion)

    if not ubicacion:
        return jsonify({"error": "Dirección no encontrada"}), 404

    lat, lon = ubicacion.latitude, ubicacion.longitude
    estaciones = generar_estaciones_cercanas(lat, lon, radio)

    return jsonify({
        "lat": lat,
        "lon": lon,
        "estaciones": estaciones
    })

@app.route("/heatmap", methods=["POST"])
def heatmap():
    data = request.get_json()
    minutos = int(data.get("minutos", 10))  # podés usar esto luego si hacés predicciones por tiempo

    # Cargar dataset
    df = pd.read_csv(r"C:\Users\Catalina\OneDrive\Documents\UDESAa\AñoIII\Machine Learning\Proyecto_Final\FINAL_ML\data\estaciones_con_barrios.csv")  # cambiá si está en otra ruta

    # Convertir a lista de puntos: lat, lon, barrio
    df = df.dropna(subset=["lat", "lon", "barrio"])
    puntos = df[["lat", "lon", "barrio"]].values.tolist()

    # Coordenadas para centrar el mapa
    base_lat = df["lat"].mean()
    base_lon = df["lon"].mean()

    return jsonify({
        "puntos": puntos,
        "base_lat": base_lat,
        "base_lon": base_lon
    })

@app.route("/perfil_usuario", methods=["POST"])
def perfil_usuario():
    data = request.get_json()
    direccion = data.get("direccion", "")
    edad = data.get("edad", None)
    genero = data.get("genero", None)

    estaciones = [
        {"nombre": "Estación 1", "lat": -34.599, "lon": -58.381, "bicis": 3},
        {"nombre": "Estación 2", "lat": -34.603, "lon": -58.385, "bicis": 5},
        {"nombre": "Estación 3", "lat": -34.609, "lon": -58.377, "bicis": 2},
    ]

    geo = Nominatim(user_agent="ecobici-app")
    ubicacion = geo.geocode(direccion)
    if not ubicacion:
        return jsonify({"error": "Dirección no encontrada"}), 404

    user_coord = (ubicacion.latitude, ubicacion.longitude)

    estaciones_cercanas = []
    for est in estaciones:
        est_coord = (est["lat"], est["lon"])
        dist = geodesic(user_coord, est_coord).km
        if dist <= 1.5:
            estaciones_cercanas.append(est)

    zona = "zona céntrica" if user_coord[0] < -34.60 else "zona norte"

    return jsonify({
        "zona": zona,
        "lat": ubicacion.latitude,
        "lon": ubicacion.longitude,
        "radio": 1.5,
        "estaciones": estaciones_cercanas,
        "perfil": {
            "edad": edad,
            "genero": genero,
            "direccion": direccion
        }
    })

@app.route("/grafico_edades_por_anio")
def grafico_edades_por_anio():
    # Cargar datos de todos los años
    dfs = []
    for anio in range(2020, 2025):
        df = cargar_datos_usuario(anio)
        if df is not None:
            df = df.copy()
            df["anio"] = anio
            dfs.append(df)
    if not dfs:
        return jsonify({"html": "<p>No hay datos disponibles para los años solicitados.</p>"})
    
    df_all = pd.concat(dfs)

    # Gráfico de barras agrupadas por año y edad
    fig = px.histogram(
        df_all,
        x="edad_usuario",
        color="anio",
        barmode="group",   # barra agrupada
        nbins=20,
        labels={"edad_usuario": "Edad", "anio": "Año"},
        title="Distribución de edades por año",
        color_discrete_sequence=px.colors.qualitative.Safe  # colores distintos
    )

    html_grafico = fig.to_html(include_plotlyjs='cdn', full_html=False)
    return jsonify({"html": html_grafico})


@app.route("/api/estaciones")
def api_estaciones():
    estaciones = generar_estaciones_cercanas(-34.6, -58.38, 2)
    return jsonify({"estaciones": estaciones})

if __name__ == "__main__":
    app.run(debug=True)
