from flask import Flask, render_template, request, jsonify
import folium
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
import random

app = Flask(__name__)

# Simular estaciones (esto se reemplazará por el modelo luego)
def generar_estaciones_cercanas(lat, lon, radio_km):
    estaciones = []
    for i in range(5):
        estaciones.append({
            "nombre": f"Estación {i+1}",
            "lat": lat + random.uniform(-0.01, 0.01),
            "lon": lon + random.uniform(-0.01, 0.01),
            "bicis_disponibles": random.randint(0, 10)
        })
    return estaciones

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/buscar", methods=["POST"])
def buscar():
    direccion = request.form["direccion"]
    tiempo = int(request.form["tiempo"])
    radio = float(request.form["radio"])

    geolocator = Nominatim(user_agent="bike-flask-app")
    ubicacion = geolocator.geocode(direccion)

    if not ubicacion:
        return jsonify({"error": "Dirección no encontrada"})

    lat, lon = ubicacion.latitude, ubicacion.longitude

    estaciones = generar_estaciones_cercanas(lat, lon, radio)

    # Crear mapa
    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], tooltip="Tu ubicación", icon=folium.Icon(color="blue")).add_to(m)
    folium.Circle([lat, lon], radius=radio*1000, color="red", fill=True, fill_opacity=0.1).add_to(m)

    for est in estaciones:
        folium.Marker(
            [est["lat"], est["lon"]],
            tooltip=f"{est['nombre']} ({est['bicis_disponibles']} bicis)",
            icon=folium.Icon(color="green")
        ).add_to(m)

    mapa_html = m._repr_html_()

    return jsonify({"mapa": mapa_html, "estaciones": estaciones})

@app.route("/heatmap", methods=["POST"])
def heatmap():
    minutos = int(request.form["minutos"])
    puntos = [[-34.60 + random.uniform(-0.01, 0.01), -58.38 + random.uniform(-0.01, 0.01)] for _ in range(30)]

    m = folium.Map(location=[-34.60, -58.38], zoom_start=13)
    HeatMap(puntos).add_to(m)
    mapa_html = m._repr_html_()

    return jsonify({"heatmap": mapa_html})

if __name__ == "__main__":
    app.run(debug=True)