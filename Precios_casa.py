%pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📌 Función para generar datos simulados
@st.cache_data
def generate_data():
    np.random.seed(42)
    size = np.random.randint(50, 300, 200)  # Tamaño de la casa en m²
    rooms = np.random.randint(1, 6, 200)  # Número de habitaciones
    age = np.random.randint(0, 50, 200)  # Antigüedad en años
    distance = np.random.uniform(0.5, 20, 200)  # Distancia al centro en km
    price = (size * 3000) + (rooms * 50000) - (age * 2000) - (distance * 10000) + np.random.normal(0, 50000, 200)

    df = pd.DataFrame({"Tamaño (m²)": size, "Habitaciones": rooms, "Antigüedad (años)": age, "Proximidad al Centro (km)": distance, "Precio": price})
    return df

# 📌 Función para entrenar y guardar el modelo
def train_and_save_model():
    df = generate_data()
    X = df.drop(columns=["Precio"])
    y = df["Precio"]
    
    # División en train (70%), validation (15%) y test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluación en conjunto de validación
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    # Guardar modelo entrenado
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, X_train, X_val, X_test, y_train, y_val, y_test, val_mse, val_r2

# 📌 Cargar modelo si ya existe, si no, entrenarlo
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    df = generate_data()
    X = df.drop(columns=["Precio"])
    y = df["Precio"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
else:
    model, X_train, X_val, X_test, y_train, y_val, y_test, val_mse, val_r2 = train_and_save_model()

# 📌 UI con Streamlit
st.title("Predicción de Precio de Viviendas 🏡")
st.write("Ingrese las características de la vivienda para obtener una estimación del precio.")

# 📌 Entrada del usuario
tamaño = st.number_input("Tamaño (m²):", min_value=30, max_value=500, value=100, step=10)
habitaciones = st.number_input("Habitaciones:", min_value=1, max_value=10, value=3, step=1)
antigüedad = st.number_input("Antigüedad (años):", min_value=0, max_value=100, value=10, step=1)
proximidad = st.number_input("Proximidad al Centro (km):", min_value=0.5, max_value=50.0, value=5.0, step=0.5)

# 📌 Predicción
if st.button("Predecir Precio"):
    input_data = np.array([[tamaño, habitaciones, antigüedad, proximidad]])
    precio_predicho = model.predict(input_data)[0]
    st.success(f"El precio estimado de la vivienda es: **${precio_predicho:,.2f}**")

# 📌 Evaluación del modelo en conjunto de test
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 📌 Mostrar métricas del modelo
st.write("### Evaluación del Modelo 📊")
st.write(f"🔹 Error Cuadrático Medio (MSE) - Validación: {val_mse:,.2f}")
st.write(f"🔹 R² Score - Validación: {val_r2:.2f}")
st.write(f"🔹 Error Cuadrático Medio (MSE) - Test: {test_mse:,.2f}")
st.write(f"🔹 R² Score - Test: {test_r2:.2f}")

# 📌 Mostrar datos de entrenamiento
st.write("### Datos de Entrenamiento 📊")
st.write(generate_data().head())


fig, ax = plt.subplots()
ax.scatter(y_test, y_test_pred, alpha=0.5, label="Predicción vs Realidad")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Línea ideal")
ax.set_xlabel("Precio Real")
ax.set_ylabel("Precio Predicho")
ax.legend()
st.pyplot(fig)

st.write("🚀 Desarrollado con Python y potenciado por Streamlit")