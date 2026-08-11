import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import os
from skimage.morphology import skeletonize
import re 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analizador de Movilidad EOR", layout="wide")
st.title("Evaluación de Micromodelos de Desplazamiento EOR: Tortuosidad y Velocidad de Polímeros")

# --- 1. LECTURA AUTOMÁTICA DE LA CARPETA ---
CARPETA_MICROMODELOS = "micromodelos"
if not os.path.exists(CARPETA_MICROMODELOS):
    st.warning(f"⚠️ Crea la carpeta '{CARPETA_MICROMODELOS}' en tu repositorio.")
    archivos_validos = []
else:
    archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. PARÁMETROS FÍSICOS (BARRAS LATERALES) ---
st.sidebar.header("📝 2. Parámetros Físicos")
ancho_mm = st.sidebar.number_input("Ancho Micromodelo (mm)", value=180.00)
largo_mm = st.sidebar.number_input("Largo Micromodelo (mm)", value=200.00)
espesor_mm = st.sidebar.number_input("Espesor (mm)", value=0.800, format="%.3f")
porosidad_abs = st.sidebar.number_input("Porosidad Absoluta", value=0.39)
t_bt = st.sidebar.number_input("Tiempo al Breakthrough (min)", value=650)

datos_consolidados = []

if archivos_validos:
    for nombre_archivo in archivos_validos:
        ruta_imagen = os.path.join(CARPETA_MICROMODELOS, nombre_archivo)
        img = cv2.imread(ruta_imagen)
        if img is None: continue
            
        # Procesamiento
        img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([150, 255, 255]))
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # Volúmenes en ml
        area_total_cm2 = (ancho_mm / 10) * (largo_mm / 10)
        Vp_ml = area_total_cm2 * (espesor_mm / 10) * porosidad_abs
        
        fraccion_polimero = np.sum(mask_limpia == 255) / (img.shape[0] * img.shape[1])
        Np_ml = fraccion_polimero * area_total_cm2 * (espesor_mm / 10) # simplificado volumen barrido
        
        # --- CÁLCULO DE SATURACIÓN Y RECUPERACIÓN (IGUAL A LA TESIS) ---
        recuperacion_porcentaje = (Np_ml / Vp_ml) * 100
        saturacion_aceite = 100 - recuperacion_porcentaje
        
        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Polímero": re.search(r'Iny\s+([a-zA-Z]+)', nombre_archivo, re.IGNORECASE).group(1).upper() if re.search(r'Iny\s+([a-zA-Z]+)', nombre_archivo) else "N/A",
            "Concentración (ppm)": int(re.search(r'(\d+)\s*ppm', nombre_archivo, re.IGNORECASE).group(1)) if re.search(r'(\d+)\s*ppm', nombre_archivo) else 0,
            "Saturación de Aceite (%)": saturacion_aceite,
            "% Recuperación": recuperacion_porcentaje,
            "Np al BT (ml)": Np_ml,
            "VPI al BT": (float(re.search(r'Q\s*(\d+[,.]\d+)', nombre_archivo, re.IGNORECASE).group(1).replace(',', '.')) * t_bt) / Vp_ml if re.search(r'Q\s*(\d+[,.]\d+)', nombre_archivo) else 0
        })

    # --- 4. REPORTE CONSOLIDADO ---
    st.subheader("📋 RESUMEN RESULTADOS (Validación con Tesis)")
    df_maestro = pd.DataFrame(datos_consolidados)
    st.table(df_maestro.style.format({"Saturación de Aceite (%)": "{:.2f}", "% Recuperación": "{:.2f}", "Np al BT (ml)": "{:.2f}", "VPI al BT": "{:.2f}"}))

    # ... [RESTO DEL CÓDIGO (Panel de inspección, Cálculos matemáticos, Gráficas) SE MANTIENE IGUAL] ...
