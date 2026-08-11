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
st.title("Evaluación de Micromodelos de Desplazamiento EOR: Balance de Masas por Píxeles")
st.markdown("---")

# --- (CONFIGURACIÓN PREVIA Y LECTURA DE CARPETA SE MANTIENE IGUAL) ---
CARPETA_MICROMODELOS = "micromodelos"
# ... [Código de lectura de carpeta idéntico] ...
if not os.path.exists(CARPETA_MICROMODELOS):
    st.warning("⚠️ Crea la carpeta 'micromodelos'.")
    archivos_validos = []
else:
    archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. PARÁMETROS FÍSICOS ---
st.sidebar.header("📝 2. Parámetros Físicos")
ancho_mm = st.sidebar.number_input("Ancho Micromodelo (mm)", value=180.00)
largo_mm = st.sidebar.number_input("Largo Micromodelo (mm)", value=200.00)
espesor_mm = st.sidebar.number_input("Espesor (mm)", value=0.800, format="%.3f")
porosidad_abs = st.sidebar.number_input("Porosidad Absoluta (fracción)", value=0.39)
t_bt = st.sidebar.number_input("Tiempo al Breakthrough (min)", value=650)

datos_consolidados = []

# --- 3. PROCESAMIENTO CÁLCULO DIRECTO (SIN RESTRICCIONES) ---
if archivos_validos:
    for nombre_archivo in archivos_validos:
        img = cv2.imread(os.path.join(CARPETA_MICROMODELOS, nombre_archivo))
        if img is None: continue
            
        # Binarización HSV (Optimiza estos rangos según la iluminación de tus fotos)
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 120, 100]), np.array([130, 255, 255]))
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # CÁLCULO MATEMÁTICO REAL
        total_pixels = img.shape[0] * img.shape[1]
        polymer_pixels = np.sum(mask_limpia == 255)
        
        # Poros disponibles teóricos (Área total * Porosidad)
        pore_pixels = total_pixels * porosidad_abs
        
        # Eficiencia de barrido real = Píxeles de polímero / Píxeles de poro disponibles
        # Aquí eliminamos el 'min(1.0, ...)' para ver la realidad de la imagen
        eficiencia_barrido = (polymer_pixels / pore_pixels) if pore_pixels > 0 else 0
        
        # Recuperación porcentual (%Fr)
        fr_porcentaje = eficiencia_barrido * 100
        
        # Saturación Residual (Sor)
        sor_fraccion = 1.0 - eficiencia_barrido
        
        # Volúmenes para la tabla
        area_total_cm2 = (ancho_mm / 10) * (largo_mm / 10)
        Vp_ml = area_total_cm2 * (espesor_mm / 10) * porosidad_abs
        Np_ml = (eficiencia_barrido * porosidad_abs) * Vp_ml
        
        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Eficiencia Barrido EA (%)": eficiencia_barrido * 100,
            "% Fr": fr_porcentaje,
            "Sor (fracción)": sor_fraccion,
            "Np al BT (ml)": Np_ml
        })

    # --- 4. TABLA RESULTADOS ---
    st.table(pd.DataFrame(datos_consolidados))
