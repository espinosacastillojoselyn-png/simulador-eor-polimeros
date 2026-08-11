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

# --- 1. LECTURA AUTOMÁTICA DE LA CARPETA ---
CARPETA_MICROMODELOS = "micromodelos"
if not os.path.exists(CARPETA_MICROMODELOS):
    st.warning("⚠️ Crea la carpeta 'micromodelos' en tu repositorio.")
    archivos_validos = []
else:
    archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. INGRESO DE PARÁMETROS FÍSICOS ---
st.sidebar.header("📝 2. Parámetros Físicos")
ancho_mm = st.sidebar.number_input("Ancho Micromodelo (mm)", value=180.00)
largo_mm = st.sidebar.number_input("Largo Micromodelo (mm)", value=200.00)
espesor_mm = st.sidebar.number_input("Espesor (mm)", value=0.800, format="%.3f")
porosidad_abs = st.sidebar.number_input("Porosidad Absoluta (fracción)", value=0.39)
t_bt = st.sidebar.number_input("Tiempo al Breakthrough (min)", value=650)

datos_consolidados = []

# --- 3. PROCESAMIENTO CÁLCULO DIRECTO (MÉTODO DE PÍXELES PUROS) ---
if archivos_validos:
    for nombre_archivo in archivos_validos:
        img = cv2.imread(os.path.join(CARPETA_MICROMODELOS, nombre_archivo))
        if img is None: continue
            
        # Filtro HSV estricto (Ajustar aquí si detectas ruido)
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([100, 120, 100]), np.array([130, 255, 255]))
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # CÁLCULO DE BALANCE DE MASAS
        total_pixels = img.shape[0] * img.shape[1]
        polymer_pixels = np.sum(mask_limpia == 255)
        pore_pixels_disponibles = total_pixels * porosidad_abs
        
        # Recuperación Fraccional Real (%Fr)
        fr_porcentaje = (polymer_pixels / pore_pixels_disponibles) * 100 if pore_pixels_disponibles > 0 else 0
        
        # Saturación Residual Real (Sor)
        sor_fraccion = max(0.0, 1.0 - (fr_porcentaje / 100.0))
        
        # Cinemática y otras propiedades
        tortuosidad = max(1.0, np.sum(skeletonize(mask_limpia > 0)) / img.shape[1])
        area_total_cm2 = (ancho_mm / 10) * (largo_mm / 10)
        Np_ml = (fr_porcentaje / 100) * area_total_cm2 * (espesor_mm / 10) * porosidad_abs
        
        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Eficiencia Barrido EA (%)": fr_porcentaje, # EA es equivalente a Fr en micromodelos de saturación inicial 1
            "% Fr": fr_porcentaje,
            "Sor (fracción)": sor_fraccion,
            "Np al BT (ml)": Np_ml
        })

    # --- 4. REPORTE Y VISUALIZACIÓN ---
    st.subheader("📋 RESUMEN RESULTADOS (Cálculo Basado en Píxeles)")
    df_maestro = pd.DataFrame(datos_consolidados)
    st.table(df_maestro.style.format({"Eficiencia Barrido EA (%)": "{:.2f}", "% Fr": "{:.2f}", "Sor (fracción)": "{:.4f}", "Np al BT (ml)": "{:.2f}"}))

    # --- 5. REFERENCIAS BIBLIOGRÁFICAS ---
    st.markdown("---")
    st.subheader("📚 Respaldo Teórico")
    st.markdown("""
    > **Herrera Silva, L. R. (2020).** *Estudio experimental del desplazamiento y eficiencia de una inundación polimérica en micromodelos transparentes*. Tesis de Maestría. Universidad de Buenos Aires, Facultad de Ingeniería (IGPUBA).
    
    *Validación: Los resultados de %Fr y Sor se calculan mediante balance de masas volumétrico directo, utilizando el área barrida efectiva extraída por segmentación HSV sobre el área porosa total disponible.*
    """)
else:
    st.info("Sube las imágenes a la carpeta 'micromodelos'.")
