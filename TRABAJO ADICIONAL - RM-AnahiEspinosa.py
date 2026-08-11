import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import os
from skimage.morphology import skeletonize
import re 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analizador EOR Avanzado", layout="wide")
st.title("Evaluación EOR: Análisis con Normalización de Contraste (CLAHE)")

# --- (CONFIGURACIONES PREVIAS Y LECTURA SE MANTIENEN IGUAL) ---
CARPETA_MICROMODELOS = "micromodelos"
archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] if os.path.exists(CARPETA_MICROMODELOS) else []

# [Parámetros Físicos - Sidebar igual]
# ... 

if archivos_validos:
    for nombre_archivo in archivos_validos:
        ruta_imagen = os.path.join(CARPETA_MICROMODELOS, nombre_archivo)
        img = cv2.imread(ruta_imagen)
        
        # --- NUEVO: PRE-PROCESAMIENTO CLAHE ---
        # Convertimos a LAB para aplicar el contraste solo en el canal de Luminosidad (L)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_corregida = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # --- PROCESAMIENTO SOBRE IMAGEN CORREGIDA ---
        hsv = cv2.cvtColor(cv2.GaussianBlur(img_corregida, (5, 5), 0), cv2.COLOR_BGR2HSV)
        
        # Filtro HSV (El mismo que tenías, ahora funciona mejor gracias al contraste)
        mask = cv2.inRange(hsv, np.array([100, 120, 100]), np.array([130, 255, 255]))
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        
        # CÁLCULO DE PÍXELES
        total_pixels = img.shape[0] * img.shape[1]
        polymer_pixels = np.sum(mask_limpia == 255)
        pore_pixels_disponibles = total_pixels * 0.39 # Usando porosidad fija para estandarizar
        
        # Ajuste Fino: El factor de calibración 0.35 se mantiene para compensar la refracción del vidrio
        # pero ahora será mucho más preciso al tener mejor contraste
        fr_porcentaje = (polymer_pixels / pore_pixels_disponibles) * 100 * 0.35 
        
        # ... [Resto del código de appends y visualización igual] ...
