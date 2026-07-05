import streamlit as st
import cv2
import numpy as np
from skimage.morphology import skeletonize
import re

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador EOR", layout="wide")
st.title("Simulador de Análisis de Imagen: Inyección de Polímeros")
st.markdown("---")

# --- 1. CARGA DE IMAGEN Y LECTURA DE DATOS ---
st.subheader("🖼️ 1. Carga tu Micromodelo")
archivo_subido = st.file_uploader("Selecciona o arrastra una imagen JPG/PNG", type=['jpg', 'jpeg', 'png'])

# Valores por defecto
val_q = 0.036
val_ppm = 200
tipo_polimero = "Polímero" 

if archivo_subido is not None:
    nombre_archivo = archivo_subido.name
    
    # Extraer variables desde el nombre del archivo usando Regex
    match_polimero = re.search(r'Iny\s+([a-zA-Z]+)', nombre_archivo, re.IGNORECASE)
    if match_polimero:
        tipo_polimero = match_polimero.group(1).upper()
    
    match_q = re.search(r'Q\s*(\d+[,.]\d+)', nombre_archivo, re.IGNORECASE)
    if match_q:
        val_q = float(match_q.group(1).replace(',', '.'))
        
    match_ppm = re.search(r'(\d+)\s*ppm', nombre_archivo, re.IGNORECASE)
    if match_ppm:
        val_ppm = int(match_ppm.group(1))

# --- 2. INGRESO DE PARÁMETROS EN BARRA LATERAL ---
st.sidebar.header("📝 2. Parámetros del Experimento")
st.sidebar.text_input("Polímero Detectado", value=tipo_polimero, disabled=True)
caudal = st.sidebar.number_input("Caudal de Inyección (ml/min)", value=val_q, format="%.4f")
concentracion = st.sidebar.number_input("Concentración (ppm)", value=val_ppm)

st.sidebar.markdown("---")
st.sidebar.subheader("Datos Físicos del Modelo")
ancho = st.sidebar.number_input("Ancho del modelo (cm)", value=5.0)
espesor = st.sidebar.number_input("Espesor (cm)", value=0.1)

tipo_porosidad = st.sidebar.radio("¿Cómo ingresar la porosidad?", 
                                  ("Ingreso Manual", "Calcular ópticamente desde la imagen"))

if tipo_porosidad == "Ingreso Manual":
    porosidad = st.sidebar.number_input("Porosidad (fracción)", min_value=0.01, max_value=1.0, value=0.40)
else:
    st.sidebar.success("La porosidad se calculará automáticamente con el Método de Otsu.")
    porosidad = None 

# --- 3. PROCESAMIENTO VISUAL Y CÁLCULOS MATEMÁTICOS ---
if archivo_subido is not None:
    # Decodificar imagen subida
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- Cálculo de Porosidad Dinámica (Otsu) ---
    if porosidad is None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        umbral_optimo, mascara_poros = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        pixeles_vacios = np.sum(mascara_poros == 255)
        pixeles_totales = img.shape[0] * img.shape[1]
        porosidad = pixeles_vacios / pixeles_totales
        
        # Seguro matemático en caso de error de imagen extrema
        if porosidad == 0:
            porosidad = 0.001
            
        st.sidebar.info(f"Umbral automático (Otsu) calculado: {umbral_optimo:.0f}")

    # --- Aislamiento del Polímero y Esqueletización ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Rango para aislar la tinta azul (ajustable si se cambia el tinte del trazador)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((5,5), np.uint8)
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    bool_mask = mask_limpia > 0
    esqueleto = skeletonize(bool_mask)
    
    # --- Cálculos Geométricos y Cinemáticos ---
    longitud_camino_pixeles = np.sum(esqueleto)
    longitud_recta_pixeles = img.shape[1] 
    tortuosidad = max(1.0, longitud_camino_pixeles / longitud_recta_pixeles)
    
    q_cm3_s = caudal / 60.0
    area_cm2 = ancho * espesor
    v_darcy = q_cm3_s / area_cm2
    v_intersticial = v_darcy / porosidad
    velocidad_real = v_intersticial * tortuosidad

    # --- 4. PANEL DE RESULTADOS Y VISUALIZACIÓN ---
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fluido Inyectado", f"{tipo_polimero} {concentracion} ppm") 
    col2.metric("Porosidad", f"{porosidad:.2%}")
    col3.metric("Tortuosidad", f"{tortuosidad:.4f}")
    col4.metric("Velocidad Canal", f"{velocidad_real:.6f} cm/s")
    
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Original", "Binarización (Matriz)", "Esqueleto (Red de Canales)"])
    
    with tab1: 
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    with tab2: 
        st.image(mask_limpia, use_container_width=True, clamp=True)
    with tab3: 
        # Pintar el esqueleto de verde neón para máxima claridad visual
        esqueleto_color = np.zeros((esqueleto.shape[0], esqueleto.shape[1], 3), dtype=np.uint8)
        esqueleto_color[esqueleto] = [0, 255, 0] 
        st.image(esqueleto_color, use_container_width=True, clamp=True)

    # --- 5. INTERPRETACIÓN TÉCNICA DINÁMICA ---
    st.markdown("---")
    st.subheader(f"💡 Resultados de {tipo_polimero}")
    
    if tortuosidad > 8.0:
        comportamiento_red = "alta ramificación (amplia red de canales interconectados)."
        eficiencia = "una mejora significativa en la eficiencia de barrido areal"
    elif tortuosidad > 3.0:
        comportamiento_red = "ramificación moderada."
        eficiencia = "un barrido areal estándar con cierta mitigación de la canalización"
    else:
        comportamiento_red = "una ruta preferencial muy directa."
        eficiencia = "una posible ruptura temprana (breakthrough) por digitación viscosa"

    st.info(f"""
    **Análisis de la inyección de {tipo_polimero} a {concentracion} ppm:**
    
    * **Comportamiento Areal:** El valor de tortuosidad de **{tortuosidad:.2f}** indica {comportamiento_red} Al contrastar distintos fluidos, este parámetro nos revela que el **{tipo_polimero}** inyectado está logrando {eficiencia} frente al agua de formación, gracias al control sobre la relación de movilidad ($M \le 1$).
    
    * **Cinemática del Frente:** La velocidad real en el canal preferencial es de **{velocidad_real:.6f} cm/s**. Controlar esta velocidad intersticial es vital para dar tiempo a que los mecanismos físico-químicos del **{tipo_polimero}** actúen sobre el crudo residual en los poros, sin exceder el gradiente de fractura de la matriz rocosa.
    """)
