import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
from skimage.morphology import skeletonize
import re 

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Analizador de Movilidad EOR", layout="wide")
st.title("Evaluación de Micromodelos de Desplazamiento EOR: Tortuosidad y Velocidad de Polímeros")
st.markdown("---")

st.subheader("💧 Micromodelo Base - Inyección de Agua (Waterflooding al Breakthrough)")
try:
    # Carga la imagen local de waterflooding
    st.image("Iny Water.jpeg", caption="Micromodelo - Inyección de Agua", use_container_width=True)
except Exception as e:
    st.error("⚠️ No se encontró la imagen 'Iny Water.jpeg'. Asegúrate de que esté en la misma carpeta que el script.")
st.markdown("---")

# --- 1. CARGA DE IMAGEN MÚLTIPLE (BATCH PROCESSING) ---
st.subheader("🖼️ 1. Carga tus Micromodelos (Procesamiento por Lotes)")
# accept_multiple_files=True permite subir todas las imágenes de la carpeta a la vez
archivos_subidos = st.file_uploader("Selecciona múltiples imágenes JPG/PNG", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

# --- 2. INGRESO DE PARÁMETROS GLOBALES ---
st.sidebar.header("📝 2. Parámetros Físicos Constantes")
st.sidebar.info("El polímero, concentración y caudal se extraerán automáticamente de los nombres de los archivos.")

ancho_mm = st.sidebar.number_input("Ancho del Micromodelo (mm)", value=5.00)
ancho = ancho_mm / 10.0 # cm
espesor_mm = st.sidebar.number_input("Espesor del Micromodelo (mm)", value=0.08, format="%.3f")
espesor = espesor_mm / 10.0 # cm
Dp_cm_input = st.sidebar.number_input("Tamaño del Grano (mm)", value=0.03, format="%.3f")
Dp_cm = Dp_cm_input / 10.0 # cm
porosidad_abs = st.sidebar.number_input("Porosidad Absoluta (fracción)", min_value=0.01, max_value=1.0, value=0.39)

st.sidebar.markdown("---")
# Los nuevos controles ópticos dinámicos integrados
st.sidebar.subheader("🎛️ Calibración Óptica del Fluido")
st.sidebar.caption("Ajusta si el modelo detecta ruido de vidrio o pierde canales finos.")
matiz_min = st.sidebar.slider("Matiz Azul Mínimo", 80, 110, 100)
matiz_max = st.sidebar.slider("Matiz Azul Máximo", 130, 170, 140)
sat_min = st.sidebar.slider("Saturación Mínima", 20, 150, 50)

# Lista para almacenar los resultados consolidados
datos_consolidados = []

# --- 3. PROCESAMIENTO EN BUCLE (MATEMÁTICAS EN SEGUNDO PLANO) ---
if archivos_subidos:
    for archivo in archivos_subidos:
        nombre_archivo = archivo.name
        
        # Extracción automática vía Regex para esta imagen específica
        tipo_polimero = "Desconocido"
        val_q = 0.036
        val_ppm = 200
        
        match_polimero = re.search(r'Iny\s+([a-zA-Z]+)', nombre_archivo, re.IGNORECASE)
        if match_polimero:
            tipo_polimero = match_polimero.group(1).upper()
        
        match_q = re.search(r'Q\s*(\d+[,.]\d+)', nombre_archivo, re.IGNORECASE)
        if match_q:
            val_q = float(match_q.group(1).replace(',', '.'))
            
        match_ppm = re.search(r'(\d+)\s*ppm', nombre_archivo, re.IGNORECASE)
        if match_ppm:
            val_ppm = int(match_ppm.group(1))

        # Decodificación
        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        pixeles_totales = img.shape[0] * img.shape[1] 
        
        # Aislamiento del Polímero (Filtro Suavizado + HSV Dinámico)
        img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([matiz_min, sat_min, 50])
        upper_blue = np.array([matiz_max, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5,5), np.uint8)
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Cálculos de Porosidad
        pixeles_polimero = np.sum(mask_limpia == 255)
        porosidad_efectiva = pixeles_polimero / pixeles_totales 
        
        # Restricción Petrofísica
        if porosidad_efectiva >= porosidad_abs:
            porosidad_efectiva = porosidad_abs * 0.95
            
        # Esqueletización y Tortuosidad
        bool_mask = mask_limpia > 0
        esqueleto = skeletonize(bool_mask) 
        
        longitud_camino_pixeles = np.sum(esqueleto)
        longitud_recta_pixeles = img.shape[1] 
        tortuosidad = max(1.0, longitud_camino_pixeles / longitud_recta_pixeles)
        
        # Cinemática y Eficiencia
        q_cm3_s = val_q / 60.0 
        area_transversal_cm2 = ancho * espesor
        v_darcy = q_cm3_s / area_transversal_cm2
        v_intersticial = v_darcy / porosidad_efectiva if porosidad_efectiva > 0 else 0
        velocidad_real = v_intersticial * tortuosidad 
        
        longitud_calculada = ancho * (img.shape[1] / img.shape[0])
        area_total_vista_superior = ancho * longitud_calculada
        area_barrida_cm2 = porosidad_efectiva * area_total_vista_superior    
        eficiencia_barrido = min(1.0, porosidad_efectiva / porosidad_abs) if porosidad_abs > 0 else 0

        # Kozeny-Carman Modificado
        if porosidad_efectiva > 0 and tortuosidad > 0:
            S_vp = (2 / espesor) + ((4 * (1 - porosidad_efectiva)) / (porosidad_efectiva * Dp_cm))
            k_cm2 = porosidad_efectiva / (2 * tortuosidad * (S_vp**2))
            permeabilidad_mD = k_cm2 * 1.013e11 
        else:
            permeabilidad_mD = 0.0

        # Guardar en la tabla maestra
        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Polímero": tipo_polimero,
            "Concentración (ppm)": val_ppm,
            "Caudal (ml/min)": val_q,
            "Porosidad Efectiva (%)": porosidad_efectiva * 100,
            "Tortuosidad Areal (τ)": tortuosidad,
            "Velocidad Real (cm/s)": velocidad_real,
            "Área Barrida (cm²)": area_barrida_cm2,
            "Eficiencia Barrido EA (%)": eficiencia_barrido * 100,
            "Permeabilidad Mod. (mD)": permeabilidad_mD
        })
        
        # Resetear el archivo en memoria para poder visualizarlo después
        archivo.seek(0)

    # --- 4. REPORTE CONSOLIDADO EXCEL ---
    st.markdown("---")
    st.subheader("📋 Resumen Consolidado de la Corrida")
    df_maestro = pd.DataFrame(datos_consolidados)
    
    st.markdown("""
    <style>
    [data-testid="stTable"] table { border: none !important; border-collapse: collapse !important; }
    [data-testid="stTable"] th { border-bottom: 2px solid #2C3E50 !important; text-transform: uppercase !important; font-weight: 700 !important; color: #2C3E50 !important; }
    [data-testid="stTable"] td { border-bottom: 1px solid #E0E0E0 !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.table(df_maestro)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_maestro.to_excel(writer, index=False, sheet_name='Batch_Resultados_EOR')

    st.download_button(
        label="📊 Descargar Reporte Completo en Excel",
        data=buffer.getvalue(),
        file_name="Reporte_Consolidado_Micromodelos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # --- 5. INSPECCIÓN VISUAL INDIVIDUAL ---
    st.markdown("---")
    st.subheader("🔍 Inspección Visual Detallada")
    st.info("Selecciona un micromodelo específico del lote para visualizar sus máscaras y el análisis científico detallado.")
    
    nombres_archivos = [f.name for f in archivos_subidos]
    archivo_seleccionado = st.selectbox("Seleccionar Micromodelo:", nombres_archivos)
    
    # Extraer los datos guardados en el DataFrame para evitar recalcular matemáticamente todo
    datos_fila = df_maestro[df_maestro["Archivo"] == archivo_seleccionado].iloc[0]
    
    # Recuperar el archivo de imagen para dibujar la UI
    archivo_obj = next(f for f in archivos_subidos if f.name == archivo_seleccionado)
    archivo_obj.seek(0)
    file_bytes_ui = np.asarray(bytearray(archivo_obj.read()), dtype=np.uint8)
    img_ui = cv2.imdecode(file_bytes_ui, 1)
    
    # Re-aplicar solo los filtros visuales para dibujar las columnas
    img_suavizada_ui = cv2.GaussianBlur(img_ui, (5, 5), 0)
    hsv_ui = cv2.cvtColor(img_suavizada_ui, cv2.COLOR_BGR2HSV)
    mask_ui = cv2.inRange(hsv_ui, lower_blue, upper_blue)
    mask_limpia_ui = cv2.morphologyEx(mask_ui, cv2.MORPH_OPEN, kernel)
    esqueleto_ui = skeletonize(mask_limpia_ui > 0)

    # Visualización en Columnas
    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.markdown(f"**Original: {datos_fila['Polímero']}**")
        st.image(cv2.cvtColor(img_ui, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_img2:
        st.markdown("**Binarización**")
        st.image(mask_limpia_ui, use_container_width=True, clamp=True)
    with col_img3:
        st.markdown("**Esqueleto (Red de Canales)**")
        esqueleto_color = np.zeros((esqueleto_ui.shape[0], esqueleto_ui.shape[1], 3), dtype=np.uint8)
        esqueleto_color[esqueleto_ui] = [255, 255, 0] 
        esqueleto_grueso = cv2.dilate(esqueleto_color, np.ones((3,3), np.uint8), iterations=1)
        st.image(esqueleto_grueso, use_container_width=True, clamp=True)

    # --- 6. PANEL DE RESULTADOS CIENTÍFICO (SOLO PARA LA IMAGEN SELECCIONADA) ---
    st.markdown("---")
    st.subheader(f"📊 Análisis Petrofísico: {archivo_seleccionado}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fluido Inyectado", f"{datos_fila['Polímero']} {datos_fila['Concentración (ppm)']} ppm")
        st.metric("Caudal de Inyección", f"{datos_fila['Caudal (ml/min)']} ml/min")
    with col2:
        st.markdown("**Porosidad Efectiva**")
        st.latex(r"\phi_{eff} = \frac{Píxeles_{polímero}}{Píxeles_{Totales}}")
        st.latex(rf"\phi_{{eff}} = {datos_fila['Porosidad Efectiva (%)']:.1f} \%")
    with col3:
        st.markdown("**Tortuosidad Areal (τ)**")
        st.latex(r"\tau = \frac{L_e}{L_r}")
        st.latex(rf"\tau = {datos_fila['Tortuosidad Areal (τ)']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Cinemática del Fluido (Velocidades)")
    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        st.markdown("**Velocidad Real del Polímero**")
        st.latex(r"v_{Darcy} = \frac{q}{A_{transversal}} \quad ; \quad v_{int} = \frac{v_{Darcy}}{\phi_{eff}} \quad ; \quad v_{real} = v_{int} \cdot \tau")
        st.latex(rf"v_{{real}} = {datos_fila['Velocidad Real (cm/s)']:.6f} \text{{ cm/s}}")
    with col_v2:
        st.markdown(r"""
        **Donde:**
        *   $q$: Caudal de inyección ($\text{cm}^3\text{/s}$).
        *   $A_{transversal}$: Área de la sección transversal ($ancho \times espesor$).
        *   $\phi_{eff}$: Porosidad efectiva (fracción).
        *   $\tau$: Tortuosidad areal (adimensional).
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Eficiencia de Barrido y Geometría")
    col_a, col_b, col_c = st.columns(3)
    with col_b:
        st.markdown("**Área Real Barrida**")
        st.latex(r"A_B = A_T \cdot \left(\frac{Píxeles_{polímero}}{Píxeles_{Totales}}\right)")
        st.latex(rf"A_B = {datos_fila['Área Barrida (cm²)']:.4f} \text{{ cm}}^2")
    with col_c:
        st.markdown("**Eficiencia de Barrido Areal (EA)**")
        st.latex(r"E_A = \frac{A_B}{A_T} \times 100")
        st.latex(rf"E_A = {datos_fila['Eficiencia Barrido EA (%)']:.2f} \%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Propiedades Petrofísicas Modificadas")
    col_k1, col_k2 = st.columns([1.5, 1])
    with col_k1:
        st.markdown("**Permeabilidad Estimada - Modelo de Kozeny-Carman Modificado**")
        st.latex(r"S_{vp} = \frac{2}{h} + \frac{4 \cdot (1 - \phi_{eff})}{\phi_{eff} \cdot D_p}")
        st.latex(r"k = \frac{\phi_{eff}}{2 \cdot \tau \cdot S_{vp}^2}")
        st.latex(rf"k = {datos_fila['Permeabilidad Mod. (mD)']:.2f} \text{{ mD}}")
    with col_k2:
        st.markdown(r"""
        **Donde:**
        *   $S_{vp}$: Área superficial específica ($\text{cm}^{-1}$).
        *   $h$: Espesor del micromodelo ($\text{cm}$).
        *   $\phi_{eff}$: Porosidad efectiva (fracción).
        *   $D_p$: Diámetro del grano cilíndrico ($\text{cm}$).
        *   $\tau$: Tortuosidad areal (adimensional).
        """)
