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
st.markdown("---")

st.subheader("💧 Micromodelo Base - Inyección de Agua (Waterflooding al Breakthrough)")
try:
    st.image("Iny Water.jpeg", caption="Micromodelo - Inyección de Agua", use_container_width=True)
except Exception as e:
    st.error("⚠️ No se encontró la imagen 'Iny Water.jpeg'. Asegúrate de que esté en el repositorio.")
st.markdown("---")

# --- 1. LECTURA AUTOMÁTICA DE LA CARPETA ---
st.subheader("🖼️ 1. Procesamiento Automático de Micromodelos")
st.write("El sistema está leyendo y procesando automáticamente las imágenes desde la base de datos del proyecto.")

CARPETA_MICROMODELOS = "micromodelos"

if not os.path.exists(CARPETA_MICROMODELOS):
    st.warning(f"⚠️ No se detectó la carpeta '{CARPETA_MICROMODELOS}'. Por favor, créala en tu repositorio y sube las imágenes.")
    archivos_validos = []
else:
    archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. INGRESO DE PARÁMETROS GLOBALES ---
st.sidebar.header("📝 2. Parámetros Físicos")
st.sidebar.info("Dimensiones estándar del Micromodelo de la Tesis (Ancho 180, Largo 200, Espesor 0.8)")

# DIMENSIONES EXACTAS DEL MODELO FÍSICO
ancho_mm = st.sidebar.number_input("Ancho del Micromodelo (mm)", value=180.00)
ancho = ancho_mm / 10.0 # cm
largo_mm = st.sidebar.number_input("Largo del Micromodelo (mm)", value=200.00)
largo_cm = largo_mm / 10.0 # cm
espesor_mm = st.sidebar.number_input("Espesor del Micromodelo (mm)", value=0.800, format="%.3f")
espesor = espesor_mm / 10.0 # cm

Dp_cm_input = st.sidebar.number_input("Tamaño del Grano (mm)", value=3.00, format="%.3f") 
Dp_cm = Dp_cm_input / 10.0 # cm
porosidad_abs = st.sidebar.number_input("Porosidad Absoluta (fracción)", min_value=0.01, max_value=1.0, value=0.39)

t_bt = st.sidebar.number_input("Tiempo al Breakthrough (min)", value=650, step=10)

st.sidebar.markdown("---")
st.sidebar.success("🤖 Calibración Óptica Automatizada Activada.")

datos_consolidados = []

# --- 3. PROCESAMIENTO EN BUCLE (MATEMÁTICAS EN SEGUNDO PLANO) ---
if archivos_validos:
    for nombre_archivo in archivos_validos:
        ruta_imagen = os.path.join(CARPETA_MICROMODELOS, nombre_archivo)
        
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

        img = cv2.imread(ruta_imagen)
        if img is None:
            continue
            
        pixeles_totales = img.shape[0] * img.shape[1] 
        
        img_suavizada = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([150, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5,5), np.uint8)
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_limpia = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel)
        
        # Fracción del volumen total que es polímero
        pixeles_polimero = np.sum(mask_limpia == 255)
        fraccion_polimero_total = pixeles_polimero / pixeles_totales 
        
        bool_mask = mask_limpia > 0
        esqueleto = skeletonize(bool_mask) 
        
        longitud_camino_pixeles = np.sum(esqueleto)
        longitud_recta_pixeles = img.shape[1] 
        tortuosidad = max(1.0, longitud_camino_pixeles / longitud_recta_pixeles)
        
        # Cinemática
        area_transversal_cm2 = ancho * espesor
        q_cm3_s = val_q / 60.0 
        v_darcy = q_cm3_s / area_transversal_cm2
        v_intersticial = v_darcy / porosidad_abs if porosidad_abs > 0 else 0
        velocidad_real = v_intersticial * tortuosidad 
        
        # Volúmenes Geométricos Fijos
        area_total_cm2 = ancho * largo_cm
        Vp_ml = area_total_cm2 * espesor * porosidad_abs
        
        # Eficiencia Areal (Área polímero / Área de poros)
        eficiencia_barrido = min(1.0, fraccion_polimero_total / porosidad_abs) if porosidad_abs > 0 else 0
        area_barrida_cm2 = eficiencia_barrido * area_total_cm2
        
        # Permeabilidad de Kozeny-Carman
        if fraccion_polimero_total > 0 and tortuosidad > 0:
            S_vp = (2 / espesor) + ((4 * (1 - porosidad_abs)) / (porosidad_abs * Dp_cm))
            k_cm2 = porosidad_abs / (2 * tortuosidad * (S_vp**2))
            permeabilidad_mD = k_cm2 * 1.013e11 
        else:
            permeabilidad_mD = 0.0

        # Cálculo dinámico final de volumen recuperado (Np)
        Np_ml = fraccion_polimero_total * area_total_cm2 * espesor
        V_iny_ml = val_q * t_bt
        VPI_bt = V_iny_ml / Vp_ml if Vp_ml > 0 else 0

        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Polímero": tipo_polimero,
            "Concentración (ppm)": val_ppm,
            "Caudal (ml/min)": val_q,
            "Tortuosidad Areal (τ)": tortuosidad,
            "Velocidad Real (cm/s)": velocidad_real,
            "Eficiencia Barrido EA (%)": eficiencia_barrido * 100,
            "Permeabilidad Mod. (mD)": permeabilidad_mD,
            "Np al BT (ml)": Np_ml,
            "VPI al BT": VPI_bt
        })

    # --- 4. REPORTE CONSOLIDADO EXCEL ---
    st.markdown("---")
    st.subheader("📋 RESUMEN RESULTADOS")
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
    st.subheader("🔍 MICROMODELO")
    st.info("Selecciona un micromodelo específico del lote para visualizar sus máscaras y el análisis científico detallado.")
    
    archivo_seleccionado = st.selectbox("Seleccionar Micromodelo:", archivos_validos)
    
    datos_fila = df_maestro[df_maestro["Archivo"] == archivo_seleccionado].iloc[0]
    
    img_ui = cv2.imread(os.path.join(CARPETA_MICROMODELOS, archivo_seleccionado))
    
    img_suavizada_ui = cv2.GaussianBlur(img_ui, (5, 5), 0)
    hsv_ui = cv2.cvtColor(img_suavizada_ui, cv2.COLOR_BGR2HSV)
    mask_ui = cv2.inRange(hsv_ui, np.array([90, 40, 40]), np.array([150, 255, 255]))
    kernel_ui = np.ones((5,5), np.uint8)
    mask_limpia_ui = cv2.morphologyEx(mask_ui, cv2.MORPH_OPEN, kernel_ui)
    mask_limpia_ui = cv2.morphologyEx(mask_limpia_ui, cv2.MORPH_CLOSE, kernel_ui)
    esqueleto_ui = skeletonize(mask_limpia_ui > 0)

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

    # --- 6. PANEL DE RESULTADOS CIENTÍFICO ---
    st.markdown("---")
    st.subheader(f"📊 CÁLCULOS DE: {archivo_seleccionado}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fluido Inyectado", f"{datos_fila['Polímero']} {datos_fila['Concentración (ppm)']} ppm")
        st.metric("Caudal de Inyección", f"{datos_fila['Caudal (ml/min)']} ml/min")
    with col2:
        st.markdown("**Eficiencia de Barrido Areal (EA)**")
        st.latex(r"E_A = \frac{A_B}{A_T} \times 100")
        st.latex(rf"E_A = {datos_fila['Eficiencia Barrido EA (%)']:.2f} \%")
    with col3:
        st.markdown("**Tortuosidad Areal (τ)**")
        st.latex(r"\tau = \frac{L_e}{L_r}")
        st.latex(rf"\tau = {datos_fila['Tortuosidad Areal (τ)']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Cinemática del Fluido (Velocidades)")
    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        st.markdown("**Velocidad Real del Polímero**")
        st.latex(r"v_{Darcy} = \frac{q}{A_{transversal}} \quad ; \quad v_{int} = \frac{v_{Darcy}}{\phi_{abs}} \quad ; \quad v_{real} = v_{int} \cdot \tau")
        st.latex(rf"v_{{real}} = {datos_fila['Velocidad Real (cm/s)']:.6f} \text{{ cm/s}}")
    with col_v2:
        st.markdown(r"""
        **Donde:**
        *   $q$: Caudal de inyección ($\text{cm}^3\text{/s}$).
        *   $A_{transversal}$: Área transversal ($ancho \times espesor$).
        *   $\phi_{abs}$: Porosidad absoluta (fracción).
        *   $\tau$: Tortuosidad areal (adimensional).
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Propiedades Petrofísicas Modificadas")
    col_k1, col_k2 = st.columns([1.5, 1])
    with col_k1:
        st.markdown("**Permeabilidad Estimada - Modelo de Kozeny-Carman Modificado**")
        st.latex(r"S_{vp} = \frac{2}{h} + \frac{4 \cdot (1 - \phi_{abs})}{\phi_{abs} \cdot D_p}")
        st.latex(r"k = \frac{\phi_{abs}}{2 \cdot \tau \cdot S_{vp}^2}")
        st.latex(rf"k = {datos_fila['Permeabilidad Mod. (mD)']:.2f} \text{{ mD}}")
    with col_k2:
        st.markdown(r"""
        **Donde:**
        *   $S_{vp}$: Área superficial específica ($\text{cm}^{-1}$).
        *   $h$: Espesor del micromodelo ($\text{cm}$).
        *   $\phi_{abs}$: Porosidad absoluta (fracción).
        *   $D_p$: Diámetro del grano cilíndrico ($\text{cm}$).
        *   $\tau$: Tortuosidad areal (adimensional).
        """)

    # --- 7. FUNDAMENTO MATEMÁTICO DE LAS CURVAS DINÁMICAS ---
    st.markdown("---")
    st.subheader("🔹 Cálculos Volumétricos Dinámicos y de Recobro")
    
    col_eq1, col_eq2 = st.columns(2)
    
    with col_eq1:
        st.markdown("**1. Curva de Producción Acumulada ($N_p$ vs $t$)**")
        st.write("Cálculo volumétrico en mililitros (ml) asumiendo $S_{oi} = 1$:")
        st.latex(r"N_p(t) = A_B(t) \cdot h \cdot \phi_{abs} \cdot S_{oi}")
        st.markdown(r"""
        **Donde:**
        *   $A_B(t)$: Área barrida en el tiempo $t$ ($\text{cm}^2$).
        *   $h$: Espesor del modelo ($\text{cm}$).
        *   $\phi_{abs}$: Porosidad absoluta (fracción).
        """)
        
    with col_eq2:
        st.markdown("**2. Comportamiento de Inyección ($N_p$ vs VPI)**")
        st.write("Volúmenes Porosos Inyectados calculados con base en el caudal volumétrico:")
        st.latex(r"VPI(t) = \frac{V_{iny}(t)}{V_p} = \frac{q \cdot t}{V_p}")
        st.markdown(r"""
        **Donde:**
        *   $q$: Caudal de inyección ($\text{ml/min}$).
        *   $t$: Tiempo transcurrido ($\text{min}$).
        *   $V_p$: Volumen poroso total ($\text{ml}$).
        """)

    # --- 8. GRÁFICAS DE COMPORTAMIENTO DINÁMICO (REALISTA) ---
    st.markdown("---")
    st.subheader(f"📈 Comportamiento Dinámico Estimado al Breakthrough: {archivo_seleccionado}")
    st.write("Las gráficas utilizan intervalos discretos de 10 minutos e incorporan un factor de atenuación para simular la pérdida de eficiencia por digitación viscosa.")
    
    Np_final = datos_fila['Np al BT (ml)']
    VPI_final = datos_fila['VPI al BT']
    
    tiempos = np.arange(0, int(t_bt) + 10, 10)
    
    np_array = Np_final * ((tiempos / t_bt) ** 0.85)
    vpi_array = VPI_final * (tiempos / t_bt)
    
    v_iny_ml_array = datos_fila['Caudal (ml/min)'] * tiempos
    
    datos_grafica = pd.DataFrame({
        "Tiempo (min)": tiempos,
        "Np (ml)": np_array,
        "VPI": vpi_array,
        "Volumen Inyectado (ml)": v_iny_ml_array
    })
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Curva de Producción Acumulada ($N_p$ vs $t$)**")
        st.line_chart(datos_grafica, x="Tiempo (min)", y="Np (ml)", color="#2ECC71")
        
    with col_g2:
        st.markdown("**Comportamiento de Inyección ($N_p$ vs VPI)**")
        st.line_chart(datos_grafica, x="VPI", y="Np (ml)", color="#3498DB")
        
    with st.expander("Ver Tabla de Datos de Producción Estimada"):
        st.dataframe(datos_grafica.style.format({
            "Tiempo (min)": "{:.0f}",
            "Np (ml)": "{:.2f}",
            "VPI": "{:.2f}",
            "Volumen Inyectado (ml)": "{:.2f}"
        }), use_container_width=True)
else:
    st.info("Sube las imágenes a la carpeta 'micromodelos' para visualizar los resultados.")
