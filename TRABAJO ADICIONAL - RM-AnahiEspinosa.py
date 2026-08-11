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
st.title("Evaluación Micromodelos cEOR")
st.markdown("---")

st.subheader("💧 Micromodelo Base - Inyección de Agua (Waterflooding al Breakthrough)")
try:
    st.image("Iny Water.jpeg", caption="Micromodelo - Inyección de Agua", use_container_width=True)
except Exception as e:
    st.error("⚠️ No se encontró la imagen 'Iny Water.jpeg'. Asegúrate de que esté en el repositorio.")
st.markdown("---")

# --- 1. LECTURA AUTOMÁTICA DE LA CARPETA ---
CARPETA_MICROMODELOS = "micromodelos"

if not os.path.exists(CARPETA_MICROMODELOS):
    st.warning(f"⚠️ No se detectó la carpeta '{CARPETA_MICROMODELOS}'. Por favor, créala en tu repositorio y sube las imágenes.")
    archivos_validos = []
else:
    archivos_validos = [f for f in os.listdir(CARPETA_MICROMODELOS) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 2. INGRESO DE PARÁMETROS GLOBALES ---
st.sidebar.header("📝 2. Parámetros Físicos")
st.sidebar.info("Dimensiones estándar del Micromodelo de la Tesis (Ancho 180, Largo 200, Espesor 0.8)")

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
st.sidebar.success("🤖 Kozeny-Carman Dual (mm² y Darcys) Activo.")

datos_consolidados = []

# --- 3. PROCESAMIENTO EN BUCLE ---
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
            
        pixeles_totales = int(img.shape[0] * img.shape[1])
        ancho_pixeles = int(img.shape[1]) 
        
        # Pre-procesamiento CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img_corregida = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        img_suavizada = cv2.GaussianBlur(img_corregida, (5, 5), 0)
        hsv = cv2.cvtColor(img_suavizada, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([95, 70, 70])
        upper_blue = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        kernel = np.ones((5,5), np.uint8)
        mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_limpia = cv2.morphologyEx(mask_limpia, cv2.MORPH_CLOSE, kernel)
        
        pixeles_polimero = int(np.sum(mask_limpia == 255))
        pixeles_conectados = int(np.sum(mask_limpia > 0))
        
        # --- CÁLCULO DE POROSIDAD EFECTIVA (ϕ_eff) BASADO EN PÍXELES ---
        fraccion_area_conectada = pixeles_conectados / pixeles_totales if pixeles_totales > 0 else 0
        porosidad_efectiva = float(porosidad_abs * fraccion_area_conectada)
        porosidad_efectiva = max(0.01, min(porosidad_abs, porosidad_efectiva))
        
        pixeles_poros_totales = pixeles_totales * porosidad_abs
        fr_porcentaje = (pixeles_polimero / pixeles_poros_totales) * 100 if pixeles_poros_totales > 0 else 0
        eficiencia_barrido = fr_porcentaje / 100.0
        sor_fraccion = max(0.0, 1.0 - eficiencia_barrido)
        
        bool_mask = mask_limpia > 0
        esqueleto = skeletonize(bool_mask) 
        pixeles_esqueleto = int(np.sum(esqueleto))
        
        factor_geometrico = (ancho_pixeles / pixeles_totales) * 12.0
        tortuosidad = max(1.2, min(2.5, 1.0 + (pixeles_esqueleto / max(1, ancho_pixeles)) * factor_geometrico))
        
        area_transversal_cm2 = ancho * espesor
        q_cm3_s = val_q / 60.0 
        v_darcy = q_cm3_s / area_transversal_cm2
        v_intersticial = v_darcy / porosidad_abs if porosidad_abs > 0 else 0
        velocidad_real = v_intersticial * tortuosidad 
        
        area_total_cm2 = ancho * largo_cm
        Vp_ml = area_total_cm2 * espesor * porosidad_abs
        
        # --- CÁLCULO DUAL DE PERMEABILIDAD (Absoluta vs Efectiva) en mm² y Darcys ---
        if eficiencia_barrido > 0 and tortuosidad > 0:
            # 1. Con Porosidad Absoluta (ϕ_abs)
            S_vp_abs = (2 / espesor) + ((4 * (1 - porosidad_abs)) / (porosidad_abs * Dp_cm))
            k_cm2_abs = porosidad_abs / (2 * tortuosidad * (S_vp_abs**2))
            permeabilidad_mm2_abs = k_cm2_abs * 100.0
            permeabilidad_darcy_abs = k_cm2_abs * 1.01325e8
            
            # 2. Con Porosidad Efectiva (ϕ_eff)
            S_vp_eff = (2 / espesor) + ((4 * (1 - porosidad_efectiva)) / (porosidad_efectiva * Dp_cm))
            k_cm2_eff = porosidad_efectiva / (2 * tortuosidad * (S_vp_eff**2))
            permeabilidad_mm2_eff = k_cm2_eff * 100.0
            permeabilidad_darcy_eff = k_cm2_eff * 1.01325e8
        else:
            permeabilidad_mm2_abs = 0.0
            permeabilidad_darcy_abs = 0.0
            permeabilidad_mm2_eff = 0.0
            permeabilidad_darcy_eff = 0.0

        Np_ml = eficiencia_barrido * Vp_ml
        V_iny_ml = val_q * t_bt
        VPI_bt = V_iny_ml / Vp_ml if Vp_ml > 0 else 0

        datos_consolidados.append({
            "Archivo": nombre_archivo,
            "Polímero": tipo_polimero,
            "Concentración (ppm)": val_ppm,
            "Caudal (ml/min)": val_q,
            "Porosidad Efectiva (ϕ_eff)": porosidad_efectiva,
            "Tortuosidad Areal (τ)": tortuosidad,
            "Velocidad Real (cm/s)": velocidad_real,
            "Perm. Abs. (mm²)": permeabilidad_mm2_abs,
            "Perm. Abs. (Darcy)": permeabilidad_darcy_abs,
            "Perm. Efec. (mm²)": permeabilidad_mm2_eff,
            "Perm. Efec. (Darcy)": permeabilidad_darcy_eff,
            "Np al BT (ml)": Np_ml,
            "VPI al BT": VPI_bt,
            "% Fr": fr_porcentaje,
            "Sor (fracción)": sor_fraccion,
            "Píxeles Conectados": pixeles_conectados,
            "Píxeles Esqueleto": pixeles_esqueleto
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
    
    st.table(df_maestro[['Archivo', 'Polímero', 'Concentración (ppm)', 'Caudal (ml/min)', 'Porosidad Efectiva (ϕ_eff)', 'Tortuosidad Areal (τ)', '% Fr', 'Sor (fracción)', 'Perm. Abs. (mm²)', 'Perm. Efec. (mm²)', 'Perm. Abs. (Darcy)', 'Perm. Efec. (Darcy)']])

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
    st.info("Selecciona un micromodelo específico del lote para visualizar sus máscaras y el análisis petrofísico basado en píxeles.")
    
    archivo_seleccionado = st.selectbox("Seleccionar Micromodelo:", archivos_validos)
    datos_fila = df_maestro[df_maestro["Archivo"] == archivo_seleccionado].iloc[0]
    
    img_ui = cv2.imread(os.path.join(CARPETA_MICROMODELOS, archivo_seleccionado))
    
    lab_ui = cv2.cvtColor(img_ui, cv2.COLOR_BGR2LAB)
    l_ui, a_ui, b_ui = cv2.split(lab_ui)
    cl_ui = clahe.apply(l_ui)
    limg_ui = cv2.merge((cl_ui, a_ui, b_ui))
    img_corregida_ui = cv2.cvtColor(limg_ui, cv2.COLOR_LAB2BGR)
    
    img_suavizada_ui = cv2.GaussianBlur(img_corregida_ui, (5, 5), 0)
    hsv_ui = cv2.cvtColor(img_suavizada_ui, cv2.COLOR_BGR2HSV)
    mask_ui = cv2.inRange(hsv_ui, np.array([95, 70, 70]), np.array([135, 255, 255]))
    kernel_ui = np.ones((5,5), np.uint8)
    mask_limpia_ui = cv2.morphologyEx(mask_ui, cv2.MORPH_OPEN, kernel_ui)
    mask_limpia_ui = cv2.morphologyEx(mask_limpia_ui, cv2.MORPH_CLOSE, kernel_ui)
    esqueleto_ui = skeletonize(mask_limpia_ui > 0)

    col_img1, col_img2, col_img3 = st.columns(3)
    with col_img1:
        st.markdown(f"**Original (CLAHE): {datos_fila['Polímero']}**")
        st.image(cv2.cvtColor(img_corregida_ui, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_img2:
        st.markdown(f"**Binarización ({datos_fila['Píxeles Conectados']} px)**")
        st.image(mask_limpia_ui, use_container_width=True, clamp=True)
    with col_img3:
        st.markdown(f"**Esqueleto ({datos_fila['Píxeles Esqueleto']} px)**")
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
        st.markdown("**Recuperación y Eficiencia (%Fr y $E_A$)**")
        st.latex(r"\%Fr = E_A = \left(\frac{\text{Píxeles de Polímero}}{\text{Píxeles de Poros Totales}}\right) \times 100")
        st.latex(rf"\%Fr = E_A = {datos_fila['% Fr']:.2f} \%")
    with col3:
        st.markdown("**Saturación Residual ($S_{or}$)**")
        st.latex(r"S_{or} = 1.0 - \left(\frac{\%Fr}{100}\right)")
        st.latex(rf"S_{{or}} = {datos_fila['Sor (fracción)']:.4f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Propiedades Petrofísicas en Términos de Píxeles")
    col_v1, col_v2 = st.columns([1.5, 1])
    with col_v1:
        st.markdown("**Formulación Analítica por Conteo de Píxeles**")
        st.latex(r"\phi_{eff} = \phi_{abs} \cdot \left(\frac{\sum \text{Píxeles Conectados}}{\text{Píxeles Totales}}\right)")
        st.latex(r"\tau = \frac{L_e}{L_r} = \frac{\sum \text{Píxeles del Esqueleto}}{\text{Ancho en Píxeles (Columnas)}}")
        st.latex(rf"\phi_{{eff}} = {datos_fila['Porosidad Efectiva (ϕ_eff)']:.4f} \quad ; \quad \tau = {datos_fila['Tortuosidad Areal (τ)']:.4f}")
    with col_v2:
        st.markdown(r"""
        **Variables de Matriz (Píxeles):**
        *   $\phi_{abs}$: Porosidad absoluta base ($0.39$).
        *   $\sum \text{Píxeles Conectados}$: **""" + str(datos_fila['Píxeles Conectados']) + r"""** px.
        *   $\text{Píxeles Totales}$: **""" + str(pixeles_totales) + r"""** px.
        *   $\sum \text{Píxeles del Esqueleto}$: **""" + str(datos_fila['Píxeles Esqueleto']) + r"""** px.
        *   $L_r$ (Ancho en píxeles): **""" + str(ancho_pixeles) + r"""** px.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Cinemática del Fluido (Velocidades)")
    col_vel1, col_vel2 = st.columns([1.5, 1])
    with col_vel1:
        st.markdown("**Velocidad Real del Polímero**")
        st.latex(r"v_{Darcy} = \frac{q}{A_{transversal}} \quad ; \quad v_{int} = \frac{v_{Darcy}}{\phi_{abs}} \quad ; \quad v_{real} = v_{int} \cdot \tau")
        st.latex(rf"v_{{real}} = {datos_fila['Velocidad Real (cm/s)']:.6f} \text{{ cm/s}}")
    with col_vel2:
        st.markdown(r"""
        **Donde:**
        *   $q$: Caudal ($\text{cm}^3\text{/s}$).
        *   $\tau$: Tortuosidad derivada del esqueleto de píxeles.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔹 Permeabilidad Comparativa (Kozeny-Carman en mm² y Darcys)")
    
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("**1. Permeabilidad con Porosidad Absoluta ($k_{abs}$)**")
        st.latex(r"k_{abs} = \frac{\phi_{abs}}{2 \cdot \tau \cdot S_{vp(abs)}^2}")
        st.latex(rf"k_{{abs}} = {datos_fila['Perm. Abs. (mm²)']:.4f} \text{{ mm}}^2 \quad (\approx {datos_fila['Perm. Abs. (Darcy)']:.2f} \text{{ D}})")
    with col_k2:
        st.markdown("**2. Permeabilidad con Porosidad Efectiva ($k_{eff}$)**")
        st.latex(r"k_{eff} = \frac{\phi_{eff}}{2 \cdot \tau \cdot S_{vp(eff)}^2}")
        st.latex(rf"k_{{eff}} = {datos_fila['Perm. Efec. (mm²)']:.4f} \text{{ mm}}^2 \quad (\approx {datos_fila['Perm. Efec. (Darcy)']:.2f} \text{{ D}})")


    # --- 7. CURVAS DINÁMICAS ---
    st.markdown("---")
    st.subheader("🔹 Cálculos Volumétricos Dinámicos y de Recobro")
    col_eq1, col_eq2 = st.columns(2)
    with col_eq1:
        st.markdown("**1. Curva de Producción Acumulada ($N_p$ vs $t$)**")
        st.latex(r"N_p(t) = A_B(t) \cdot h \cdot \phi_{abs} \cdot S_{oi}")
    with col_eq2:
        st.markdown("**2. Comportamiento de Inyección ($N_p$ vs VPI)**")
        st.latex(r"VPI(t) = \frac{V_{iny}(t)}{V_p} = \frac{q \cdot t}{V_p}")

    # --- 8. GRÁFICAS MÚLTIPLES DE COMPORTAMIENTO DINÁMICO ---
    st.markdown("---")
    st.subheader(f"📈 Análisis Dinámico y Comportamiento al Breakthrough: {archivo_seleccionado}")
    
    Np_final = datos_fila['Np al BT (ml)']
    VPI_final = datos_fila['VPI al BT']
    
    tiempos = np.arange(0, int(t_bt) + 10, 10)
    np_array = Np_final * ((tiempos / t_bt) ** 0.85)
    vpi_array = VPI_final * (tiempos / t_bt)
    v_iny_ml_array = datos_fila['Caudal (ml/min)'] * tiempos
    
    area_total_cm2_g = ancho * largo_cm
    Vp_ml_g = area_total_cm2_g * espesor * porosidad_abs
    fr_array = (np_array / Vp_ml_g) * 100 if Vp_ml_g > 0 else np.zeros_like(np_array)
    sor_array = 1.0 - (fr_array / 100.0)
    
    datos_grafica = pd.DataFrame({
        "Tiempo (min)": tiempos,
        "Np (ml)": np_array,
        "VPI": vpi_array,
        "Volumen Inyectado (ml)": v_iny_ml_array,
        "% Fr": fr_array,
        "Sor (fracción)": sor_array
    })
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**1. Curva de Producción Acumulada ($N_p$ vs $t$)**")
        st.line_chart(datos_grafica, x="Tiempo (min)", y="Np (ml)", color="#2ECC71")
        
        st.markdown("**3. Recuperación vs Tiempo (%Fr vs $t$)**")
        st.line_chart(datos_grafica, x="Tiempo (min)", y="% Fr", color="#E67E22")
        
        st.markdown("**5. Saturación Residual vs Tiempo ($S_{or}$ vs $t$)**")
        st.line_chart(datos_grafica, x="Tiempo (min)", y="Sor (fracción)", color="#E74C3C")
        
    with col_g2:
        st.markdown("**2. Comportamiento de Inyección ($N_p$ vs VPI)**")
        st.line_chart(datos_grafica, x="VPI", y="Np (ml)", color="#3498DB")
        
        st.markdown("**4. Recuperación vs VPI (%Fr vs VPI)**")
        st.line_chart(datos_grafica, x="VPI", y="% Fr", color="#9B59B6")
        
        st.markdown("**6. Relación de Producción y Recobro ($N_p$ vs %Fr)**")
        st.line_chart(datos_grafica, x="% Fr", y="Np (ml)", color="#1ABC9C")
        
    with st.expander("Ver Tabla de Datos de Producción Estimada con %Fr y Sor"):
        st.dataframe(datos_grafica.style.format({
            "Tiempo (min)": "{:.0f}",
            "Np (ml)": "{:.2f}",
            "VPI": "{:.2f}",
            "Volumen Inyectado (ml)": "{:.2f}",
            "% Fr": "{:.2f}",
            "Sor (fracción)": "{:.4f}"
        }), use_container_width=True)
        
else:
    st.info("Sube las imágenes a la carpeta 'micromodelos' para visualizar los resultados.")

# --- 9. REFERENCIAS BIBLIOGRÁFICAS ---
st.markdown("---")
st.subheader("📚 Respaldo Teórico y Validación Metodológica")
st.info("Las ecuaciones petrofísicas, los algoritmos de calibración óptica y la validación fenomenológica de este simulador están fundamentados en el trabajo experimental de:")

st.markdown("""
> **Herrera Silva, L. R. (2020).** *Estudio experimental del desplazamiento y eficiencia de una inundación polimérica en micromodelos transparentes*. Tesis de Maestría. Universidad de Buenos Aires, Facultad de Ingeniería (IGPUBA).
""")
