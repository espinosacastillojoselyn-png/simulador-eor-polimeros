import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
from skimage.morphology import skeletonize
from skimage import measure
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
#  1. CARGA DE IMAGEN Y LECTURA DE DATOS 
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

# 2. INGRESO DE PARÁMETROS
st.sidebar.header("📝 2. Parámetros del Experimento")
st.sidebar.text_input("Polímero Detectado", value=tipo_polimero, disabled=True)
caudal = st.sidebar.number_input("Caudal de Inyección (ml/min)", value=val_q, format="%.4f")
concentracion = st.sidebar.number_input("Concentración (ppm)", value=val_ppm)

st.sidebar.markdown("---")
st.sidebar.subheader("Datos Físicos del Modelo")
ancho_mm = st.sidebar.number_input("Ancho del Micromodelo (mm)", value=5.00)
ancho= ancho_mm / 10.0 #Conversión del valor de espesor mm a cm para cálculo de Área
espesor_mm = st.sidebar.number_input("Espesor del Micromodelo (mm)", value=0.08, format="%.3f")
espesor = espesor_mm / 10.0 #Conversión del valor de espesor mm a cm para cálculo de Área
Dp_cm= st.sidebar.number_input("Tamaño del Grano (mm)", value=0.03, format="%.3f")
Dp_cm = 0.03 / 10.0  #Tamaño de grano

tipo_porosidad = st.sidebar.radio("¿Cómo ingresar la porosidad?", 
                                  ("Ingreso Manual", "Calcular ópticamente desde la imagen"))

if tipo_porosidad == "Ingreso Manual":
    porosidad = st.sidebar.number_input("Porosidad Absoluta (fracción)", min_value=0.01, max_value=1.0, value=0.39)
else:
    st.sidebar.success("La porosidad se calculará automáticamente con el Método de Otsu.")
    porosidad = None 
    pixeles_vacios = None

# 3. PROCESAMIENTO RBG Y CÁLCULOS MATEMÁTICOS
if archivo_subido is not None:
    # Decodificar imagen subida
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    file_bytes = np.asarray(bytearray(archivo_subido.read()), dtype=np.uint8)
    
    # --- AQUÍ EMPIEZA EL REEMPLAZO DE TU ANTIGUA LÍNEA 75 ---
    img_original = cv2.imdecode(file_bytes, 1)

    # A. Balance de Blancos: Algoritmo Gray World
    B, G, R = cv2.split(img_original)
    mB, mG, mR = np.mean(B), np.mean(G), np.mean(R)
    mean_gray = (mB + mG + mR) / 3.0
    
    B_gw = np.clip((B * (mean_gray / mB)), 0, 255).astype(np.uint8)
    G_gw = np.clip((G * (mean_gray / mG)), 0, 255).astype(np.uint8)
    R_gw = np.clip((R * (mean_gray / mR)), 0, 255).astype(np.uint8)
    img_gw = cv2.merge([B_gw, G_gw, R_gw])

    # B. Corrección de Iluminación: Pseudo Flat-Field Correction
    fondo_iluminacion = cv2.GaussianBlur(img_gw, (151, 151), 0)
    img_flat = img_gw.astype(np.float32) / (fondo_iluminacion.astype(np.float32) + 1e-5) 
    img_flat = cv2.normalize(img_flat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # C. Resolución y formato (Usando una resolución alta para no perder detalles de los poros)
    img = cv2.resize(img_flat, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    pixeles_totales = img.shape[0] * img.shape[1] #Área total del micromodelo (toda la imagen)

    #Dimensionalidad de píxeles
    # pixeles_totales volumen bulk
    #img.shape [0] alto total de la imagen en píxeles.
    # img.shape [1] ancho total de la imagen en píxeles.
    
    #  Cálculo de Porosidad Dinámica (Otsu) 
    if porosidad is None or tipo_porosidad == "Calcular ópticamente desde la imagen":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        umbral_optimo, mascara_poros = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        pixeles_vacios = np.sum(mascara_poros == 255)
        porosidad = pixeles_vacios / pixeles_totales #Ecuación de Porosidad Efectiva
        
        if porosidad == 0:
            porosidad = 0.001
            
        st.sidebar.info(f"Umbral automático (Otsu) calculado: {umbral_optimo:.0f}")
    else:
        # Si es manual, estimamos los píxeles vacíos teóricos del espacio poroso
        pixeles_vacios = int(pixeles_totales * porosidad)

    # Aislamiento del Polímero y Esqueletización 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    kernel = np.ones((5,5), np.uint8)
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    bool_mask = mask_limpia > 0
    esqueleto = skeletonize(bool_mask) 
    
    # Cálculos Geométricos y Cinemáticos
    longitud_camino_pixeles = np.sum(esqueleto) #Le
    longitud_recta_pixeles = img.shape[1] #Lr
    tortuosidad = max(1.0, longitud_camino_pixeles / longitud_recta_pixeles)

    #Cálculo de Velocidad
    q_cm3_s = caudal / 60.0 #Conversión de ml/min a cm^3/s
    area_transversal_cm2 = ancho * espesor
    v_darcy = q_cm3_s / area_transversal_cm2 #Velocidad de Darcy
    v_intersticial = v_darcy / porosidad #Velocidad Intersticial 
    velocidad_real = v_intersticial * tortuosidad 

    #  NUEVOS CÁLCULOS: ÁREA REAL BARRIDA Y EFICIENCIA 
    pixeles_polimero = np.sum(mask_limpia == 255)
    fraccion_modelo_invadida = pixeles_polimero / pixeles_totales #Cálculo de Eficiencia Areal
    longitud_calculada = ancho * (img.shape[1] / img.shape[0])
    area_total_vista_superior = ancho * longitud_calculada
    area_barrida_cm2 = fraccion_modelo_invadida * area_total_vista_superior    
    
    # Eficiencia de barrido areal (EA) respecto al espacio poroso disponible
    if pixeles_vacios and pixeles_vacios > 0:
        eficiencia_barrido = min(1.0, pixeles_polimero / pixeles_vacios)
    else:
        eficiencia_barrido = 0.0
    #Visualización en Columnas
    st.markdown("---")
    st.subheader(f"🔬 Visualización: Inyección de {tipo_polimero}")
    
    col_img1, col_img2, col_img3 = st.columns(3)
    
    with col_img1:
        st.markdown("**Tipo de Polímero:**")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col_img2:
        st.markdown("**Binarización**")
        st.image(mask_limpia, use_container_width=True, clamp=True)
        
    with col_img3:
        st.markdown("**Esqueleto (Red de Canales)**")
        # Preparación del esqueleto visual
        esqueleto_color = np.zeros((esqueleto.shape[0], esqueleto.shape[1], 3), dtype=np.uint8)
        esqueleto_color[esqueleto] = [255, 255, 0] 
        kernel_visual = np.ones((3,3), np.uint8)
        esqueleto_grueso = cv2.dilate(esqueleto_color, kernel_visual, iterations=1)
        st.image(esqueleto_grueso, use_container_width=True, clamp=True)

    # --- 4. PANEL DE RESULTADOS Y VISUALIZACIÓN ---
    st.markdown("---")
    st.subheader("📊 Resultados del Análisis")

    # Fila 1: Parámetros Hidrodinámicos de los Canales
    col1, col2, col3, col4 = st.columns(4)

    with col1:
     st.metric("Fluido Inyectado", f"{tipo_polimero} {concentracion} ppm")

    with col2:
     st.metric("Porosidad", f"{porosidad:.2%}")

    with col3:
     st.metric("Tortuosidad Areal (τ)", f"{tortuosidad:.4f}")
     # Ecuación de la tortuosidad (Longitud efectiva / Longitud recta)
     st.latex(r"\tau = \frac{L_e}{L_r}")

    with col4:
     st.metric("Velocidad del Polímero Inyectado", f"{velocidad_real:.6f} cm/s")
     # Ecuación de la velocidad real (Velocidad intersticial * tortuosidad)
     st.latex(r"v_{real} = \frac{q}{A_{Transversal} \cdot \phi} \cdot \tau")

    # Fila 2: Cuantificación de Áreas Sólidas y Eficiencia de Barrido
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    # Fila 3: Propiedades Microscópicas (Trabajo Adicional)
    st.markdown("<br>", unsafe_allow_html=True)
    col_k, col_vacia1, col_vacia2 = st.columns(3)

    # 1. Preparación de la variable matemática
    # Convertimos el tamaño de grano de mm a cm (utilizando el dato de 0.03 mm de tu tutor)
    Dp_cm = 0.03 / 10.0  

    # 2. Cálculo de Permeabilidad (Carman-Kozeny)
    if porosidad > 0 and tortuosidad > 0:
     # Cálculo en cm²
      k_cm2 = (porosidad**3 * Dp_cm**2) / (72 * tortuosidad * (1 - porosidad)**2)
     # Conversión de cm² a miliDarcys (mD) -> Factor: 1.013e11
      permeabilidad_mD = k_cm2 * 1.013e11 
    else:
      permeabilidad_mD = 0.0

    # 3. Impresión en la Interfaz con LaTeX
    with col_k:
     st.metric("Permeabilidad Estimada (k)", f"{permeabilidad_mD:.2f} mD")
     st.latex(r"k = \frac{\phi^3 \cdot D_p^2}{72 \cdot \tau \cdot (1-\phi)^2}")
    

    with col_a:
     st.metric("Área Total (Vista Superior)", f"{area_total_vista_superior:.2f} cm²")
     # Ecuación del área de la fotografía
     st.latex(r"A_{T} = ancho \cdot Longitud")

    with col_b:
     st.metric("Área Real Barrida", f"{area_barrida_cm2:.4f} cm²")
     # Ecuación de los píxeles convertidos a área
     st.latex(r"A_{B} = A_{T} \cdot \left(\frac{Pixeles_{polímero}}{Pixeles_{Totales}}\right)")

    with col_c:
     st.metric("Eficiencia de Barrido Areal (EA)", f"{eficiencia_barrido:.2%}")
     # Ecuación de la eficiencia de barrido
     st.latex(r"E_A = \frac{A_{B}}{A_{T}} \times 100")
   
   # --- 5. INTERPRETACIÓN TÉCNICA DINÁMICA ---
st.markdown("---")
st.subheader(f"💡 Resultados del {tipo_polimero}")

st.info(f""""
**Análisis de la inyección de {tipo_polimero} a {concentracion} ppm:**

* **Comportamiento Areal y Eficiencia:** El valor de tortuosidad de **{tortuosidad:.2f}** indica el grado de ramificación de la red de canales. Al contrastar distintos fluidos, este parámetro revela que el **{tipo_polimero}** inyectado está logrando una **Eficiencia de Barrido Areal ($E_A$) del {eficiencia_barrido:.2%}**, ocupando un área neta de **{area_barrida_cm2:.4f} cm²** dentro del medio poroso. Un valor de tortuosidad alto sugiere una dispersión más amplia y una mejor mitigación de la canalización.

* **Cinemática del Frente:** La velocidad real en el canal preferencial es de **{velocidad_real:.6f} cm/s**. Controlar esta velocidad intersticial es vital para dar tiempo a que los mecanismos físico-químicos del **{tipo_polimero}** actúen sobre el crudo residual en los poros, sin exceder el gradiente de fractura de la matriz rocosa.
""")
# --- 6. RESUMEN TABULAR Y EXPORTACIÓN ---
st.markdown("### 📋 Resumen de Datos Obtenidos")

# Organizamos los datos asegurando que ambas listas tengan exactamente 10 elementos
datos_resumen = {
    "Parámetro": [
        "Fluido Inyectado",
        "Concentración (ppm)",
        "Porosidad Absoluta (%)",
        "Tamaño de Grano Estimado (mm)",
        "Tortuosidad Areal (τ)",
        "Velocidad del Polímero Inyectado (cm/s)",
        "Área Total del Modelo (cm²)",
        "Área Real Barrida (cm²)",
        "Eficiencia de Barrido Areal (EA %)",
        "Permeabilidad Estimada (mD)"
    ],
    "Valor": [
        f"{tipo_polimero}",
        f"{concentracion}",
        f"{porosidad * 100:.2f}",
        "0.03",
        f"{tortuosidad:.4f}",
        f"{velocidad_real:.6f}",
        f"{area_total_vista_superior:.4f}",
        f"{area_barrida_cm2:.4f}",
        f"{eficiencia_barrido * 100:.2f}",
        f"{permeabilidad_mD:.2f}"
    ]
}

# Convertimos el diccionario en una tabla y la mostramos en pantalla
df_resumen = pd.DataFrame(datos_resumen)
st.table(df_resumen)

st.markdown("""
<style>
/* Eliminar bordes externos de la tabla */
[data-testid="stTable"] table {
    border: none !important;
    border-collapse: collapse !important;
}
/* Estilo del encabezado (Mayúsculas, negrita y solo línea inferior) */
[data-testid="stTable"] th {
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    border-bottom: 2px solid #2C3E50 !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    color: #2C3E50 !important;
    padding-bottom: 12px !important;
}
/* Estilo de las filas (Sin bordes laterales, solo línea divisoria suave) */
[data-testid="stTable"] td {
    border-top: none !important;
    border-left: none !important;
    border-right: none !important;
    border-bottom: 1px solid #E0E0E0 !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""", unsafe_allow_html=True)



# PODEMOS EXPORTAR LOS RESULTADOS EN UN ARCHIVO EXCEL
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    df_resumen.to_excel(writer, index=False, sheet_name='Resultados_EOR')

st.download_button(
    label="📊 Descargar Resultados en Excel",
    data=buffer.getvalue(),
    file_name=f"Reporte_Micromodelo_{tipo_polimero}_{concentracion}ppm.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
