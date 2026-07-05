import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def analizar_micromodelo(ruta_imagen, caudal, ancho, espesor, porosidad):
    # 1. Cargar la imagen
    img = cv2.imread(ruta_imagen)
    # Convertir a espacio de color HSV para aislar el azul más fácil
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. Definir el rango de color AZUL (Ajusta estos valores según la iluminación de tus fotos)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Crear una máscara binaria (Blanco = Fluido, Negro = Matriz)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Limpiar un poco el ruido de la imagen (operaciones morfológicas)
    kernel = np.ones((5,5), np.uint8)
    mask_limpia = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Esqueletización (Reducir el camino a 1 píxel de grosor)
    # skimage requiere que la imagen sea booleana (0 y 1)
    bool_mask = mask_limpia > 0
    esqueleto = skeletonize(bool_mask)
    
    # 4. Calcular Tortuosidad
    # Longitud del camino = sumar todos los píxeles que son True en el esqueleto
    longitud_camino_pixeles = np.sum(esqueleto)
    
    # Longitud recta = el ancho de la imagen (asumiendo flujo de izquierda a derecha)
    longitud_recta_pixeles = img.shape[1] 
    
    tortuosidad = longitud_camino_pixeles / longitud_recta_pixeles
    
    # Asegurar que la tortuosidad no sea menor a 1 (físicamente imposible)
    tortuosidad = max(1.0, tortuosidad)
    
    # 5. Calcular Velocidad (Usando la lógica que vimos antes)
    q_cm3_s = caudal / 60.0
    area_cm2 = ancho * espesor
    v_darcy = q_cm3_s / area_cm2
    v_intersticial = v_darcy / porosidad
    velocidad_real = v_intersticial * tortuosidad
    
    # 6. Visualización para comprobar que el código "ve" bien
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title('Original')
    axes[1].imshow(mask_limpia, cmap='gray'); axes[1].set_title('Binarización (Fluido aislado)')
    axes[2].imshow(esqueleto, cmap='gray'); axes[2].set_title('Esqueleto (Camino preferencial)')
    plt.show()
    
    return tortuosidad, velocidad_real

# ==========================================
# PRUEBA TU CÓDIGO
# ==========================================
# Reemplaza 'tu_imagen.jpg' con el nombre de tu archivo
tort, vel = analizar_micromodelo('ñ.1.2-5.5.jpg', caudal=0.5, ancho=5.0, espesor=0.1, porosidad=0.40)
print(f"Tortuosidad Calculada: {tort:.2f}")
print(f"Velocidad Real: {vel:.4f} cm/s")