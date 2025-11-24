import numpy as np
from PIL import Image
from Utils.guardar_cargar_modelo import cargar_modelo

CLASES = ["Girasol", "Hortensias", "Margarita", "Rosa", "Tulipan"]
RUTA_MODELO = "modelo_flores.pkl"
UMBRAL = 0.55

print("Cargando modelo desde", RUTA_MODELO)
modelo = cargar_modelo(RUTA_MODELO)

imagenes = [
    "Datos/Pruebas/daisy.jpg",  # Margarita
    "Datos/Pruebas/rose.jpg",  # Rosa
    "Datos/Pruebas/sunflower.jpg",  # Girasol
    "Datos/Pruebas/tulip.jpg",  # Tulipán
    "Datos/Pruebas/hydrangea.jpg",  # Hortensia
    "Datos/Pruebas/microphone.jpg",  # Ninguna
]

for i, img_path in enumerate(imagenes, 1):
    # Cargar y preprocesar
    img = Image.open(img_path).convert("RGB")
    img = img.resize((128, 128))
    x = np.array(img, dtype=np.float32) / 255.0

    # Predecir
    probs = modelo.forward(x)
    pred_idx = np.argmax(probs)
    confianza = probs[pred_idx]
    resultado = CLASES[pred_idx]

    if confianza < UMBRAL:
        resultado = "Ninguna"

    # Mostrar resultado
    print(f"Imagen {i}: Predicción = {resultado}, Confianza = {confianza:.1%}")
