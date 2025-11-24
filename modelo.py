import numpy as np
from Data.dataset import CLASES, cargar_dataset, dividir_dataset
from Red.cnn import CNN
from Utils.entrenamiento import entrenar, exactitud_por_clase
from Utils.guardar_cargar_modelo import guardar_modelo

DATA_DIR = "Datos/Entrenamiento"
IMG_HEIGHT = 128
IMG_WIDTH = 128
UMBRAL = 0.55

X, y = cargar_dataset(DATA_DIR, IMG_HEIGHT, IMG_WIDTH)
X_train, y_train, X_val, y_val = dividir_dataset(X, y)

red_neuronal = CNN(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)

entrenar(red_neuronal, X_train, y_train, tasa_aprendizaje=0.005, epocas=15)

exactitud_por_clase(red_neuronal, X_val, y_val)

guardar_modelo(red_neuronal, "modelo_flores.pkl")
