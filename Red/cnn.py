import numpy as np
from Capas import conv, dense, flatten, maxpool, relu, softmax
from Data.dataset import CLASES


class CNN:
    def __init__(self, img_height, img_width):
        # CAPA CONVOLUCIONAL 1
        # 16 filtros de tamaño 3x3
        self.conv1 = conv.Conv2D(16, 3)
        self.relu1 = relu.ReLU()
        self.pool1 = maxpool.MaxPool2(2, 2)

        # CAPA CONVOLUCIONAL 2
        # 32 filtros 3x3
        self.conv2 = conv.Conv2D(32, 3)
        self.relu2 = relu.ReLU()
        self.pool2 = maxpool.MaxPool2(2, 2)

        # Dimensión para 128x128 con 2 capas
        h_final = 30
        w_final = 30
        dimension_final = 32 * h_final * w_final  # 32 filtros de conv2
        print(f"Dimensión final: {dimension_final}")

        # CAPA DE APLANADO
        self.flatten = flatten.Flatten()

        # CAPA TOTALMENTE CONECTADA
        self.fc1 = dense.Dense(dimension_final, 256)
        self.relu3 = relu.ReLU()

        # CAPA DE SALIDA
        self.fc2 = dense.Dense(256, len(CLASES))
        self.softmax = softmax.Softmax()

    def forward(self, x):
        # Pasada hacia adelante a través de TODAS LAS CAPAS
        salida = self.conv1.forward(x)
        salida = self.relu1.forward(salida)
        salida = self.pool1.forward(salida)

        salida = self.conv2.forward(salida)
        salida = self.relu2.forward(salida)
        salida = self.pool2.forward(salida)

        salida = self.flatten.forward(salida)
        salida = self.fc1.forward(salida)
        salida = self.relu3.forward(salida)
        logits = self.fc2.forward(salida)
        probs = self.softmax.forward(logits)
        return probs

    def backward(self, etiqueta, tasa_aprendizaje):
        # Backpropagation de la capa de salida hacia atrás
        derivada = self.softmax.backward(etiqueta)
        derivada = self.fc2.backward(derivada, tasa_aprendizaje)
        derivada = self.relu3.backward(derivada)
        derivada = self.fc1.backward(derivada, tasa_aprendizaje)
        derivada = self.flatten.backward(derivada)

        derivada = self.pool2.backward(derivada)
        derivada = self.relu2.backward(derivada)
        derivada = self.conv2.backward(derivada, tasa_aprendizaje)
        derivada = self.pool1.backward(derivada)
        derivada = self.relu1.backward(derivada)
        self.conv1.backward(derivada, tasa_aprendizaje)
