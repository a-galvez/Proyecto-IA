import numpy as np


# Función de activación
class ReLU:
    def forward(self, x):
        self.entrada = x
        return np.maximum(0, x)

    def backward(self, d_salida):
        derivada = d_salida.copy()
        derivada[self.entrada <= 0] = 0
        return derivada
