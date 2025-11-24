import numpy as np
import math


class Dense:
    def __init__(self, tamanio_entrada, tamanio_salida):
        # Inicialización He porque es mejor para ReLU
        std = np.sqrt(2.0 / tamanio_entrada)
        self.pesos = np.random.normal(0, std, (tamanio_salida, tamanio_entrada))
        self.sesgos = np.ones((tamanio_salida,)) * 0.1

        print(f"  Dense: {tamanio_entrada} → {tamanio_salida}, std={std:.6f}")

    def forward(self, x):
        self.entrada = x
        return self.pesos.dot(x) + self.sesgos

    def backward(self, d_salida, tasa_aprendizaje):
        d_pesos = np.outer(d_salida, self.entrada)
        d_sesgos = d_salida
        d_entrada = self.pesos.T.dot(d_salida)

        self.pesos -= tasa_aprendizaje * d_pesos
        self.sesgos -= tasa_aprendizaje * d_sesgos
        return d_entrada
