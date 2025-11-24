import numpy as np


class MaxPool2:
    def __init__(self, tamanio=2, paso=2):
        self.tamanio = tamanio
        self.paso = paso
        self.entrada = None
        self.arg_max = None

    def forward(self, x):
        # x: (H, W, C)
        self.entrada = x
        H, W, C = x.shape
        S = self.paso
        K = self.tamanio
        salida_altura = (H - K) // S + 1
        salida_anchura = (W - K) // S + 1
        salida = np.zeros((salida_altura, salida_anchura, C))
        self.arg_max = {}
        for c in range(C):
            for i in range(salida_altura):
                for j in range(salida_anchura):
                    parche = x[i * S : i * S + K, j * S : j * S + K, c]
                    salida[i, j, c] = np.max(parche)
                    # guardar índice del max para backprop
                    idx = np.argmax(parche)
                    self.arg_max[(c, i, j)] = idx
        return salida

    def backward(self, d_salida):
        H, W, C = self.entrada.shape
        S = self.paso
        K = self.tamanio
        salida_altura = (H - K) // S + 1
        salida_anchura = (W - K) // S + 1
        d_entrada = np.zeros_like(self.entrada)
        for c in range(C):
            for i in range(salida_altura):
                for j in range(salida_anchura):
                    idx = self.arg_max[(c, i, j)]
                    # convertir idx a coordenadas dentro del parche
                    r = idx // K
                    s = idx % K
                    d_entrada[i * S + r, j * S + s, c] += d_salida[i, j, c]
        return d_entrada
