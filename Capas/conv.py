import numpy as np
import math


def imagenes_a_columnas(x, filtro_altura, filtro_anchura, paso=1):
    H, W, C = x.shape
    salida_altura = (H - filtro_altura) // paso + 1
    salida_anchura = (W - filtro_anchura) // paso + 1
    columnas = []
    for i in range(0, H - filtro_altura + 1, paso):
        for j in range(0, W - filtro_anchura + 1, paso):
            parche = x[i : i + filtro_altura, j : j + filtro_anchura, :].reshape(-1)
            columnas.append(parche)
    return np.array(columnas).T  # (F*F*C, salida_altura*salida_anchura)


class Conv2D:
    def __init__(self, numero_filtros, tamanio_filtro=3, paso=1, canales_entrada=None):
        self.numero_filtros = numero_filtros
        self.tamanio_filtro = tamanio_filtro
        self.paso = paso
        self.canales_entrada = canales_entrada  # puede ser None hasta el primer forward
        self.inicializado = False

    def _inicializar_filtros(self, canales_entrada):
        self.canales_entrada = canales_entrada
        limit = 1.0 / math.sqrt(
            self.tamanio_filtro * self.tamanio_filtro * canales_entrada
        )
        self.filtros = np.random.uniform(
            -limit,
            limit,
            (
                self.numero_filtros,
                self.tamanio_filtro,
                self.tamanio_filtro,
                canales_entrada,
            ),
        )
        self.sesgos = np.zeros((self.numero_filtros,))
        self.inicializado = True

    def forward(self, x):
        # Inicializar filtros al primer forward según canales reales
        if not self.inicializado:
            canales_entrada = x.shape[2]
            self._inicializar_filtros(canales_entrada)

        self.entrada = x
        F = self.tamanio_filtro

        self.columnas = imagenes_a_columnas(x, F, F, self.paso)
        filtros_planos = self.filtros.reshape(self.numero_filtros, -1)

        salida = filtros_planos.dot(self.columnas) + self.sesgos[:, None]

        salida_altura = int((x.shape[0] - F) // self.paso + 1)
        salida_anchura = int((x.shape[1] - F) // self.paso + 1)

        salida = salida.reshape(self.numero_filtros, salida_altura, salida_anchura)
        return np.transpose(salida, (1, 2, 0))

    def backward(self, d_out, lr):
        d_salida_t = np.transpose(d_out, (2, 0, 1)).reshape(self.numero_filtros, -1)
        filtros_planos = self.filtros.reshape(self.numero_filtros, -1)

        # Gradientes de los filtros
        d_filtros = d_salida_t.dot(self.columnas.T).reshape(self.filtros.shape)
        d_sesgos = d_salida_t.sum(axis=1)

        # Gradiente sobre la entrada
        d_columnas = filtros_planos.T.dot(d_salida_t)
        H, W, C = self.entrada.shape
        F = self.tamanio_filtro
        paso = self.paso

        d_entrada = np.zeros_like(self.entrada)
        k = 0
        for i in range(0, H - F + 1, paso):
            for j in range(0, W - F + 1, paso):
                patch = d_columnas[:, k].reshape(F, F, C)
                d_entrada[i : i + F, j : j + F, :] += patch
                k += 1

        # Actualizar pesos
        self.filtros -= lr * (d_filtros / d_salida_t.shape[1])
        self.sesgos -= lr * (d_sesgos / d_salida_t.shape[1])

        return d_entrada
