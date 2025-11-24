import numpy as np


class Softmax:
    def forward(self, logits):
        # Exponentes estabilizados
        exponentes = np.exp(logits - np.max(logits))
        # Probabilidades finales
        self.probs = exponentes / np.sum(exponentes)
        return self.probs

    def perdida(self, probs, etiqueta):
        # Cálculo de la pérdida de entropía cruzada
        return -np.log(probs[etiqueta])

    def backward(self, etiqueta):
        # Gradiente de la pérdida con respecto a los logits
        gradiente = self.probs.copy()
        gradiente[etiqueta] -= 1
        return gradiente
