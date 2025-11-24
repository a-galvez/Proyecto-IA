class Flatten:
    def forward(self, x):
        self.forma_original = x.shape
        return x.reshape(-1)

    def backward(self, d_salida):
        return d_salida.reshape(self.forma_original)
