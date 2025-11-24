import os
import numpy as np
from PIL import Image

CLASES = ["Girasol", "Hortensias", "Margarita", "Rosa", "Tulipan"]


def cargar_dataset(data_dir, img_height, img_width):
    X, y = [], []
    label_map = {name: i for i, name in enumerate(CLASES)}
    for cls in CLASES:
        carpeta = os.path.join(data_dir, cls)
        for fname in os.listdir(carpeta):
            fpath = os.path.join(carpeta, fname)
            try:
                img = Image.open(fpath).convert("RGB").resize((img_width, img_height))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                X.append(arr)
                y.append(label_map[cls])
            except:
                pass
        print(f"Cargadas {len(os.listdir(carpeta))} imágenes de la clase '{cls}'")
    return np.array(X), np.array(y)


def dividir_dataset(X, y, val_ratio=0.15):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    cut = int(len(X) * (1 - val_ratio))
    print(f"Dividido en {cut} para entrenamiento y {len(X) - cut} para validación")
    return X[:cut], y[:cut], X[cut:], y[cut:]
