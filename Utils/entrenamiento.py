from Data.dataset import CLASES
import numpy as np

UMBRAL = 0.55


def entrenar(
    modelo, X_train, y_train, X_val=None, y_val=None, tasa_aprendizaje=0.01, epocas=15
):
    print(
        "Entrenando por ", epocas, "épocas con tasa de aprendizaje =", tasa_aprendizaje
    )
    for ep in range(1, epocas + 1):
        total_perdida = 0
        perdidas = []

        # Entrenamiento
        indices = np.random.permutation(len(X_train))

        for idx in indices:
            x = X_train[idx]
            y = y_train[idx]

            probs = modelo.forward(x)
            perdida = modelo.softmax.perdida(probs, y)
            total_perdida += perdida
            perdidas.append(perdida)

            modelo.backward(y, tasa_aprendizaje)

        perdida_media = total_perdida / len(X_train)

        # Validación con accuracy
        if X_val is not None:
            correctos = 0
            for x, real in zip(X_val, y_val):
                probs = modelo.forward(x)
                pred = np.argmax(probs)
                if pred == real and np.max(probs) >= UMBRAL:
                    correctos += 1

            exactitud = correctos / len(X_val)
            print(
                f"Epoca {ep} perdida = {perdida_media:.4f} exactitud = {exactitud*100:.2f}%"
            )

        else:
            print(f"Epoca {ep} perdida = {perdida_media:.4f}")


def exactitud_por_clase(modelo, X, y):
    total = {c: 0 for c in CLASES}
    correctos = {c: 0 for c in CLASES}

    for x, real in zip(X, y):
        probs = modelo.forward(x)
        pred = np.argmax(probs)

        clase_real = CLASES[real]
        total[clase_real] += 1

        if pred == real:
            correctos[clase_real] += 1

    print("\nExactitud por clase:")
    for c in CLASES:
        if total[c] > 0:
            exactitud = (correctos[c] / total[c]) * 100
            print(f"{c}: {exactitud:.2f}% ({correctos[c]}/{total[c]})")
