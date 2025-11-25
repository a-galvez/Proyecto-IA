import pickle


def guardar_modelo(modelo, ruta_archivo="modelo_entrenado.pkl"):
    """Guarda el modelo entrenado en un archivo"""
    datos_modelo = {
        "conv1_filtros": modelo.conv1.filtros,
        "conv1_sesgos": modelo.conv1.sesgos,
        "conv2_filtros": modelo.conv2.filtros,
        "conv2_sesgos": modelo.conv2.sesgos,
        "fc1_pesos": modelo.fc1.pesos,
        "fc1_sesgos": modelo.fc1.sesgos,
        "fc2_pesos": modelo.fc2.pesos,
        "fc2_sesgos": modelo.fc2.sesgos,
        "img_height": 128,
        "img_width": 128,
        "num_classes": 5,
    }

    with open(ruta_archivo, "wb") as f:
        pickle.dump(datos_modelo, f)

    print(f"Modelo guardado en: {ruta_archivo}")


def cargar_modelo(ruta_archivo="modelo_entrenado.pkl"):
    """Carga un modelo previamente guardado"""
    from Red.cnn import CNN

    with open(ruta_archivo, "rb") as f:
        datos_modelo = pickle.load(f)

    # Crear nueva instancia del modelo
    modelo = CNN(
        img_height=datos_modelo["img_height"],
        img_width=datos_modelo["img_width"],
    )

    # Cargar los pesos entrenados
    modelo.conv1.filtros = datos_modelo["conv1_filtros"]
    modelo.conv1.sesgos = datos_modelo["conv1_sesgos"]
    modelo.conv2.filtros = datos_modelo["conv2_filtros"]
    modelo.conv2.sesgos = datos_modelo["conv2_sesgos"]
    modelo.conv3.filtros = datos_modelo["conv3_filtros"]
    modelo.conv3.sesgos = datos_modelo["conv3_sesgos"]
    modelo.fc1.pesos = datos_modelo["fc1_pesos"]
    modelo.fc1.sesgos = datos_modelo["fc1_sesgos"]
    modelo.fc2.pesos = datos_modelo["fc2_pesos"]
    modelo.fc2.sesgos = datos_modelo["fc2_sesgos"]

    print(f"Modelo cargado desde: {ruta_archivo}")
    return modelo
