import cv2
import numpy as np

# Ruta al modelo preentrenado MobileNet-SSD
prototxt_path = "C:/Users/vicdc/Desktop/GAMEZ IA 3ER PARCIAL/PROYECTO IA/deploy.prototxt"
model_path = "C:/Users/vicdc/Desktop/GAMEZ IA 3ER PARCIAL/PROYECTO IA/mobilenet_iter_73000.caffemodel"

# Cargar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# ...existing code...
reference_image = cv2.imread(r"C:\Users\vicdc\Desktop\GAMEZ IA 3ER PARCIAL\PROYECTO IA\samsungz4.jfif.jpg")

if reference_image is None:
    print("No se pudo cargar la imagen de referencia.")
    exit()
# ...existing code...
# Preprocesar la imagen de referencia para obtener sus dimensiones
ref_h, ref_w = reference_image.shape[:2]
ref_aspect_ratio = ref_w / ref_h  # Relación de aspecto de referencia (ancho/alto)

# Clases de interés para la detección
INTERESTED_CLASSES = {
    "person": "Persona",
    "bottle": "Botella",
    "cell phone": "Celular",
    "shoe": "Tenis"
}

# Crear una lista de clases para filtrar
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor", "cell phone"
]

# Asignar un color específico a cada clase de interés
COLORS = {
    "person": (255, 0, 0),        # Rojo
    "bottle": (0, 255, 0),       # Verde
    "cell phone": (0, 0, 255),    # Azul
    "shoe": (255, 255, 0)        # Amarillo
}

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preparar el blob para la red neuronal
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Pasar el blob a la red
    net.setInput(blob)
    detections = net.forward()

    # Procesar las detecciones
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filtrar por una confianza mínima más alta
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])  # Índice de la clase detectada
            label = CLASSES[idx]

            # Solo procesar las clases de interés
            if label in INTERESTED_CLASSES:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Comparar tamaño relativo para determinar si es un celular
                box_width = endX - startX
                box_height = endY - startY
                aspect_ratio = box_width / box_height if box_height != 0 else 0

                # Comparación con la relación de aspecto de referencia de la imagen de un celular
                if label == "cell phone" and abs(aspect_ratio - ref_aspect_ratio) < 0.2:
                    # Probablemente es un celular
                    color = COLORS[label]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    text = f"{INTERESTED_CLASSES[label]}: {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif label != "cell phone":
                    # Dibujar otras clases directamente
                    color = COLORS[label]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    text = f"{INTERESTED_CLASSES[label]}: {confidence:.2f}"
                    cv2.putText(frame, text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar la ventana con la detección en tiempo real
    cv2.imshow("Detección de objetos", frame)

    # Salir del bucle con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
