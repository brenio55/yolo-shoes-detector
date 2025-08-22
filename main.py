# from ultralytics import YOLO

# model = YOLO("./my_model2/train/weights/best.pt")  # Load a pretrained YOLOv8 model

# # model.train(data="data.yaml", epochs=3)  # Train the model on the COCO128 dataset for 3 epochs

# results = model.predict(source=0, show=True, conf=0.1, stream=True)  # Predict on an image and display results

# for r in results: #conta os resultados pelo length do objeto
#     count = len(r.boxes)
#     print(f"Objetos detectados: {count}")


import cv2
import numpy as np
from ultralytics import YOLO

# Carregue seu modelo treinado
model = YOLO("./my_model2/train/weights/best.pt")

# Inicie a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

# Verifique se a webcam abriu corretamente
if not cap.isOpened():
    raise IOError("Não foi possível abrir a webcam")

# Obtenha as dimensões do frame
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Dimensões do frame: {width}x{height}")

# Defina os pontos da sua Região de Interesse (ROI)
# Neste exemplo, a ROI é a metade inferior da tela
roi_points = np.array([
    [0, 130],             # Ponto superior esquerdo
    [width, 180],         # Ponto superior direito
    [width, height],    # Ponto inferior direito
    [0, height]         # Ponto inferior esquerdo
], np.int32)

while True:
    # Leia um frame da webcam
    success, frame = cap.read()
    if not success:
        break

    # 1. Definir a ROI
    # Crie uma máscara preta com as mesmas dimensões do frame
    mask = np.zeros_like(frame)
    # Preencha o polígono da ROI com branco na máscara
    cv2.fillPoly(mask, [roi_points], (255, 255, 255))
    # Aplique a máscara ao frame. A detecção só ocorrerá na área da ROI.
    roi_frame = cv2.bitwise_and(frame, mask)

    # 2. Rastrear objetos com IDs na ROI
    # Use model.track() para obter detecções com IDs de rastreamento
    # O argumento 'persist=True' mantém os IDs entre os frames
    results = model.track(source=roi_frame, persist=True, verbose=False, conf=0.1)

    # Obtenha o frame com as anotações (caixas, IDs, etc.). Este frame tem a área fora da ROI em preto.
    annotated_roi = results[0].plot()

    # Combine o frame original com a ROI anotada para um efeito "transparente"
    # Onde a máscara é branca (dentro da ROI), usamos os pixels da ROI anotada.
    # Onde a máscara é preta (fora da ROI), usamos os pixels do frame original.
    annotated_frame = np.where(mask==(255, 255, 255), annotated_roi, frame)

    # Desenhe o polígono da ROI no frame final para visualização
    cv2.polylines(annotated_frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # 3. Contar objetos detectados
    # Verifique se há caixas de detecção e obtenha o número de objetos
    if results[0].boxes.id is not None:
        count = len(results[0].boxes)
        print(f"Objetos detectados: {count}")
    else:
        count = 0
        print("Nenhum objeto detectado.")

    # Adicione o texto da contagem no frame
    cv2.putText(annotated_frame, f"Contagem: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Mostre o frame resultante
    cv2.imshow("YOLOv8 Rastreamento com ROI", annotated_frame)

    # Saia do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()