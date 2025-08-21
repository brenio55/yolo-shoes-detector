from ultralytics import YOLO

model = YOLO("./my_model2/train/weights/best.pt")  # Load a pretrained YOLOv8 model

# model.train(data="data.yaml", epochs=3)  # Train the model on the COCO128 dataset for 3 epochs

results = model.predict(source=0, show=True, conf=0.1, stream=True)  # Predict on an image and display results

for r in results: #conta os resultados pelo length do objeto
    count = len(r.boxes)
    print(f"Objetos detectados: {count}")