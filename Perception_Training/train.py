from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data="data/dataset.yaml", imgsz=640, batch=8, epochs=5, workers=1, device=0)