from ultralytics import YOLO

model = YOLO("/home/ubuntu/Desktop/yolo/runs/detect/train/weights/best.pt")

model.predict(source="test.png", show=True, save=True)