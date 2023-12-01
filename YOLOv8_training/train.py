from ultralytics import YOLO

model = YOLO("yolov8l.pt")

model.train(data="VisDrone.yaml", imgsz=640, epochs=100, workers=8)