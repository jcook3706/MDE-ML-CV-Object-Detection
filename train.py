from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="VisDrone.yaml", imgsz=640, epochs=100, workers=8)