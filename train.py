from ultralytics import YOLO

model = YOLO("yolov8l.pt")

if __name__ == '__main__':
    model.train(data="VisDrone.yaml", imgsz=480, epochs=100, workers=6)