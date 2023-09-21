from ultralytics import YOLO

# import the base, untrained model
model = YOLO("yolov8l.pt")

# train it on the data described by ./VisDrone.yaml - need to download this dataset first
model.train(data="./VisDrone.yaml", imgsz=640, epochs=100, workers=8)
