### Trained Models

This directory contains a few trained YOLOv8 models.
- `yolov8m_trained.pt`: This is a medium size YOLO model trained on the VisDrone dataset. 
- `yolov8l_trained.pt`: This is a large size YOLO model trained on the VisDrone dataset. 
- `yolov8l_trained_on_tiled_640_set.pt`: This is a large size YOLO model trained on a tiled VisDrone dataset. The tiles are 640x640 pixels. This is used for our best overall model when combined with SAHI@640.
- `yolov8l_trained_on_robustness_set.pt`: This is a large size YOLO model trained on an augmented version of the  VisDrone dataset. This model is used for testing on suboptimal conditions.
