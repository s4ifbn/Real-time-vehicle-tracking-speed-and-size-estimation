from ultralytics import YOLO

MODEL = "../models/yolov8s-seg.pt"      # x = Extra Large, l = large, m = medium, s = small, n = nano

# Load a model
model = YOLO(MODEL)  # load an official model
model = YOLO(MODEL)  # load an official model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps   # a list contains map50-95 