from ultralytics import YOLO



#detection
MODEL1 = "../models/yolov8n-seg.pt"
MODEL2 = "../models/yolov8s-seg.pt"
MODEL3 = "../models/yolov8m-seg.pt"
MODEL4 = "../models/yolov8l-seg.pt"
MODEL5 = "../models/yolov8x-seg.pt"

model1 = YOLO(MODEL1)
model1.fuse()

model2 = YOLO(MODEL2)
model2.fuse()

model3 = YOLO(MODEL3)
model3.fuse()

model4 = YOLO(MODEL4)
model4.fuse()

model5 = YOLO(MODEL5)
model5.fuse()