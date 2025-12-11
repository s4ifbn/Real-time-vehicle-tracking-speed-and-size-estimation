# Multi-Moving Vehicle Classification and Counting based YOLOv8 DNN Model
# Author: Saif Bashar 2023
# CONDA VENV: test10
# Version: 11.0 Update 2023-08-04, 10:00 AM
# All essential packages are up-to-date

import time
import math
import datetime
import cv2 as cv
import numpy as np
import torch.backends.mps
import supervision as sv
from ultralytics import YOLO
from collections import deque           # double ended queue

# COLORS & FONTS & KEYS
BLACK = (0, 0, 0)
BLUE = (188, 108, 87)
GREEN = (0, 255, 0)
TEAL = (221, 207, 69)
RED = (121, 100, 233)
WHITE = (255, 255, 255)
YELLOW = (90, 217, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
ESC = 27                                # escape key ASCII code

# SYSTEM FLAGS
SAVE = False                            # enable saving output file

# SYSTEM OUTPUT
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.csv"                         # output log file

# MODEL PARAMETERS
CLASSES = [2, 3, 5, 7]                  # detection on classes (Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker
MODEL = "../models/yolov8l.pt"          # x = Extra Large, l = large, m = medium, s = small, n = nano
VIDEO = "../data/test1.mov"             # default test video
ms = 1                                  # waiting 1 ms between frames, set ms=0 for debugging

# DEVICE-AGNOSTIC CODE
DEVICE = "cpu"                          # CPU with No GPU Support
if torch.backends.mps.is_available():
    DEVICE = "mps"                      # for Apple MPS
elif torch.cuda.is_available():
    DEVICE = "cuda:0"                   # for NVIDIA GPU

print("SYSTEM RUNNING USING " + DEVICE + " Graphics")


# GET VIDEO INFO
def get_vid_info(video):
    info = sv.VideoInfo.from_video_path(video)
    return info.width, info.height, info.fps, info.total_frames


# SOURCE VIDEO INFORMATION
WIDTH, HEIGHT, FPS, TOTAL_FRAMES = get_vid_info(VIDEO)
print(f"SOURCE VIDEO: WIDTH= {WIDTH}, HEIGHT= {HEIGHT}, FPS={FPS}, FRAMES={TOTAL_FRAMES}")

if SAVE:
    OUT_FRAME = cv.VideoWriter(OUT, cv.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))


# INFO DISPLAY
LEFT_CORNER1 = (10, 30)                              # for car counting display
LEFT_CORNER2 = (10, 55)                              # for car counting display
LEFT_CORNER3 = (10, 75)                              # for car counting display
LEFT_CORNER4 = (10, 95)                              # for car counting display
RIGHT_CORNER = (WIDTH - 110, 30)                    # for fps display
CENTER_TOP = (250, 60)                              # for car speed display

# CALIBRATION PARAMETER
OFFSET = 10                                         # offset for each line
ROI_OFFSET = 120                                    # ROI offset (Move ROI up and down)
LEFT_SHIFT = 300                                    # ROI offset (Move ROI left)

HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

# ROI COORDINATES
LINE1_START = (0, (HALF_HEIGHT + ROI_OFFSET) - 100)
LINE1_END = (HALF_WIDTH + 120, (HALF_HEIGHT + ROI_OFFSET) - 100)

LINE2_START = (250, HALF_HEIGHT + ROI_OFFSET)
LINE2_END = (HALF_WIDTH + LEFT_SHIFT + 50, HALF_HEIGHT + ROI_OFFSET)

LINE3_START = (LEFT_SHIFT, (HALF_HEIGHT + ROI_OFFSET) + 100)
LINE3_END = (WIDTH, (HALF_HEIGHT + ROI_OFFSET) + 100)

# COUNTING & SPEED ESTIMATION & SIZE ESTIMATION PARAMETERS
CARS_Q = []                             # queue used in car counting
CLASS_IDS_Q = []                        # queue used to store class IDs for classification

CARS_COUNT = 0
BUS_COUNT = 0
TRUCK_COUNT = 0
MOTOR_COUNT = 0


def car_in_q(object_id, queue):
    found = False
    for m in range(0, len(queue)):
        if queue[m][0] == object_id:
            found = True
            break
    return found


model = YOLO(MODEL)

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=FONT_SCALE)

# FPS COUNT
start_time = time.time()
frame_no = 0

now = datetime.datetime.now()
log = open(LOG, mode='w')                               # create log file
log.write("\nSystem Startup at:" + str(now) + "\n\n")

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=640,                     # default image size
                          conf=0.6,                     # default confidence value = 0.25
                          stream=True,
                          save=False,
                          show=False):

    frame = result.orig_img
    frame_no += 1

    # MANUAL ROI
    if not str(VIDEO).isnumeric():
        cv.line(frame, LINE1_START, LINE1_END, TEAL, THICKNESS)
        cv.line(frame, LINE2_START, LINE2_END, TEAL, THICKNESS)
        cv.line(frame, LINE3_START, LINE3_END, TEAL, THICKNESS)

    detections = sv.Detections.from_yolov8(result)
    detections.tracker_id = result.boxes.id

    car_ids = None

    if detections.tracker_id is not None:
        detections.tracker_id = np.array(detections.tracker_id, np.int32)

        # GETTING X, Y, W, H for EACH DETECTED OBJECT
        car_ids = detections.tracker_id
        class_ids = detections.class_id
        boxes = np.array(detections.xyxy, dtype="int32")

        for box, car_id, class_id in zip(boxes, car_ids, class_ids):
            x, y, w, h = box
            cv.circle(frame, (x, y), 5, GREEN, -1)
            cv.circle(frame, (w, h), 5, GREEN, -1)

            # BBOX CENTER
            cx = (x + w) // 2
            cy = (y + h) // 2

            # VEHICLE COUNTING STAGE
            check1 = cy <= (LINE1_START[1] + OFFSET)            # vehicle center at line 1
            check2 = cy >= (LINE1_START[1] - OFFSET)
            check3 = cx <= LINE1_END[0]

            checkQ = not car_in_q(car_id, CARS_Q)

            check5 = cy <= (LINE2_START[1] + OFFSET)            # vehicle center at line 2
            check6 = cy >= (LINE2_START[1] - OFFSET)
            check7 = cx <= LINE2_END[0]

            if check1 and check2 and check3 and checkQ:
                CENTER = [cx, cy]
                CARS_Q.append([car_id, CENTER])
                if class_id == 2:
                    CARS_COUNT += 1
                elif class_id == 5:
                    BUS_COUNT += 1
                elif class_id == 7:
                    TRUCK_COUNT += 1
                elif class_id == 3:
                    MOTOR_COUNT += 1

            if check5 and check6 and check7 and checkQ:
                CENTER = [cx, cy]
                CARS_Q.append([car_id, CENTER])
                if class_id == 2:
                    CARS_COUNT += 1
                elif class_id == 5:
                    BUS_COUNT += 1
                elif class_id == 7:
                    TRUCK_COUNT += 1
                elif class_id == 3:
                    MOTOR_COUNT += 1

            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER1, FONT, FONT_SCALE, RED, THICKNESS)
            cv.putText(frame, f"LEFT LANE : {BUS_COUNT} BUSES", LEFT_CORNER2, FONT, FONT_SCALE, RED, THICKNESS)
            cv.putText(frame, f"LEFT LANE : {TRUCK_COUNT} TRUCKS", LEFT_CORNER3, FONT, FONT_SCALE, RED, THICKNESS)
            cv.putText(frame, f"LEFT LANE : {MOTOR_COUNT} MOTORS", LEFT_CORNER4, FONT, FONT_SCALE, RED, THICKNESS)

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    elapsed_time = time.time() - start_time
    fps = round(frame_no / elapsed_time, 1)
    cv.putText(frame, "FPS : " + str(fps), RIGHT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

    cv.imshow("RESULT", frame)

    # SAVING THE OUTPUT
    if SAVE:
        OUT_FRAME.write(frame)

    key = cv.waitKey(ms)
    if key == ESC:
        cv.destroyAllWindows()
        log.write("\nTotal Cars: " + str(CARS_COUNT))
        now = datetime.datetime.now()
        log.write("\nShutdown at: " + str(now) + "\n\n")
        log.close()
        exit()

cv.destroyAllWindows()
log.write("\nTotal Cars: " + str(CARS_COUNT))
now = datetime.datetime.now()
log.write("\nSystem Shutdown at: " + str(now) + "\n\n")
log.close()

print("Done....")
