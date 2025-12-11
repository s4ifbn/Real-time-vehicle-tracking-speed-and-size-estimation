# Multi-Moving Object Speed and Size estimation based YOLOv8 DNN Model & ByteTrack Tracking Algorithm
# Author: Saif Bashar 2023
# VENV: test10
# Version: 9.0 Update 2023-05-16, 08:00 AM
# All important packages are up-to-date

import math
import time
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
RED = (71, 41, 252)
WHITE = (255, 255, 255)
YELLOW = (90, 217, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
ESC = 27                                # escape key ASCII code
ms = 1                                  # waiting 1 ms between frames, set ms=0 for debugging

# SOURCE
VIDEO = "../data/traffic.avi"           # default test video
# VIDEO = "../data/test4.mov"           # recorded video
# VIDEO = 0                             # live camera

CAMERA = False
SKIP = False                            # enable frame skipping for faster processing
SEGMENT = False                          # enable segmentation
TRAIL = False                           # enable tracking trail drawing
SAVE = False                            # enable saving output file
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.txt"                         # output log file

# MODEL PARAMETERS
CLASSES = [0, 2, 3, 5, 7]                  # detection on classes (Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker

# DEVICE AGNOSTIC CODE
DEVICE = "cpu"                          # CPU with No GPU Support
if torch.backends.mps.is_available():
    DEVICE = "mps"                      # for Apple GPU
elif torch.cuda.is_available():
    DEVICE = "cuda:0"                   # for NVIDIA GPU

print("RUNNING USING " + DEVICE)

# PRETRAINED MODEL SIZE
MODEL = "../models/yolov8s-seg.pt"      # x = Extra Large, l = large, m = medium, s = small, n = nano


# GET VIDEO INFO
def get_vid_info(video):
    info = sv.VideoInfo.from_video_path(video)
    width = info.width
    height = info.height
    source_fps = info.fps
    total_frames = info.total_frames
    return width, height, source_fps, total_frames


# SOURCE VIDEO INFORMATION
WIDTH, HEIGHT, FPS, TOTAL_FRAMES = get_vid_info(VIDEO)
print(f"SOURCE VIDEO: WIDTH= {WIDTH}, HEIGHT= {HEIGHT}, FPS={FPS}, FRAMES={TOTAL_FRAMES}")

if SAVE:
    OUT_FRAME = cv.VideoWriter(OUT, cv.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))


# RESIZE VIDEO
def resize(video):
    global VIDEO, FPS
    resolution = (640, 640)
    VIDEO = "../data/output.mp4"
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(VIDEO, fourcc, FPS, resolution)

    while cap.isOpened():
        ret, source_frame = cap.read()
        if ret:
            resized_frame = cv.resize(source_frame, resolution, interpolation=cv.INTER_AREA)
            out.write(resized_frame)
        else:
            break

    cap.release()
    out.release()

# if WIDTH != 640:
#     resize(VIDEO)
#     WIDTH, HEIGHT, FPS, TOTAL_FRAMES = get_vid_info(VIDEO)
#     print(f"NEW RESIZED SOURCE VIDEO: WIDTH= {WIDTH}, HEIGHT= {HEIGHT}, FPS={FPS}, FRAMES={TOTAL_FRAMES}")


# INFO DISPLAY
LEFT_CORNER = (10, 30)                              # for car counting display
RIGHT_CORNER = (WIDTH - 100, 30)                    # for fps display
CENTER_TOP = (250, 60)                              # for car speed display

# CALIBRATION PARAMETER
PPM = 31.5                                          # pixel per meter

# ROI COORDINATES
LINE1_START = LINE1_END = 0
LINE2_START = LINE2_END = 0
LINE3_START = LINE3_END = 0
LEFT_LANE = 0

if not str(VIDEO).isnumeric():
    if VIDEO.endswith("avi"):                       # select default test video
        LINE1_START = (420, 400)
        LINE1_END = (650, 400)
        LINE2_START = (240, 500)
        LINE2_END = (620, 500)
        LINE3_START = (5, 600)
        LINE3_END = (590, 600)
        PPM = 6

    elif VIDEO.endswith("mov"):                     # select recorded video
        LINE1_START = (220, 390)
        LINE1_END = (700, 390)
        LINE2_START = (400, 490)
        LINE2_END = (985, 490)
        LINE3_START = (500, 590)
        LINE3_END = (1250, 590)
        PPM = 8
else:                                              # select camera
    LINE1_START = (420, 400)
    LINE1_END = (650, 400)
    LINE2_START = (240, 500)
    LINE2_END = (620, 500)
    LINE3_START = (5, 600)
    LINE3_END = (590, 600)
    PPM = 6
    CAMERA = True

OFFSET = 20                             # offset for ROI lines

# FOR COUNTING & SPEED ESTIMATION
CENTER1 = [0, 0]                        # car center coordinates at line 1
CENTER2 = [0, 0]                        # car center coordinates at line 2
CENTER3 = [0, 0]                        # car center coordinates at line 3
TIME1 = 0                               # car timestamp at line 1
TIME2 = 0                               # car timestamp at line 2
TIME3 = 0                               # car timestamp at line 3
TRACKS_Q = {}                           # queue used in track drawing function
CARS_Q1 = []                            # queue used in car counting
CARS_Q2 = []                            # queue used in line 2 speed estimation
CARS_Q3 = []                            # queue used in line 3 speed estimation
SPEEDS_Q1 = []                          # queue used to store speed value at line 2
SPEEDS_Q2 = []                          # queue used to store speed value at line 3
SIZES_Q = []                            # queue used to store size value at line 2
CARS_COUNT = 0
SPEED_TXT = ""
AREA_TXT = ""
SIZE_TXT = ""


# DRAW TRACKING TRAIL
def draw_trail(a1, b1, a2, b2, obj, img):
    center = (int((a1 + a2) / 2), int((b1 + b2) / 2))
    if obj not in TRACKS_Q:
        TRACKS_Q[obj] = deque(maxlen=32)

    TRACKS_Q[obj].appendleft(center)
    for j in range(1, len(TRACKS_Q[obj])):
        if TRACKS_Q[obj][j - 1] is None or TRACKS_Q[obj][j] is None:
            continue
        cv.line(img, TRACKS_Q[obj][j - 1], TRACKS_Q[obj][j], RED, THICKNESS)


def car_in_q(object_id, queue):
    found = False
    for m in range(0, len(queue)):
        if queue[m][0] == object_id:
            found = True
            break
    return found


def find_index(object_id, queue):
    for m in range(0, len(queue)):
        if queue[m][0] == object_id:
            return m


model = YOLO(MODEL)
model.fuse()

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=FONT_SCALE)

# FPS COUNT
start_time = time.time()
now = datetime.datetime.now()
frame_no = 0

log = open(LOG, mode='w')                               # create log file
log.write("\nRun at: " + str(now) + "\n\n")

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=640,                    # default image size
                          conf=0.25,                    # default confidence value = 0.25
                          stream=True,
                          save=False,
                          show=False):

    frame = result.orig_img
    frame_no += 1
    area = 0

    LEAVE = 0

    # FRAME SKIPPING
    if SKIP and frame_no % 3 == 0:
        continue

    # MANUAL ROI
    if not str(VIDEO).isnumeric():
        cv.line(frame, LINE1_START, LINE1_END, GREEN, THICKNESS + 1)
        cv.line(frame, LINE2_START, LINE2_END, GREEN, THICKNESS + 1)
        cv.line(frame, LINE3_START, LINE3_END, GREEN, THICKNESS + 1)

    detections = sv.Detections.from_yolov8(result)
    detections.tracker_id = result.boxes.id

    car_ids = None

    if detections.tracker_id is not None:
        detections.tracker_id = np.array(detections.tracker_id, np.int32)

        # GETTING X, Y, W, H for EACH DETECTED OBJECT
        car_ids = detections.tracker_id
        boxes = np.array(detections.xyxy, dtype="int32")

        for box, car_id in zip(boxes, car_ids):
            x, y, w, h = box
            cv.circle(frame, (x, y), 5, GREEN, -1)
            cv.circle(frame, (w, h), 5, GREEN, -1)

            # DRAW TRACKING TRAIL
            if TRAIL:
                draw_trail(x, y, w, h, car_id, frame)

            # COUNTING CARS PROCESS
            cx = int((x + w) / 2)
            cy = int((y + h) / 2)
            cv.circle(frame, (cx, cy), 5, RED, -1)

            check1 = cy <= (LINE1_START[1] + OFFSET)            # car center at line 1
            check2 = cy >= (LINE1_START[1] - OFFSET)
            check3 = cx <= LINE1_END[0]
            checkQ1 = not car_in_q(car_id, CARS_Q1)

            check5 = cy <= (LINE2_START[1] + OFFSET)            # car center at line 2
            check6 = cy >= (LINE2_START[1] - OFFSET)
            check7 = cx <= LINE2_END[0]
            checkQ2 = not car_in_q(car_id, CARS_Q2)

            check8 = cy <= (LINE3_START[1] + OFFSET)            # car center at line 3
            check9 = cy >= (LINE3_START[1] - OFFSET)
            check10 = cx <= LINE3_END[0]
            checkQ3 = not car_in_q(car_id, CARS_Q3)

            speed = 0
            speed1 = 0
            speed2 = 0

            if check1 and check2 and check3 and checkQ1:
                CENTER1 = [cx, cy]
                TIME1 = time.monotonic()
                CARS_Q1.append([car_id, CENTER1, TIME1])
                CARS_COUNT += 1

            if check5 and check6 and check7 and checkQ2:
                CENTER2 = [cx, cy]
                TIME2 = time.monotonic()
                CARS_Q2.append([car_id, CENTER2, TIME2])
                LEAVE = 2

                if not car_in_q(car_id, CARS_Q1):
                    CARS_COUNT += 1
                else:                                            # ESTIMATING SPEED 1 PROCESS
                    for i in range(0, len(CARS_Q1)):
                        if car_id == CARS_Q1[i][0]:
                            dx = CENTER2[0] - CARS_Q1[i][1][0]
                            dy = CENTER2[1] - CARS_Q1[i][1][1]
                            pixel_distance1 = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))      # distance in pixels
                            real_distance1 = pixel_distance1 / PPM                              # distance in meters
                            delta_time1 = TIME2 - CARS_Q1[i][2]
                            speed1 = (real_distance1 / delta_time1) * 3.6                       # speed in km/h
                            SPEEDS_Q1.append([car_id, speed1])
                            # SPEED_TXT = f"Car {car_id} Speed 1: {speed1} km/h"
                            # log.write("Speed1: " + SPEED_TXT + "\n")

            if check8 and check9 and check10 and checkQ3:
                CENTER3 = [cx, cy]
                TIME3 = time.monotonic()
                CARS_Q3.append([car_id, CENTER3, TIME3])

                if car_in_q(car_id, CARS_Q2):                   # ESTIMATING SPEED 2 PROCESS
                    for i in range(0, len(CARS_Q2)):
                        if car_id == CARS_Q2[i][0]:
                            dx = CENTER3[0] - CARS_Q2[i][1][0]
                            dy = CENTER3[1] - CARS_Q2[i][1][1]
                            pixel_distance2 = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))  # distance in pixels
                            real_distance2 = pixel_distance2 / PPM                          # distance in meters
                            delta_time2 = TIME3 - CARS_Q2[i][2]
                            speed2 = (real_distance2 / delta_time2) * 3.6                   # 1 m/s = 3.6 km/h
                            SPEEDS_Q2.append([car_id, speed2])
                            # SPEED_TXT = f"Car {car_id} Speed 2: {speed2} km/h"
                            # log.write("Speed2: " + SPEED_TXT + "\n")

                if car_in_q(car_id, SPEEDS_Q1) and car_in_q(car_id, SPEEDS_Q2):
                    index1 = find_index(car_id, SPEEDS_Q1)
                    index2 = find_index(car_id, SPEEDS_Q2)
                    speed = int((SPEEDS_Q1[index1][1] + SPEEDS_Q2[index2][1]) / 2)
                    NOW = datetime.datetime.now()
                    SPEED_TXT = f"Car {car_id} Average Speed: {speed} km/h\t\t" + f"Measured at: {NOW}"
                    log.write(SPEED_TXT + "\n")
                    SPEEDS_Q1.pop(index1)
                    SPEEDS_Q2.pop(index2)

            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

        cv.putText(frame, SPEED_TXT, CENTER_TOP, FONT, FONT_SCALE + 0.1, BLACK, THICKNESS)

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # ESTIMATING SIZE PROCESS
    if SEGMENT:
        try:
            masks = result.masks

            if masks is not None:
                for mask, obj_id in zip(masks.xyn, car_ids):
                    mask[:, 0] *= WIDTH                             # denormalize values
                    mask[:, 1] *= HEIGHT                            # denormalize values
                    mask = np.array(mask, dtype=np.int32)
                    x1 = mask[0][0]                                 # first point of each mask
                    y1 = mask[0][1]
                    cv.fillPoly(frame, [mask], RED)

                    check_mask1 = y1 <= (LINE2_START[1] + OFFSET)
                    check_mask2 = y1 >= (LINE2_START[1] - OFFSET)
                    check_mask3 = x1 <= (LINE2_END[0])
                    check_maskQ = not car_in_q(obj_id, SIZES_Q)

                    if check_mask1 and check_mask2 and check_mask3 and check_maskQ:
                        pixels_area = cv.contourArea(mask)
                        real_area = pixels_area // (PPM * PPM)
                        print(real_area)
                        SIZES_Q.append([obj_id, real_area])
                        NOW = datetime.datetime.now()
                        AREA_TXT = f"Area of Car:{obj_id}: {int(real_area)} m\t\t\t\t" + f"Measured at: {NOW}"
                        log.write(AREA_TXT + "\n")

        except TypeError:
            print("Segmentation Error")

    seconds = time.time() - start_time
    fps = round(frame_no / seconds)
    cv.putText(frame, "FPS: " + str(fps), RIGHT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

    cv.imshow("RESULT", frame)

    # SAVING THE OUTPUT
    if SAVE:
        OUT_FRAME.write(frame)

    key = cv.waitKey(ms)
    if key == ESC:
        cv.destroyAllWindows()
        log.write("\n Total Cars: " + str(CARS_COUNT))
        log.close()
        exit()

cv.destroyAllWindows()
log.write("\n Total Cars: " + str(CARS_COUNT))
log.close()

print("Done....")
