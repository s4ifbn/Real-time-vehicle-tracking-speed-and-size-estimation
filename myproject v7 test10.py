# Multi-Moving Object Speed and Size estimation based YOLOv8 DNN Model & ByteTrack Algorithm
# Author: Saif Bashar 2023
# VENV: test10
# Version: 7.0 Update 2023-04-30, 08:00 AM
# All packages are up-to-date

import math
import time
import datetime
import cv2 as cv
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque

# COLORS & FONTS
BLACK = (0, 0, 0)
BLUE = (188, 108, 87)
GREEN = (0, 255, 0)
RED = (71, 41, 252)
WHITE = (255, 255, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
SCALE = 0.6
THICKNESS = 2
ESC = 27

# SOURCE
# VIDEO = "../data/traffic.avi"
VIDEO = "../data/test1.mov"
# VIDEO = 0                 # iPhone 13 Camera
# VIDEO = 1                 # Laptop Camera

SKIP = False                # Enable Frame Skipping
SAVE = False                # Enable Output Save File
TRAIL = True                # Enable Tracking Trail Drawing
OUT = "output.mp4"
LOG = "LOG.txt"
OUT_FRAME = None

# MODEL PARAMETERS
CLASSES = [0, 2, 3, 5, 7]   # Detection on Vehicles Only (Person, Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"
DEVICE = "mps"              # or cuda

MODEL = "../models/yolov8s-seg.pt"


# GET VIDEO INFO
def get_vid_info(video):
    global VIDEO
    info = sv.VideoInfo.from_video_path(VIDEO)
    width = info.width
    height = info.height
    frames_per_second = info.fps
    total_frames = info.total_frames
    return width, height, frames_per_second, total_frames


# SOURCE VIDEO INFORMATION
WIDTH, HEIGHT, FPS, TOTAL_FRAMES = get_vid_info(VIDEO)
print(f"SOURCE: WIDTH= {WIDTH}, HEIGHT= {HEIGHT}, FPS={FPS}, FRAMES={TOTAL_FRAMES}")

if SAVE:
    OUT_FRAME = cv.VideoWriter(OUT, cv.VideoWriter_fourcc(*'mp4v'), FPS, (WIDTH, HEIGHT))


# RESIZE VIDEO
def resize(video):
    global VIDEO, FPS
    resolution = (640, 640)
    VIDEO = "../data/traffic.mp4"
    cap = cv.VideoCapture(video)
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(VIDEO, fourcc, FPS, resolution)

    while cap.isOpened():
        ret, f = cap.read()
        if ret:
            resized_frame = cv.resize(f, resolution, interpolation=cv.INTER_AREA)
            out.write(resized_frame)
        else:
            break

    cap.release()
    out.release()


# if WIDTH != 640:
#     resize(VIDEO)
#     WIDTH, HEIGHT, FPS, TOTAL_FRAMES = get_vid_info(VIDEO)
#     print(f"SOURCE: WIDTH= {WIDTH}, HEIGHT= {HEIGHT}, FPS={FPS}, FRAMES={TOTAL_FRAMES}")


# INFO DISPLAY
LEFT_CORNER = (10, 30)                              # FOR CAR COUNTING DISPLAY
RIGHT_CORNER = (WIDTH - 100, 30)                    # FOR FPS DISPLAY
CENTER_TOP = (550, 60)                              # FOR CAR SPEED DISPLAY

# ROI
LINE1_START = (70, HEIGHT // 2)
LINE1_END = ((WIDTH // 2) - 100, HEIGHT // 2)
LINE2_START = (260, (HEIGHT // 2) + 120)
LINE2_END = ((WIDTH // 2) + 190, (HEIGHT // 2) + 100)

# CALIBRATION PARAMETERS
PPM = 70                     # Pixel per Meter = IMAGE_OBJ_SIZE / REAL_OBJ_SIZE
# F2M = 15.24                # For 2 Lines and the space between them total=50 feet

# FOR COUNTING & SPEED ESTIMATION
ENTER_CENTER = (0, 0)
LEAVE_CENTER = (0, 0)
ENTER_TIME = 0
LEAVE_TIME = 0
TRACKS_Q = {}
CARS_Q = []
CARS_COUNT = 0
SPEED_TXT = ""
AREA_TXT = ""
SIZE_TXT = ""
SPEED = 0


# DRAW TRACKING TRAIL
def draw_trail(a1, b1, a2, b2, obj, img):
    center = (int((a1 + a2) / 2), int((b1 + b2) / 2))
    if obj not in TRACKS_Q:
        TRACKS_Q[obj] = deque(maxlen=32)

    TRACKS_Q[obj].appendleft(center)
    for i in range(1, len(TRACKS_Q[obj])):
        if TRACKS_Q[obj][i - 1] is None or TRACKS_Q[obj][i] is None:
            continue
        cv.line(img, TRACKS_Q[obj][i - 1], TRACKS_Q[obj][i], RED, THICKNESS)


def count_cars(a1, b1, a2, b2, car_id):
    global ENTER_CENTER, ENTER_TIME, CARS_Q
    count = 0
    cx = int(a1 + a2) / 2
    cy = int(b1 + b2) / 2
    offset = 3

    check1 = cy <= (LINE1_START[1] + offset)
    check2 = cy >= (LINE1_START[1] - offset)
    check3 = cx <= LINE1_END[0]                  # for left lane only
    check4 = car_id not in CARS_Q

    if check1 and check2 and check3 and check4:
        ENTER_CENTER = (cx, cy)
        ENTER_TIME = time.time()
        CARS_Q.append(car_id)
        count = 1
    return count


def check_leaving(cx, cy, car_id):
    offset = 3
    check1 = cy <= (LINE2_START[1] + offset)
    check2 = cy >= (LINE2_START[1] - offset)
    check3 = cx <= LINE2_END[0]                 # for left lane only
    check4 = car_id not in CARS_Q

    if check1 and check2 and check3:
        return True
    else:
        return False


# ESTIMATE SPEED
def get_speed(a1, b1, a2, b2, car_id):
    global ENTER_CENTER, ENTER_TIME, LEAVE_CENTER, LEAVE_TIME, SPEED_TXT

    cx = int(a1 + a2) / 2
    cy = int(b1 + b2) / 2
    leave = check_leaving(cx, cy, car_id)

    if leave:
        LEAVE_TIME = time.time()
        LEAVE_CENTER = (cx, cy)
        dx = LEAVE_CENTER[0] - ENTER_CENTER[0]
        dy = LEAVE_CENTER[1] - ENTER_CENTER[1]
        distance = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        real_distance = distance / PPM
        delta_time = (LEAVE_TIME - ENTER_TIME) / FPS
        speed = real_distance / delta_time
        print(f"speed: {speed}, distance: {real_distance}, took: {delta_time} s for: Obj: {car_id}")
        return int(speed)


# ESTIMATE SIZE
def get_size(a1, b1, a2, b2, segment):
    cx = int(a1 + a2) / 2
    cy = int(b1 + b2) / 2
    if check_leaving(cx, cy):
        pixels = cv.contourArea(segment)
        return pixels


model = YOLO(MODEL)
model.fuse()

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=SCALE)

# FPS COUNT
start_time = time.time()
now = datetime.datetime.now()
frame_no = 0

log = open(LOG, mode='w')
log.write("\nRun at: " + str(now) + "\n\n")

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=640,
                          conf=0.25,
                          stream=True,
                          save=False,
                          show=False):

    frame = result.orig_img
    frame_no += 1
    area = 0

    # FRAME SKIPPING
    if SKIP and frame_no % 2 == 0:
        continue

    # MANUAL ROI
    if not str(VIDEO).isnumeric():
        cv.line(frame, LINE1_START, LINE1_END, GREEN, THICKNESS + 1)
        cv.line(frame, LINE2_START, LINE2_END, GREEN, THICKNESS + 1)

    detections = sv.Detections.from_yolov8(result)
    detections.tracker_id = result.boxes.id

    ids = None
    if detections.tracker_id is not None:
        detections.tracker_id = detections.tracker_id.cpu().numpy().astype(int)

        # GETTING X, Y, W, H for EACH DETECTED OBJECT
        ids = detections.tracker_id
        boxes = np.array(detections.xyxy, dtype="int32")
        for box, obj_id in zip(boxes, ids):
            x, y, w, h = box
            cv.circle(frame, (x, y), 5, GREEN, -1)
            cv.circle(frame, (w, h), 5, GREEN, -1)

            if TRAIL:
                draw_trail(x, y, w, h, obj_id, frame)

            # COUNTING CARS
            CARS_COUNT += count_cars(x, y, w, h, obj_id)
            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER, FONT, SCALE, RED, THICKNESS)

            # cv.rectangle(frame, (x, y), (w, h), GREEN, thickness=10)
            # print(f"For Object {obj_id} x= {x}, y= {y}, w= {w}, h= {h}")

            SPEED = get_speed(x, y, w, h, obj_id)
            if SPEED is not None:
                SPEED_TXT = f"Car {obj_id} Speed: {SPEED} KM/S"

    cv.putText(frame, SPEED_TXT, CENTER_TOP, FONT, SCALE + 0.1, WHITE, THICKNESS)

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    try:
        masks = result.masks

        if masks is not None:
            for mask, obj_id in zip(masks.xyn, ids):
                mask[:, 0] *= WIDTH                 # denormalize values
                mask[:, 1] *= HEIGHT                # denormalize values
                mask = np.array(mask, dtype=np.int32)
                x1 = mask[0][0]                     # first point of each mask
                y1 = mask[0][1]
                # cv.circle(frame, (x1, y1), 5, GREEN, -1)
                # cv.polylines(frame, [mask], True, BLUE, THICKNESS)
                cv.fillPoly(frame, [mask], RED)
                area = cv.contourArea(mask)
                if area is not None and ids is not None:
                    AREA_TXT = f" Area of Car:{obj_id} = {int(area)} px"
                    # SIZE_TXT = f" Size of Car:{obj_id} = {int(area)} px"
                    print(AREA_TXT)
                    # print(SIZE_TXT)

    except:
        print("Segmentation Error")

    if SPEED_TXT != "":
        log.write("Speed: " + SPEED_TXT + " " + "Size: " + AREA_TXT + "\n")

    seconds = time.time() - start_time
    fps = round(frame_no / seconds)
    cv.putText(frame, "FPS: " + str(fps), RIGHT_CORNER, FONT, SCALE, RED, THICKNESS)

    cv.imshow("RESULT", frame)

    # SAVING THE OUTPUT
    if SAVE:
        OUT_FRAME.write(frame)

    key = cv.waitKey(1)
    if key == ESC:
        cv.destroyAllWindows()
        log.write("\n Total Cars: " + str(CARS_COUNT))
        log.close()
        exit()

cv.destroyAllWindows()
log.write("\n Total Cars: " + str(CARS_COUNT))
log.close()

print("Done....")
