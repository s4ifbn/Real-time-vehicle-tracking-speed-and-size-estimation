# Multi-Moving Object Speed and Size estimation based YOLOv8 DNN Model & ByteTrack Algorithm
# Author: Saif Bashar 2023
# VENV: test10
# Version: 8.0 Update 2023-05-12, 08:00 AM
# All important packages are up-to-date

import math
import time
import datetime
import cv2 as cv
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque            # double ended queue

# COLORS & FONTS & KEYS
BLACK = (0, 0, 0)
BLUE = (188, 108, 87)
GREEN = (0, 255, 0)
RED = (71, 41, 252)
WHITE = (255, 255, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
ESC = 27                                # escape key ASCII code
ms = 1                                  # waiting 1 ms between frames, set ms=0 for debugging

# SOURCE
VIDEO = "../data/traffic.avi"           # default test video
# VIDEO = "../data/test1.mov"           # recorded video
# VIDEO = 0                             # live phone camera
# VIDEO = 1                             # live laptop camera

SKIP = False                            # enable frame skipping for faster processing
SAVE = False                            # enable saving output file
TRAIL = False                           # enable tracking trail drawing
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.txt"                         # output log file

# MODEL PARAMETERS
CLASSES = [0, 2, 3, 5, 7]               # detection on classes (Person, Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker
DEVICE = "mps"                          # or cuda for nvidia graphics

# PRETRAINED MODEL SIZE
MODEL = "../models/yolov8s-seg.pt"      # x = Extra Large, l = large, m = medium, s = small, n = nano


# GET VIDEO INFO
def get_vid_info(video):
    global VIDEO
    info = sv.VideoInfo.from_video_path(VIDEO)
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
CENTER_TOP = (550, 60)                              # for car speed display

# ROI COORDINATES
LINE1_START = LINE1_END = 0
LINE2_START = LINE2_END = 0
LINE3_START = LINE3_END = 0
LEFT_LANE = 0

if VIDEO.endswith("avi"):                         # select default test video
    LINE1_START = (420, 400)
    LINE1_END = (650, 400)
    LINE2_START = (240, 500)
    LINE2_END = (620, 500)
    LINE3_START = (5, 600)
    LINE3_END = (590, 600)
elif VIDEO.endswith("mov"):                       # select recorded video
    LINE1_START = (70, 380)
    LINE1_END = (500, 360)
    LINE2_START = (260, 490)
    LINE2_END = (780, 460)
    LINE3_START = (400, 680)
    LINE3_END = (1130, 570)

# CALIBRATION PARAMETER
PPM = 10                                         # pixel per meter = IMAGE_OBJ_SIZE / REAL_OBJ_SIZE

# FOR COUNTING & SPEED ESTIMATION
CENTER1 = [0, 0]
CENTER2 = [0, 0]
CENTER3 = [0, 0]
TIME1 = 0
TIME2 = 0
TIME3 = 0
TRACKS_Q = {}
CARS_Q1 = []
CARS_Q2 = []
CARS_Q3 = []
SIZES_Q = []
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


# def check_car_before_add(q, object_id):
#     add_car = False
#     if len(q) != 0:
#         for k in range(0, len(q)):
#             if q[k][0] == object_id:
#                 add_car = False
#                 break
#             else:
#                 add_car = True
#     else:
#         add_car = True
#     return add_car


def check_leaving(cx, cy):

    offset = 3

    check1 = cy <= (LINE2_START[1] + offset)
    check2 = cy >= (LINE2_START[1] - offset)
    check3 = cx <= LINE2_END[0]

    check4 = cy <= (LINE3_START[1] + offset)
    check5 = cy >= (LINE3_START[1] - offset)
    check6 = cx <= LINE3_END[0]

    if check1 and check2 and check3:
        return 1
    elif check4 and check5 and check6:
        return 2
    else:
        return False


# ESTIMATE SIZE
def get_size(a1, b1, a2, b2, segment):
    cx = int(a1 + a2) / 2
    cy = int(b1 + b2) / 2
    if check_leaving(cx, cy):
        pixels = cv.contourArea(segment)
        return pixels


model = YOLO(MODEL)
# model.fuse()

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=FONT_SCALE)

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
                          imgsz=640,                    # default image size
                          conf=0.25,                    # default confidence value = 0.25
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
        cv.line(frame, LINE3_START, LINE3_END, GREEN, THICKNESS + 1)

    detections = sv.Detections.from_yolov8(result)
    detections.tracker_id = result.boxes.id

    car_ids = None

    if detections.tracker_id is not None:
        detections.tracker_id = detections.tracker_id.cpu().numpy().astype(int)

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

            offset = 4
            LEAVE = 0

            check1 = cy <= (LINE1_START[1] + offset)            # car center at line 1
            check2 = cy >= (LINE1_START[1] - offset)
            check3 = cx <= LINE1_END[0]
            checkQ1 = not car_in_q(car_id, CARS_Q1)

            check5 = cy <= (LINE2_START[1] + offset)            # car center at line 2
            check6 = cy >= (LINE2_START[1] - offset)
            check7 = cx <= LINE2_END[0]
            checkQ2 = not car_in_q(car_id, CARS_Q2)

            check8 = cy <= (LINE3_START[1] + offset)            # car center at line 3
            check9 = cy >= (LINE3_START[1] - offset)
            check10 = cx <= LINE3_END[0]
            checkQ3 = not car_in_q(car_id, CARS_Q3)

            if check1 and check2 and check3 and checkQ1:
                CENTER1 = [cx, cy]
                TIME1 = datetime.datetime.now()
                print("t1:", TIME1)
                CARS_Q1.append([car_id, CENTER1, TIME1])
                CARS_COUNT += 1
                print("Q1", CARS_Q1)

            if check5 and check6 and check7 and checkQ2:
                CENTER2 = [cx, cy]
                TIME2 = datetime.datetime.now()
                print("t2:", TIME2)
                LEAVE = 1
                CARS_Q2.append([car_id, CENTER2, TIME2])
                if not car_in_q(car_id, CARS_Q1):
                    CARS_COUNT += 1
                print("Q2", CARS_Q2)

            if check8 and check9 and check10 and checkQ3:
                CENTER3 = [cx, cy]
                TIME3 = datetime.datetime.now()
                LEAVE = 2
                CARS_Q3.append([car_id, CENTER3, TIME3])
                print("Q3", CARS_Q3)

            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

            speed = 0
            speed1 = 0
            speed2 = 0

            # ESTIMATING SPEED PROCESS
            if LEAVE == 1:
                for i in range(0, len(CARS_Q2)):
                    if car_id == CARS_Q2[i][0]:
                        dx = CARS_Q2[i][1][0] - CARS_Q1[i][1][0]
                        dy = CARS_Q2[i][1][1] - CARS_Q1[i][1][1]
                        pixel_distance1 = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))        # distance in pixels
                        real_distance1 = pixel_distance1 / PPM                                # distance in meters
                        real_distance1 = real_distance1 / 1000                                # distance in kilometers
                        delta_time1 = (CARS_Q2[i][2] - CARS_Q1[i][2])
                        delta_time1 = delta_time1.total_seconds() / 3600                      # time in hours
                        speed1 = int(real_distance1 / delta_time1)
                        # SPEED_TXT = f"Car {car_id} Speed 1: {speed1} km/h"
                        print(f"speed1: {speed1}, distance: {real_distance1}, took: {delta_time1 * 3600}s for Car:{car_id}")

            if LEAVE == 2:
                for i in range(0, len(CARS_Q3)):
                    if car_id == CARS_Q3[i][0]:
                        dx = CARS_Q3[i][1][0] - CARS_Q2[i][1][0]
                        dy = CARS_Q3[i][1][1] - CARS_Q2[i][1][1]
                        pixel_distance2 = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))        # distance in pixels
                        real_distance2 = pixel_distance2 / PPM
                        real_distance2 = real_distance2 / 1000                                # distance in kilometers
                        delta_time2 = (CARS_Q3[i][2] - CARS_Q2[i][2])
                        delta_time2 = delta_time2.total_seconds() / 3600                      # time in hours
                        speed2 = int(real_distance2 / delta_time2)
                        # SPEED_TXT = f"Car {car_id} Speed 2: {speed2} km/h"
                        print(f"speed2: {speed2}, distance: {real_distance2}, took: {delta_time2 * 3600}s for Car:{car_id}")
                        speed = int((speed1 + speed2) / 2)
                        SPEED_TXT = f"Car {car_id} Average Speed: {speed} km/h"

        cv.putText(frame, SPEED_TXT, CENTER_TOP, FONT, FONT_SCALE + 0.1, WHITE, THICKNESS)

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # ESTIMATING SIZE PROCESS
    try:
        masks = result.masks

        if masks is not None:
            for mask, obj_id in zip(masks.xyn, car_ids):
                mask[:, 0] *= WIDTH                         # denormalize values
                mask[:, 1] *= HEIGHT                        # denormalize values
                mask = np.array(mask, dtype=np.int32)
                x1 = mask[0][0]                             # first point of each mask
                y1 = mask[0][1]
                # cv.circle(frame, (x1, y1), 5, GREEN, -1)
                # cv.polylines(frame, [mask], True, BLUE, THICKNESS)
                cv.fillPoly(frame, [mask], RED)
                area = cv.contourArea(mask)
                if area is not None and car_ids is not None:
                    AREA_TXT = f" Area of Car:{obj_id} = {int(area)} px"
                    # SIZE_TXT = f" Size of Car:{obj_id} = {int(area)} px"
                    # print(SIZE_TXT)
                
    except:
        print("Segmentation Error")

    if SPEED_TXT != "":
        log.write("Speed: " + SPEED_TXT + " " + "Size: " + AREA_TXT + "\n")

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
