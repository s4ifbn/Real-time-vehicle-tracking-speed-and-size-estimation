# Multi-Moving Object Speed and Size estimation based YOLOv8 DNN Model & ByteTrack Tracking Algorithm
# Author: Saif Bashar 2023
# CONDA VENV: test10
# Version: 11.0 Update 2023-07-08, 07:00 PM
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
GREEN = (221, 207, 69)
RED = (121, 100, 233)
WHITE = (255, 255, 255)
YELLOW = (90, 217, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
ESC = 27                                # escape key ASCII code

# SYSTEM FLAGS
EVAL_VIEW = True
VERTICAL_VIEW = False                   # camera vertical on the road
SEGMENT = False                         # enable segmentation
SKIP = False                            # enable frame skipping for faster processing
TRAIL = False                           # enable tracking trail drawing
SAVE = False                            # enable saving output file
MPH = False                             # store speed in Mile per Hour Measurement

# SYSTEM OUTPUT
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.csv"                         # output log file

# MODEL PARAMETERS
CLASSES = [2, 3, 5, 7]                  # detection on classes (Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker
MODEL = "../models/yolov8s.pt"          # x = Extra Large, l = large, m = medium, s = small, n = nano
VIDEO = "../data/eval30.mp4"            # default test video
ms = 1                                  # waiting 1 ms between frames, set ms=0 for debugging

# DEVICE AGNOSTIC CODE
DEVICE = "cpu"                          # CPU with No GPU Support
if torch.backends.mps.is_available():
    DEVICE = "mps"                      # for Apple GPU
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


# RESIZE VIDEO
def resize(video):
    global VIDEO, FPS
    resolution = (1280, 720)
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
RIGHT_CORNER = (WIDTH - 110, 30)                    # for fps display
CENTER_TOP = (250, 60)                              # for car speed display

# CALIBRATION PARAMETER
OFFSET = 20                                         # offset for each line
ROI_OFFSET = 120                                    # ROI offset (Move ROI up and down)
SPPM = 35 ** 2                                      # pixel squared per 1 meter
PPM = 20                                            # 20 pixel per 1 meter
METERS = PPM * 8                                    # distance between lines

HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

# ROI COORDINATES
LEFT_SHIFT = 300

if VERTICAL_VIEW:
    PPM = 8
    METERS = PPM * 10
    LINE1_START = (130, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (0, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH + 50, HALF_HEIGHT + ROI_OFFSET)

    LINE3_START = (0, (HALF_HEIGHT + ROI_OFFSET) + METERS)
    LINE3_END = (HALF_WIDTH + 80, (HALF_HEIGHT + ROI_OFFSET) + METERS)
elif EVAL_VIEW:
    ROI_SHIFT = 90
    PPM = 11
    METERS = 100
    if VIDEO.endswith("30.mp4"):
        ROI_SHIFT = 90
        PPM = 13
    if VIDEO.endswith("42.mp4"):
        ROI_SHIFT = 90
        PPM = 10
    if VIDEO.endswith("56.mp4"):
        ROI_SHIFT = 10
        PPM = 11
        METERS = 50
    if VIDEO.endswith("64.mp4"):
        ROI_SHIFT = 90
        PPM = 11
    if VIDEO.endswith("71.mp4"):
        ROI_SHIFT = 90
        PPM = 10
    if VIDEO.endswith("88.mp4"):
        ROI_SHIFT = 10
        PPM = 10
    if VIDEO.endswith("94.mp4"):
        ROI_SHIFT = 80
        PPM = 15
    if VIDEO.endswith("101.mp4"):
        ROI_SHIFT = 10
        PPM = 14
    LINE1_START = (0, 420 - ROI_SHIFT)
    LINE1_END = (WIDTH, 420 - ROI_SHIFT)
    LINE2_START = (0, 480 - ROI_SHIFT + METERS)
    LINE2_END = (WIDTH, 480 - ROI_SHIFT + METERS)
else:
    LINE1_START = (0, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH + 120, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (250, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH + LEFT_SHIFT + 50, HALF_HEIGHT + ROI_OFFSET)

    LINE3_START = (LEFT_SHIFT, (HALF_HEIGHT + ROI_OFFSET) + METERS)
    LINE3_END = (WIDTH, (HALF_HEIGHT + ROI_OFFSET) + METERS)

# COUNTING & SPEED ESTIMATION & SIZE ESTIMATION PARAMETERS
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
    center = ((a1 + a2) // 2, (b1 + b2) // 2)
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
# model.fuse()

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=FONT_SCALE)

# FPS COUNT
start_time = time.time()
frame_no = 0

now = datetime.datetime.now()
log = open(LOG, mode='w')                               # create log file
log.write("\nStartup at: " + str(now) + "\n\n")

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=320,                    # default image size
                          conf=0.5,                    # default confidence value = 0.25
                          stream=True,
                          save=False,
                          show=False,
                          half=False,
                          visualize=False,
                          retina_masks=False):

    frame = result.orig_img
    frame_no += 1
    area = 0

    # FRAME SKIPPING
    if SKIP and frame_no % 2 == 0:
        continue

    # MANUAL ROI
    cv.line(frame, LINE1_START, LINE1_END, GREEN, THICKNESS + 1)
    cv.line(frame, LINE2_START, LINE2_END, GREEN, THICKNESS + 1)

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
            cv.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv.circle(frame, (w, h), 5, (0, 255, 0), -1)

            # DRAW TRACKING TRAIL
            if TRAIL:
                draw_trail(x, y, w, h, car_id, frame)

            # COUNTING CARS PROCESS
            cx = w
            cy = h

            check1 = cy <= (LINE1_START[1] + OFFSET)            # car center at line 1
            check2 = cy >= (LINE1_START[1] - OFFSET)
            check3 = cx <= LINE1_END[0]
            checkQ1 = not car_in_q(car_id, CARS_Q1)
            car_at_line1 = check1 and check2 and check3

            check5 = cy <= (LINE2_START[1] + OFFSET)            # car center at line 2
            check6 = cy >= (LINE2_START[1] - OFFSET)
            check7 = cx <= LINE2_END[0]
            checkQ2 = not car_in_q(car_id, CARS_Q2)

            speed = 0

            if check1 and check2 and check3 and checkQ1:
                CENTER1 = [cx, cy]
                TIME1 = time.time()
                CARS_Q1.append([car_id, CENTER1, TIME1])
                CARS_COUNT += 1

            if check5 and check6 and check7 and checkQ2:
                CENTER2 = [cx, cy]
                TIME2 = time.time()
                CARS_Q2.append([car_id, CENTER2, TIME2])

                if not car_in_q(car_id, CARS_Q1):
                    CARS_COUNT += 1
                else:                                            # ESTIMATING SPEED PROCESS
                    for i in range(0, len(CARS_Q1)):
                        if car_id == CARS_Q1[i][0]:
                            dx = CENTER2[0] - CARS_Q1[i][1][0]
                            dy = CENTER2[1] - CARS_Q1[i][1][1]
                            pixel_distance1 = math.sqrt(dx ** 2 + dy ** 2)                      # distance in pixels
                            real_distance1 = pixel_distance1 / PPM                              # distance in meters
                            delta_time1 = TIME2 - CARS_Q1[i][2]
                            speed = (real_distance1 / delta_time1) * 3.6                       # speed in km/h
                            SPEEDS_Q1.append([car_id, speed])
                            NOW = datetime.datetime.now()
                            SPEED_TXT = f"Car {car_id}, Average Speed: {speed} km/h, " + f" Measured at: {NOW}"
                            log.write(SPEED_TXT + "\n")

                    if MPH:
                        speed_mph = int(speed * 1.609344)
                        SPEED_MPH_TXT = f"Car {car_id}, Average Speed: {speed_mph} m/h, " + f" Measured at: {NOW}"
                        log.write(SPEED_MPH_TXT + "\n")

            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

        cv.putText(frame, SPEED_TXT, CENTER_TOP, FONT, FONT_SCALE + 0.1, BLACK, THICKNESS)
        print(SPEED_TXT)
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
                    try:
                        x1 = mask[0][0]                                 # first point of each mask
                        y1 = mask[0][1]
                        cv.fillPoly(frame, [mask], (0, 0, 255))

                        check_mask1 = y1 <= (LINE2_START[1] + OFFSET)
                        check_mask2 = y1 >= (LINE2_START[1] - OFFSET)
                        check_mask3 = x1 <= (LINE2_END[0])
                        check_maskQ = not car_in_q(obj_id, SIZES_Q)

                        if check_mask1 and check_mask2 and check_mask3 and check_maskQ:
                            pixels_area = cv.contourArea(mask)
                            real_area = pixels_area // SPPM
                            SIZES_Q.append([obj_id, real_area])
                            NOW = datetime.datetime.now()
                            AREA_TXT = f"Area of Car:{obj_id}, {int(real_area)} m, " + f" Measured at: {NOW}"
                            log.write(AREA_TXT + "\n")
                    except:
                        print("Segmentation Error")

        except TypeError:
            print("Segmentation Error")

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
        log.write("\n Total Cars: " + str(CARS_COUNT))
        now = datetime.datetime.now()
        log.write("\nShutdown at: " + str(now) + "\n\n")
        log.close()
        exit()

cv.destroyAllWindows()
log.write("\n Total Cars: " + str(CARS_COUNT))
now = datetime.datetime.now()
log.write("\nShutdown at: " + str(now) + "\n\n")
log.close()

print("Done....")
