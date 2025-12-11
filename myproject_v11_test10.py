# Multi-Moving Object Speed and Size estimation based YOLOv8 DNN Model & ByteTrack Tracking Algorithm
# Author: Saif Bashar 2023
# CONDA VENV: test10
# Version: 11.0 Update 2023-07-14, 10:00 AM
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
RED = (97, 12, 159)
WHITE = (255, 255, 255)
YELLOW = (90, 217, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2
ESC = 27                                # escape key ASCII code

# SYSTEM FLAGS
VERTICAL_VIEW = False                   # camera vertical on the road
EVAL = False                            # speed evaluation ROIs
SEGMENT = True                          # enable segmentation
SKIP = False                            # enable frame skipping for faster processing
TRAIL = False                           # enable tracking trail drawing
SAVE = False                            # enable saving output file
MPH = True                              # store speed in Mile per Hour Measurement

# SYSTEM OUTPUT
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.csv"                         # output log file
TIME_LOG = "TIME_LOG.csv"               # output log file

# MODEL PARAMETERS
CLASSES = [2, 3, 5, 7]                  # detection on classes (Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker
MODEL = "../models/yolov8s-seg.pt"      # x = Extra Large, l = large, m = medium, s = small, n = nano
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
LEFT_SHIFT = 300                                    # ROI offset (Move ROI left)
PPM = 10                                            # 20 pixel per 1 meter
METERS = PPM * 8                                    # distance between lines
SPPM = 8 ** 3

HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

# ROI COORDINATES
if VERTICAL_VIEW:
    PPM = 8
    METERS = PPM * 10
    LINE1_START = (130, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (0, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH + 50, HALF_HEIGHT + ROI_OFFSET)

    LINE3_START = (0, (HALF_HEIGHT + ROI_OFFSET) + METERS)
    LINE3_END = (HALF_WIDTH + 80, (HALF_HEIGHT + ROI_OFFSET) + METERS)
elif EVAL:
    PPM = 8
    METERS = PPM * 10
    LINE1_START = (350, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH + 150, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (370, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH + 550, HALF_HEIGHT + ROI_OFFSET)

    LINE3_START = (390, (HALF_HEIGHT + ROI_OFFSET) + METERS)
    LINE3_END = (HALF_WIDTH + 780, (HALF_HEIGHT + ROI_OFFSET) + METERS)
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
CLASS_IDS_Q = []                        # queue used to store class IDs for classification
CARS_COUNT = 0
SPEED_TXT = ""
AREA_TXT = ""
SIZE_TXT = ""


# DRAW TRACKING TRAIL
def draw_trail(a1, b1, obj, img):
    center = (a1, b1)
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
frame_no = 0

now = datetime.datetime.now()
log = open(LOG, mode='w')                               # create log file
time_log = open(TIME_LOG, mode='w')                     # create time log file
timing_acc = 0

log.write("\nSystem Startup at:" + str(now) + "\n\n")

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=640,                    # default image size
                          conf=0.5,                     # default confidence value = 0.25
                          stream=True,
                          save=False,
                          show=False,
                          half=False,
                          visualize=False,
                          retina_masks=False):


    frame = result.orig_img
    frame_no += 1

    timing_start = time.time()

    area = 0

    # FRAME SKIPPING
    if SKIP and frame_no % 2 == 0:
        continue

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
            CLASS_IDS_Q.append([car_id, class_id])
            cv.circle(frame, (x, y), 5, GREEN, -1)
            cv.circle(frame, (w, h), 5, GREEN, -1)

            # BBOX CENTER
            cx = (x + w) // 2
            cy = (y + h) // 2

            # DRAW TRACKING TRAIL
            if TRAIL:
                draw_trail(cx, cy, car_id, frame)

            # VEHICLE COUNTING STAGE
            check1 = cy <= (LINE1_START[1] + OFFSET)            # vehicle center at line 1
            check2 = cy >= (LINE1_START[1] - OFFSET)
            check3 = cx <= LINE1_END[0]
            checkQ1 = not car_in_q(car_id, CARS_Q1)
            car_at_line1 = check1 and check2 and check3

            check5 = cy <= (LINE2_START[1] + OFFSET)            # vehicle center at line 2
            check6 = cy >= (LINE2_START[1] - OFFSET)
            check7 = cx <= LINE2_END[0]
            checkQ2 = not car_in_q(car_id, CARS_Q2)

            check8 = cy <= (LINE3_START[1] + OFFSET)            # vehicle center at line 3
            check9 = cy >= (LINE3_START[1] - OFFSET)
            check10 = cx <= LINE3_END[0]
            checkQ3 = not car_in_q(car_id, CARS_Q3)

            speed = 0
            speed1 = 0
            speed2 = 0

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
                else:                                           # ESTIMATING SPEED 1 PROCESS
                    for i in range(0, len(CARS_Q1)):
                        if car_id == CARS_Q1[i][0]:
                            dx = CENTER2[0] - CARS_Q1[i][1][0]
                            dy = CENTER2[1] - CARS_Q1[i][1][1]
                            pixel_distance1 = math.sqrt(dx ** 2 + dy ** 2)                      # distance in pixels
                            real_distance1 = pixel_distance1 / PPM                              # distance in meters
                            delta_time1 = TIME2 - CARS_Q1[i][2]
                            speed1 = (real_distance1 / delta_time1) * 3.6                       # 1 m/s = 3.6 km/h
                            SPEEDS_Q1.append([car_id, speed1])

            if check8 and check9 and check10 and checkQ3:
                CENTER3 = [cx, cy]
                TIME3 = time.time()
                CARS_Q3.append([car_id, CENTER3, TIME3])

                if car_in_q(car_id, CARS_Q2):                  # ESTIMATING SPEED 2 PROCESS
                    for i in range(0, len(CARS_Q2)):
                        if car_id == CARS_Q2[i][0]:
                            dx = CENTER3[0] - CARS_Q2[i][1][0]
                            dy = CENTER3[1] - CARS_Q2[i][1][1]
                            pixel_distance2 = math.sqrt(dx ** 2 + dy ** 2)                  # distance in pixels
                            real_distance2 = pixel_distance2 / PPM                          # distance in meters
                            delta_time2 = TIME3 - CARS_Q2[i][2]
                            speed2 = (real_distance2 / delta_time2) * 3.6                   # 1 m/s = 3.6 km/h
                            SPEEDS_Q2.append([car_id, speed2])

                if car_in_q(car_id, SPEEDS_Q1) and car_in_q(car_id, SPEEDS_Q2):
                    index1 = find_index(car_id, SPEEDS_Q1)
                    index2 = find_index(car_id, SPEEDS_Q2)
                    speed = (SPEEDS_Q1[index1][1] + SPEEDS_Q2[index2][1]) // 2
                    NOW = datetime.datetime.now()
                    SPEED_TXT = f"Speed of Car {car_id}, {speed} km/h, " + f" Measured at: {NOW}"
                    log.write(SPEED_TXT + "\n")

                    if MPH:
                        speed_mph = int(speed * 1.609344)
                        NOW = datetime.datetime.now()
                        SPEED_MPH_TXT = f"Speed of Car {car_id}, {speed_mph} m/h, " + f" Measured at: {NOW}"
                        log.write(SPEED_MPH_TXT + "\n")

                    SPEEDS_Q1.pop(index1)
                    SPEEDS_Q2.pop(index2)

            cv.putText(frame, f"LEFT LANE : {CARS_COUNT} CARS", LEFT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

        cv.putText(frame, SPEED_TXT, CENTER_TOP, FONT, FONT_SCALE + 0.1, BLACK, THICKNESS)

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # SIZE ESTIMATING STAGE
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

                            # ESTIMATING AREA
                            pixels_area = cv.contourArea(mask)
                            real_area = pixels_area // SPPM

                            # ESTIMATING SIZE
                            polygon = mask

                            length_p1 = min(polygon, key=lambda loc: loc[0])
                            length_p2 = max(polygon, key=lambda loc: loc[0])
                            cv.line(frame, length_p1, length_p2, WHITE, THICKNESS + 1)

                            height_p1 = polygon[0]
                            height_p2 = polygon[5]
                            cv.line(frame, height_p1, height_p2, WHITE, THICKNESS + 1)

                            v_length = math.sqrt((length_p2[0] - length_p1[0]) ** 2 + (length_p2[1] - length_p1[1]) ** 2)
                            v_height = math.sqrt((height_p2[0] - length_p1[0]) ** 2 + (height_p2[1] - length_p1[1]) ** 2)
                            v_width = 1.5

                            for item in CLASS_IDS_Q:
                                if obj_id == CLASS_IDS_Q[0][0]:
                                    class_id = CLASS_IDS_Q[0][1]
                                    break
                            if class_id == 3:
                                v_width = 0.5
                            elif class_id == 5:
                                v_width = 2
                            elif class_id == 7:
                                v_width = 2.5

                            real_size = (v_length * v_width * v_height) // SPPM
                            SIZES_Q.append([obj_id, real_area, real_size])
                            NOW = datetime.datetime.now()

                            AREA_TXT = f"Area of Car:{obj_id}, {int(real_area)} m^2, " + f" Measured at: {NOW}"
                            SIZE_TXT = f"Size of Car:{obj_id}, {int(real_size)} m^3, " + f" Measured at: {NOW}"

                            log.write(AREA_TXT + "\n")
                            log.write(SIZE_TXT + "\n")

                    except TypeError:
                        print("Segmentation Error")

        except TypeError:
            print("Segmentation Error")

    elapsed_time = time.time() - start_time
    fps = round(frame_no / elapsed_time, 1)

    timing_close = time.time()
    timing_difference = timing_close - timing_start
    timing_acc += timing_difference
    #time_log.write("\n Time Acc: " + str(timing_acc) + "\n\n")

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

time_avg = (timing_acc / frame_no) * 1000
time_log.write("\nAverage Time: " + str(time_avg) + "\n\n")

cv.destroyAllWindows()
log.write("\nTotal Cars: " + str(CARS_COUNT))
now = datetime.datetime.now()
log.write("\nSystem Shutdown at: " + str(now) + "\n\n")
log.close()

print("Done....")
