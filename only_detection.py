import time
import cv2 as cv
import numpy as np
import torch.backends.mps
import supervision as sv
from ultralytics import YOLO

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

VIDEO = "../data/eval3.mp4"         # default test video

# SYSTEM FLAGS
VERTICAL_VIEW = False                   # camera vertical on the road
SEGMENT = False                          # enable segmentation
SKIP = False                            # enable frame skipping for faster processing
TRAIL = False                           # enable tracking trail drawing
SAVE = False                            # enable saving output file
MPH = True                              # store speed in Mile per Hour Measurement

# SYSTEM OUTPUT
OUT = "output.mp4"                      # output saved file
OUT_FRAME = None                        # output file frame
LOG = "LOG.csv"                         # output log file

# MODEL PARAMETERS
CLASSES = [2, 3, 5, 7]                  # detection on classes (Car, Motorcycle, Bus, Truck)
TRACKER = "bytetrack.yaml"              # ByteTrack tracker
MODEL = "../models/yolov8n.pt"      # x = Extra Large, l = large, m = medium, s = small, n = nano

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

# INFO DISPLAY
LEFT_CORNER = (10, 30)                              # for car counting display
RIGHT_CORNER = (WIDTH - 110, 30)                    # for fps display
CENTER_TOP = (250, 60)                              # for car speed display

# CALIBRATION PARAMETER
OFFSET = 20                                         # offset for each line
ROI_OFFSET = 120                                    # ROI offset (Move ROI up and down)
SPPM = 35 ** 2                                      # pixel squared per 1 meter
PPM = 10                                            # 20 pixel per 1 meter
METERS = PPM * 8                                    # distance between lines

HALF_WIDTH = WIDTH // 2
HALF_HEIGHT = HEIGHT // 2

# ROI COORDINATES
LEFT_SHIFT = 300

if VERTICAL_VIEW:
    PPM = 8
    METERS = PPM * 10
    LINE1_START = (0, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (0, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH, HALF_HEIGHT + ROI_OFFSET)

    LINE3_START = (0, (HALF_HEIGHT + ROI_OFFSET) + METERS)
    LINE3_END = (HALF_WIDTH, (HALF_HEIGHT + ROI_OFFSET) + METERS)
else:
    LINE1_START = (0, (HALF_HEIGHT + ROI_OFFSET) - METERS)
    LINE1_END = (HALF_WIDTH, (HALF_HEIGHT + ROI_OFFSET) - METERS)

    LINE2_START = (250, HALF_HEIGHT + ROI_OFFSET)
    LINE2_END = (HALF_WIDTH + LEFT_SHIFT, HALF_HEIGHT + ROI_OFFSET)

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

model = YOLO(MODEL)
model.fuse()

box_annotator = sv.BoxAnnotator(thickness=THICKNESS,
                                text_thickness=THICKNESS - 1,
                                text_scale=FONT_SCALE)

# FPS COUNT
start_time = time.time()
frame_no = 0

# MAIN LOOP
for result in model.track(source=VIDEO,
                          tracker=TRACKER,
                          classes=CLASSES,
                          device=DEVICE,
                          imgsz=640,                    # default image size
                          conf=0.5,                    # default confidence value = 0.25
                          stream=True,
                          save=False,
                          show=False):

    frame = result.orig_img

    frame_no += 1

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

    labels = [
        f"#{track_id} {model.model.names[class_id].capitalize()} {confidence:0.2f}"
        for _, _, confidence, class_id, track_id in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    elapsed_time = time.time() - start_time
    fps = round(frame_no / elapsed_time, 1)
    cv.putText(frame, "FPS: " + str(fps), RIGHT_CORNER, FONT, FONT_SCALE, RED, THICKNESS)

    cv.imshow("RESULT", frame)

    key = cv.waitKey(ms)
    if key == ESC:
        cv.destroyAllWindows()
        exit()

cv.destroyAllWindows()
print("Done....")
