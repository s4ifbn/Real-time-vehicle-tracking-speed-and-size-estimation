import cv2 as cv
import supervision as sv

resolution = (1280, 720)
in_video = "../data/eval42.mp4"
info = sv.VideoInfo.from_video_path(in_video)
fps = info.fps
out_video = "../data/output.mp4"
cap = cv.VideoCapture(in_video)
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(out_video, fourcc, fps, resolution)

while cap.isOpened():
    ret, source_frame = cap.read()
    if ret:
        resized_frame = cv.resize(source_frame, resolution, interpolation=cv.INTER_AREA)
        out.write(resized_frame)
    else:
        break

cap.release()
out.release()
print("done")
