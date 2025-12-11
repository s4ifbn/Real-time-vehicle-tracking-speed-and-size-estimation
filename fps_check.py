import cv2 as cv
import time

VIDEO = "../data/test1.mov"
cap = cv.VideoCapture(VIDEO)

start_time = time.time()
frame_no = 0

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FPS = int(cap.get(cv.CAP_PROP_FPS))
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
print(width, height, FPS, total_frames)

while True:
    ok, frame = cap.read()
    frame_no += 1
    if ok:
        elapsed_time = time.time() - start_time
        fps = round(frame_no / elapsed_time)
        cv.putText(frame, "FPS: " + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv.imshow("Road Sides Detection", frame)
        time.sleep(1 / FPS)

        key = cv.waitKey(1)
        if key == 27:
            cv.destroyAllWindows()
            cap.release()
            exit()

