import numpy as np
import cv2

cap = cv2.VideoCapture('video_testpattern.mp4')
#cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret, frame = cap.read()

    if (ret == False):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)

    c = cv2.waitKey(1)
    # 'q' pressed?
    if (c==113):
        break

cap.release()
