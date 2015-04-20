import numpy as np
import cv2
import cv2.cv as cv

from detector_state import DetectorState
from detection import Detection, Blob

"""
Ideas:
    - Do tracking to have best case heuristic of spot to look at.
    - Camshift: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift

Heuristics:
    - Return largest - there is only one dot
    - Maybe do blob detection before-after

"""


state = DetectorState((640, 480), 128, 5)


def cap_camera(state):
    cap = cv2.VideoCapture(1)
    cap.set(cv2.cv.cv_cap_prop_frame_width, state.resolution[0])
    cap.set(cv2.cv.cv_cap_prop_frame_height, state.resolution[1])
    cap.set(cv2.cv.cv_cap_prop_fps, state.resolution.fps)
    return cap


def cap_file(file_name):
    cap = cv2.VideoCapture(file_name)
    return cap


def make_modifier(index, lower_or_upper):
    def inner(x):
        if lower_or_upper:
            lower_blue[index] = x
        else:
            upper_blue[index] = x
    return inner


win = cv.NamedWindow("image")
cv.CreateTrackbar("h_low", "image", 0, 255, make_modifier(0, True))
cv.CreateTrackbar("h_high", "image", 150, 255, make_modifier(0, False))
cv.CreateTrackbar("s_low", "image", 0, 255, make_modifier(1, True))
cv.CreateTrackbar("s_high", "image", 157, 255, make_modifier(1, False))
cv.CreateTrackbar("v_low", "image", 152, 255, make_modifier(2, True))
cv.CreateTrackbar("v_high", "image", 255, 255, make_modifier(2, False))

lower_blue = np.array([
    cv.GetTrackbarPos("h_low", "image"),
    cv.GetTrackbarPos("s_low", "image"),
    cv.GetTrackbarPos("v_low", "image")
])
upper_blue = np.array([
    cv.GetTrackbarPos("h_high", "image"),
    cv.GetTrackbarPos("s_high", "image"),
    cv.GetTrackbarPos("v_high", "image")
])


def threshold_hsv(img):
    mask = cv2.inRange(img, lower_blue, upper_blue)
    return cv2.bitwise_and(img, img, mask=mask)

cap = cap_file("./white_cylinder.mjpeg")

while(True):
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    result = threshold_hsv(hsv)

    # Display the resulting frame
    cv2.imshow('original', frame)
    cv2.imshow('image', result)

    # Wait for key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
