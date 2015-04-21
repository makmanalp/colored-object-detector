import numpy as np
import cv2
import cv2.cv as cv
from collections import OrderedDict
import math

from detector_state import DetectorState
from detection import Detection, Blob

from filter_heuristics import (AbnormalSizeHeuristic, LargestHeuristic)

"""
Ideas:
    - Do tracking to have best case heuristic of spot to look at.
    - Camshift: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift

Heuristics:
    - Return largest - there is only one dot
    - Maybe do blob detection before-after

"""


detector_state = DetectorState((640, 480), 128, 5)


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

blob_params = cv2.SimpleBlobDetector_Params()
blob_params.minThreshold = 0
blob_params.maxThreshold = 255
blob_params.filterByArea = True
blob_params.minArea = 10
blob_detector = cv2.SimpleBlobDetector(blob_params)

cap = cap_file("./white_cylinder.mjpeg")

heuristics = [
    AbnormalSizeHeuristic(),
    LargestHeuristic()
]

while(True):

    # Get a frame
    ret, frame = cap.read()
    if frame is None:
        break

    # Threshold it
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = threshold_hsv(hsv)
    detector_state.current_image = result

    # Run detection
    # TODO: find a way around this conversion
    thresh = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    #_, thresh = cv2.threshold(result, 1, 254, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for contour in contours:
        (x, y), size = cv2.minEnclosingCircle(contour)
        blobs.append(Blob(x, y, int(math.ceil(size))))

    for blob in blobs:
        if 100 > blob.size > 8:
            cv.Circle(cv.fromarray(frame), (int(blob.x), int(blob.y)), blob.size, (0, 0, 255),
                      thickness=1, lineType=8, shift=0)
        else:
            cv.Circle(cv.fromarray(result), (int(blob.x), int(blob.y)), blob.size, (0, 255, 0),
                      thickness=1, lineType=8, shift=0)


    # areas = [cv2.contourArea(c) for c in contours]
    # max_index = np.argmax(areas)
    # cnt=contours[max_index]

    #cv2.drawContours(result, contours, -1, (255, 0, 0), 3)

    # Run Filter Heuristics
    detection = Detection(blobs)
    heuristic_stack = OrderedDict()
    for heuristic in heuristics:
        heuristic_result = heuristic.run(detection, detector_state)
        heuristic.print_time()
        heuristic_stack[heuristic.__class__.__name__] = sum(heuristic_result)

    print heuristic_stack


    # detection.chosen_blob = chosen

    # Update State
    detector_state.update_detections(detection)

    # Check Failure Cases

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
