# TODO: show graphs of heuristic stack etc

import numpy as np
import cv2
import cv2.cv as cv
import math

from detector_state import DetectorState
from detection import Detection, Blob

from failure_cases import (TooManyResultsFailure)
from filter_heuristics import (NormalBlobSizeHeuristic, LargestHeuristic,
                               HeuristicStack)

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

cap = cap_file("./sun_shade_grass-cylinder.mjpeg")

heuristics = HeuristicStack({
    (NormalBlobSizeHeuristic(), 1.0),
    (LargestHeuristic(), 1.0)
})

failure_cases = [
    TooManyResultsFailure(max_results=4),
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
        area = cv2.contourArea(contour)
        blobs.append(Blob(x, y, size, area))
    detection = Detection(blobs)

    #cv2.drawContours(result, contours, -1, (255, 0, 0), 3)

    # Run Filter Heuristics
    heuristics.get_weighted_result(detection, detector_state)

    # Distill heuristics into a detection
    # detection.chosen_blob = chosen
    for blob in detection:
        if 100 > blob.size > 6:
            detection.chosen_blobs.append(blob)


    # Check Failure Cases
    for failure_case in failure_cases:
        must_fail = failure_case.test(detection, detector_state)
        if must_fail:
            failure_text = failure_case.get_failure_text()
            print failure_text
            cv2.putText(frame, failure_text, (10, 20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
            # TODO: handle fixing case
            break
    else:
        print "> Success!"
        # Update State
        detector_state.update_detections(detection)

    for blob in detection:
        if not must_fail:
            if blob in detection.chosen_blobs:
                cv.Circle(cv.fromarray(frame), (int(blob.x), int(blob.y)), int(math.ceil(blob.size)), (0, 0, 255),
                          thickness=2, lineType=8, shift=0)
            else:
                cv.Circle(cv.fromarray(result), (int(blob.x), int(blob.y)), int(math.ceil(blob.size)), (0, 255, 0),
                          thickness=1, lineType=8, shift=0)
        else:
            cv.Circle(cv.fromarray(result), (int(blob.x), int(blob.y)), int(math.ceil(blob.size)), (0, 255, 0),
                      thickness=1, lineType=8, shift=0)



    # Display the resulting frame
    cv2.imshow('original', frame)
    cv2.imshow('image', result)

    # Wait for key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == 32:
        k = cv2.waitKey() & 0xFF

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
