# TODO: show graphs of heuristic stack etc

import argparse
import numpy as np
import cv2
import cv2.cv as cv
import math
import json

from detector_state import DetectorState
from detection import Detection, Blob

from failure_cases import (TooManyResultsFailure)
from filter_heuristics import (NormalBlobSizeHeuristic, LargestHeuristic, PhysicalSizeHeuristic,
                               HeuristicStack)

import zmq
context = zmq.Context()
publisher = context.socket(zmq.PUB)
publisher.bind("tcp://0.0.0.0:5561")


"""
Ideas:
    - Do tracking to have best case heuristic of spot to look at.
    - Camshift: http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_video/py_meanshift/py_meanshift.html#meanshift

Heuristics:
    - Maybe do blob detection before-after

"""

def cap_camera(cam_id):
    cap = cv2.VideoCapture(cam_id)
    try:
        cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CV_CAP_PROP_FPS, 5)
    except AttributeError as ex:
        print ex
        print "Could not set resolution / fps attributes, continuing ..."
    return cap


def cap_file(file_name):
    cap = cv2.VideoCapture(file_name)
    return cap

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--file', dest='file',
                    help='Get video from a file path.')
parser.add_argument('--cam', dest='cam',
                    help='Get video from a camera id (numeric).')
args = parser.parse_args()

if args.cam:
    cap = cap_camera(int(args.cam))
elif args.file:
    cap = cap_file(args.file)
else:
    raise ValueError("Must specify either --cam or --file!")


detector_state = DetectorState((640, 480), 128, 5, approx_object_size=15.0)




def make_modifier(index, lower_or_upper):
    def inner(x):
        if lower_or_upper:
            lower_blue[index] = x
        else:
            upper_blue[index] = x
    return inner


cv.NamedWindow("image", flags=cv.CV_WINDOW_NORMAL)
cv.MoveWindow("image", 20, 20)
cv.ResizeWindow("image", 640, 480)
cv.CreateTrackbar("h_low", "image", 0, 255, make_modifier(0, True))
cv.CreateTrackbar("h_high", "image", 150, 255, make_modifier(0, False))
cv.CreateTrackbar("s_low", "image", 0, 255, make_modifier(1, True))
cv.CreateTrackbar("s_high", "image", 157, 255, make_modifier(1, False))
cv.CreateTrackbar("v_low", "image", 152, 255, make_modifier(2, True))
cv.CreateTrackbar("v_high", "image", 255, 255, make_modifier(2, False))

cv.NamedWindow("original", flags=cv.CV_WINDOW_NORMAL)
cv.MoveWindow("original", 650, 20)
cv.ResizeWindow("original", 640, 480)

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

heuristics = HeuristicStack({
    (PhysicalSizeHeuristic(debug=False), 1.0),
    (LargestHeuristic(debug=False), 0.5),
    (NormalBlobSizeHeuristic(debug=False, min_size=1, max_size=400), 0.5)
})

failure_cases = [
    TooManyResultsFailure(max_results=4),
]

paused = False
_, saved_frame = cap.read()
while(True):

    # Get a frame
    if not paused:
        _, frame = cap.read()
        if frame is None and args.file:
            cap = cap_file(args.file)
            _, frame = cap.read()
        saved_frame = frame.copy()
    else:
        frame = saved_frame.copy()

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
        real_x, real_y = detector_state.find_blob_distance(x, y)
        real_size = detector_state.find_blob_size(size, real_y)
        blobs.append(Blob(x, y, size, area, real_x, real_y, real_size))
    detection = Detection(blobs)

    # Run Filter Heuristics
    heuristic_total = heuristics.get_weighted_result(detection, detector_state)

    # Distill heuristics into a detection
    # detection.chosen_blob = chosen
    for heuristic_weight, blob in zip(heuristic_total, detection):
        if heuristic_weight >= heuristics.threshold:
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

    if not must_fail:
        for blob in detection:
            if blob in detection.chosen_blobs:
                cv.Circle(cv.fromarray(frame),
                          (int(blob.x), int(blob.y)),
                          int(math.ceil(blob.size)),
                          (0, 0, 255),
                          thickness=2, lineType=8, shift=0)

                # Select smallest blob
                # TODO: make sure we send only ONE chosen blob
                #smallest_blob = min(detection, key=lambda x: x.real_y)
                #print list(current_detection)
                print blob.real_x, blob.real_y
                msg = [{"name": "sample!", "x": blob.real_y, "y":
                        -blob.real_x}]
                publisher.send(json.dumps(msg))


    # Display the resulting frame
    cv2.imshow('original', frame)
    cv2.imshow('image', result)

    # Wait for key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    if k == 32:
        paused = not paused

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
