# TODO: show graphs of heuristic stack etc

import argparse
import numpy as np
import cv2
import cv2.cv as cv
import math
import json
import sys
from subprocess import call

from detector_state import DetectorState
from detection import Detection, Blob

from failure_cases import (TooManyResultsFailure)
from filter_heuristics import (NormalBlobSizeHeuristic, LargestHeuristic, PhysicalSizeHeuristic, DensityHeuristic2, DensityHeuristic3,
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
parser.add_argument('--disable-cam-settings', dest='disable_cam_settings', action="store_true",
                    help='Disable automatic setting of camera settings with v4l2-ctl')
args = parser.parse_args()

if args.cam:
    if not args.disable_cam_settings:
        command = ["v4l2-ctl", "-d", "/dev/video1", "-c",
                   "exposure_auto=1,focus_auto=0,white_balance_temperature_auto=0,brightness=128,contrast=128,saturation=128,focus_absolute=0,exposure_absolute=50"]
        ret_val = call(" ".join(command), shell=True)
        if ret_val != 0:
            print "Giving up trying to set camera settings. Retry a few times, fix it or run with --disable-cam-settings"
            sys.exit(1)
    cap = cap_camera(int(args.cam))
elif args.file:
    cap = cap_file(args.file)
else:
    raise ValueError("Must specify either --cam or --file!")



detector_state = DetectorState((640, 480), 128, 5,
                               approx_object_size=15.0)



from collections import OrderedDict
import threshold

d = OrderedDict([('h_low', 0), ('h_high', 254), ('s_low', 0), ('s_high', 254), ('v_low', 0), ('v_high', 254)])
white_sample = threshold.Threshold("white_sample", d)


cv.NamedWindow("image", flags=cv.CV_WINDOW_NORMAL)
cv.MoveWindow("image", 20, 20)
cv.ResizeWindow("image", 700, 700)

cv.NamedWindow("original", flags=cv.CV_WINDOW_NORMAL)
cv.MoveWindow("original", 650, 20)
cv.ResizeWindow("original", 640, 480)

cv.NamedWindow("heuristics", flags=cv.CV_WINDOW_NORMAL)
cv.MoveWindow("heuristics", 400, 400)
cv.ResizeWindow("heuristics", 640, 480)


heuristics = HeuristicStack({
    (PhysicalSizeHeuristic(debug=False), 1.0),
    #(LargestHeuristic(debug=False), 0.5),
    (NormalBlobSizeHeuristic(debug=False), 0.5),
    (DensityHeuristic2(debug=False), 0.5)
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
    result = white_sample.threshold(hsv)
    detector_state.current_image = result

    # Run detection
    # TODO: find a way around this conversion
    thresh = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    detector_state.thresholded_image = thresh
    #_, thresh = cv2.threshold(result, 1, 254, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for i, contour in enumerate(contours):

        # If this is not a top-level blob, skip it
        parent = hierarchy[0][i][3]
        if not parent == -1:
            continue

        (x, y), size = cv2.minEnclosingCircle(contour)

        # diameter is a more accurate measure of blob size than radius
        size = size * 2
        area = cv2.contourArea(contour)

        # Calculate real-world measurements
        real_x, real_y = detector_state.find_blob_distance(x, y)
        real_size = detector_state.find_blob_size(size, real_y)

        # Calculate the area of holes in the contour
        #contour_hole_area = sum([cv2.contourArea(cont)
        #                         for j, cont in enumerate(contours)
        #if hierarchy[0][j][3] == i])
        contour_hole_area = None

        blobs.append(Blob(x, y, contour, contour_hole_area, size, area, real_x,
                          real_y, real_size))

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
        # Update State
        detector_state.update_detections(detection)

    if not must_fail:
        for blob in detection.chosen_blobs:
            cv.Circle(cv.fromarray(frame),
                      (int(blob.x), int(blob.y)),
                      int(math.ceil(blob.size)),
                      (0, 0, 255),
                      thickness=2, lineType=8, shift=0)

            # Select smallest blob
            # TODO: make sure we send only ONE chosen blob
            #smallest_blob = min(detection, key=lambda x: x.real_y)
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
