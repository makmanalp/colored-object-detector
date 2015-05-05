from instrumentation import Timed
from collections import OrderedDict
import math

import cv2.cv as cv

COLORS = [
    (31,120,180),
    (166,206,227),
    (178,223,138),
    (51,160,44),
    (251,154,153),
    (227,26,28),
    (253,191,111),
    (255,127,0),
    (202,178,214),
    (106,61,154),
    (255,255,153),
    (177,89,40),
]


class HeuristicStack(object):

    def __init__(self, heuristics):
        self.heuristics = heuristics
        for i, (heuristic, weight) in enumerate(self.heuristics):
            heuristic.stack = self
            heuristic.color = COLORS[i]

    def print_heuristic_result(self, heuristic, heuristic_result):
        selected = sum(heuristic_result)
        print "{name}: Selected: {selected} Filtered: {filtered}"\
            .format(name=heuristic.__class__.__name__, selected=selected,
                    filtered=len(heuristic_result) - selected)

    def get_weighted_result(self, detection, detector_state):

        heuristic_values = OrderedDict()
        for heuristic, weight in self.heuristics:

            heuristic_result = heuristic.run(detection, detector_state)
            heuristic_name = heuristic.__class__.__name__
            heuristic_values[heuristic_name] = [weight * 1.0 if x else 0.0 for x in heuristic_result]

            heuristic.print_time()
            self.print_heuristic_result(heuristic, heuristic_result)

        heuristic_total = map(sum, zip(*heuristic_values.values()))

        return heuristic_total


class Heuristic(Timed):

    def __init__(self, *args, **kwargs):
        super(Heuristic, self).__init__()
        self.debug = kwargs.get("debug", False)

    def run(self, detection, detector_state):
        self.start_timing()
        mask = self.filter(detection, detector_state)
        if self.debug:
            for blob, selected in zip(detection.blobs, mask):
                if selected:
                    cv.Circle(cv.fromarray(detector_state.current_image),
                              (int(blob.x), int(blob.y)),
                              int(math.ceil(blob.size)), self.color,
                              thickness=2, lineType=8, shift=0)
        self.end_timing()
        return mask

    def filter(self, detection, detector_state):
        """Takes a detection and returns a True / False mask that filters out
        blobs based on the heuristic measure."""
        raise NotImplementedError()


class PhysicalSizeHeuristic(Heuristic):
    """From camera angle (degrees, angles from downward vector), and using the
    flat ground assumption, calculate approximate distance of a blob and filter
    those above the size threshold for their distance."""

    def __init__(self, min_size=5, max_size=20, camera_angle=60.0,
                 camera_height=100.0, **kwargs):
        super(PhysicalSizeHeuristic, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size
        self.camera_angle = math.radians(camera_angle)
        self.camera_height = camera_height

    def filter(self, detection, detector_state):
        image_height = detector_state.current_image.shape[0]

        return [self.min_size
                < ((float(image_height) / blob.y) * blob.area)
                < self.max_size for blob in detection]


class NormalBlobSizeHeuristic(Heuristic):
    """Filter blobs that are just too large or too small to be realistic."""

    def __init__(self, min_size=6, max_size=400, **kwargs):
        super(NormalBlobSizeHeuristic, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, detection, detector_state):
        return [self.min_size < blob.area < self.max_size
                for blob in detection]


class ImageDistanceHeuristic(Heuristic):
    """Make farther blobs to the last accepted detection less likely to be
    real. Unless last detection was a long time ago."""

    def filter(self, detection, detector_state):
        pass


class MostDifferentColorHeuristic(Heuristic):
    """
    Sort by color intensity maybe?

    - Historical average of hsv colors in detected blobs
    - Closest to target HSV value
    """

    def filter(self, detection, detector_state):
        pass


class TrackingHeuristic(Heuristic):
    """Run CamShift on the tracked image and detect variance from result
    TODO"""

    def filter(self, detection, detector_state):
        pass


class LargestHeuristic(Heuristic):
    """Largest blob gets a bonus"""

    def filter(self, detection, detector_state):
        if len(detection) == 0:
            return []

        index, largest_blob = max(enumerate(detection),
                                  key=lambda x: x[1].area)
        return [True if blob is largest_blob
                else False
                for blob in detection]
