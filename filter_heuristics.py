from instrumentation import Timed
from collections import OrderedDict
import math

import cv2.cv as cv
import cv2

import matplotlib.pyplot as plt
plt.ion()

COLORS = [
    (178,223,138),
    (51,160,44),
    (251,154,153),
    (227,26,28),
    (253,191,111),
    (255,127,0),
    (202,178,214),
    (106,61,154),
    (255,255,153),
    (31,120,180),
    (166,206,227),
    (177,89,40),
]


class HeuristicStack(object):

    def __init__(self, heuristics, threshold=None):
        self.heuristics = heuristics

        total_weight = 0.0
        for i, (heuristic, weight) in enumerate(self.heuristics):
            heuristic.stack = self
            heuristic.color = COLORS[i]
            total_weight += weight

        if threshold is None:
            self.threshold = total_weight
        else:
            self.threshold = threshold

        self.display_window()

    def display_window(self):
        cv.NamedWindow("heuristics")
        cv.MoveWindow("heuristics", 20, 20)
        cv.ResizeWindow("heuristics", 640, 480)
        for heuristic, _ in self.heuristics:
            name = heuristic.__class__.__name__
            cv.CreateTrackbar(name, "heuristics", 0, 1, self.make_set_debug(heuristic))
            cv.CreateTrackbar("weights_" + name, "heuristics", 0, 1, self.make_set_weights(heuristic))

    def make_set_debug(self, heuristic):
        def set_debug(value):
            heuristic.debug = (True if value == 1 else False)
        return set_debug

    def make_set_weights(self, heuristic):
        def set_weights(value):
            heuristic.debug_print_weights = (True if value == 1 else False)
        return set_weights

    def print_heuristic_result(self, heuristic, heuristic_result):
        selected = sum(1 for x in heuristic_result if x >= 1.0)
        print "{name}: Selected: {selected} Filtered: {filtered}"\
            .format(name=heuristic.__class__.__name__, selected=selected,
                    filtered=len(heuristic_result) - selected)

    def get_weighted_result(self, detection, detector_state):

        heuristic_values = OrderedDict()
        for heuristic, weight in self.heuristics:

            heuristic_result = heuristic.run(detection, detector_state)
            heuristic_name = heuristic.__class__.__name__
            heuristic_values[heuristic_name] = [weight * x for x in heuristic_result]

            heuristic.print_time()
            self.print_heuristic_result(heuristic, heuristic_result)

        heuristic_total = map(sum, zip(*heuristic_values.values()))

        return heuristic_total


class Heuristic(Timed):

    def __init__(self, *args, **kwargs):
        super(Heuristic, self).__init__()
        self.debug = kwargs.get("debug", False)
        self.debug_print_weights = self.debug

    def run(self, detection, detector_state):
        self.start_timing()
        mask = self.filter(detection, detector_state)
        if self.debug:
            for blob, weight in zip(detection.blobs, mask):
                if weight >= 1.0:
                    cv.Circle(cv.fromarray(detector_state.current_image),
                              (int(blob.x), int(blob.y)),
                              int(math.ceil(blob.size)), (0, 255, 0),
                              thickness=1, lineType=8, shift=0)
                else:
                    cv.Rectangle(cv.fromarray(detector_state.current_image),
                                 (int(blob.x - blob.size), int(blob.y - blob.size)),
                                 (int(blob.x + blob.size), int(blob.y + blob.size)),
                                 (0, 0, 255),
                                 thickness=1, lineType=8, shift=0)

                if self.debug_print_weights:
                    cv2.putText(detector_state.current_image,
                                "{0:.2f}".format(weight), (int(blob.x),
                                                           int(blob.y)),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
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

    def __init__(self, min_size=4.0, max_size=16.0, **kwargs):
        super(PhysicalSizeHeuristic, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, detection, detector_state):

        def weight(x):
            if self.min_size <= x <= self.max_size:
                return 1.0
            elif x < self.min_size:
                return 0.0
            elif x > self.max_size:
                # Penalize larger blobs inversely proportionally to how large
                # they are compared to the upper limit
                return self.max_size / x
            else:
                raise ValueError(x)

        return [weight(blob.real_size) for blob in detection]


class NormalBlobSizeHeuristic(Heuristic):
    """Filter blobs that are just too large or too small to be realistic."""

    def __init__(self, min_size=6, max_size=40, **kwargs):
        super(NormalBlobSizeHeuristic, self).__init__(**kwargs)
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, detection, detector_state):

        def highpass_filter(x):
            if self.min_size <= x <= self.max_size:
                return 1.0
            elif x < self.min_size:
                return 0.0
            elif x > self.max_size:
                # Penalize larger blobs inversely proportionally to how large
                # they are compared to the upper limit
                return self.max_size / x
            else:
                raise ValueError(x)

        return [highpass_filter(blob.size) for blob in detection]


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


class DensityHeuristic(Heuristic):
    """Distinguish between solid filled blobs versus blobs that are composed
    mostly of speckle. Solid is better. Ratio of non-zero pixels in area to
    radius."""

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
        return [1.0 if blob is largest_blob
                else 0.0
                for blob in detection]
# cam: v4l2-ctl -d /dev/video1 -c exposure_auto=1,focus_auto=0,white_balance_temperature_auto=0,brightness=128,contrast=128,saturation=128,focus_absolute=0
