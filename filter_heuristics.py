
class Heuristic(object):

    def filter(self, detection, detector_state):
        """Takes a detection and returns a True / False mask that filters out
        blobs based on the heuristic measure."""
        raise NotImplementedError()


class PhysicalSizeHeuristic(Heuristic):
    """From camera angle, and using the flat ground assumption, calculate
    approximate distance of a blob and filter those above the size threshold
    for their distance."""

    def filter(self, detection, detector_state):
        pass


class AbnormalSizeHeuristic(Heuristic):
    """Filter blobs that are just too large or too small to be realistic."""

    def __init__(self, min_size=2, max_size=100):
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, detection, detector_state):
        pass


class ImageDistanceHeuristic(Heuristic):
    """Make farther blobs to the last accepted detection less likely to be
    real. Unless last detection was a long time ago."""

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
        index, largest_blob = max(enumerate(detection),
                                  key=lambda x: x[1].size)
        return [True if blob is largest_blob
                else False
                for blob in detection]
