from instrumentation import Timed


class Heuristic(Timed):

    def run(self, detection, detector_state):
        self.start_timing()
        mask = self.filter(detection, detector_state)
        self.end_timing()
        return mask

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
        super(AbnormalSizeHeuristic, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def filter(self, detection, detector_state):
        return [self.min_size < blob.size < self.max_size
                for blob in detection]


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
        if len(detection) == 0:
            return []

        index, largest_blob = max(enumerate(detection),
                                  key=lambda x: x[1].size)
        return [True if blob is largest_blob
                else False
                for blob in detection]
