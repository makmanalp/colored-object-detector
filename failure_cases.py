
class FailureCase(object):
    """Defines a possible failure case in the detection code, by inspecting the
    detection results, and possibly attempts to alleviate it by adjusting
    settings."""

    def test(self, detector_state):
        raise NotImplementedError()

    def fix(self):
        raise NotImplementedError()


class TooManyNonFilteredFailure(FailureCase):
    """If there are a few, they can be filtered. If they are many, then maybe
    it's just way wrong."""

    def test(self, detector_state):
        pass

    def fix(self):
        # Filter regions
        pass


class TooManyResultsFailure(FailureCase):
    """You'd expect only one positive result. """

    def test(self, detector_state):
        pass

    def fix(self):
        # Readjust brightness
        pass
    pass


class TimingFailure(FailureCase):
    """Detection took too long or too short."""

    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def test(self, detector_state):
        return detector_state.last_detection.since > self.threshold

    def fix(self):
        """Reduce resolution?"""
        # detector_state.resolution = blah
        pass


class VarianceFailure(FailureCase):
    """Detected object's location varies too much."""

    def test(self, detector_state):
        pass

    def fix(self):
        pass
