
class FailureCase(object):
    """Defines a possible failure case in the detection code, by inspecting the
    detection results, and possibly attempts to alleviate it by adjusting
    settings."""

    def get_failure_text(self):
        return "> Failure Case: {}".format(self.__class__.__name__)

    def test(self, detection, detector_state):
        raise NotImplementedError()

    def fix(self):
        raise NotImplementedError()


class TooManyNonFilteredFailure(FailureCase):
    """If there are a few, they can be filtered. If they are many, then maybe
    it's just way wrong."""

    def test(self, detection, detector_state):
        pass

    def fix(self):
        # Filter regions
        pass


class TooManyResultsFailure(FailureCase):
    """You'd expect only one positive result. """

    def __init__(self, max_results=10):
        self.max_results = max_results
        self.failure = None

    def get_failure_text(self):
        return "> Failure Case: {} found too many results: {}.".format(self.__class__.__name__, self.num_results)

    def test(self, detection, detector_state):
        self.num_results = len(detection.chosen_blobs)
        return self.num_results > self.max_results

    def fix(self):
        # Readjust brightness
        pass
    pass


class TimingFailure(FailureCase):
    """Detection took too long or too short."""

    def __init__(self, threshold=0.25):
        self.threshold = threshold

    def test(self, detection, detector_state):
        return detector_state.last_detection.since > self.threshold

    def fix(self):
        """Reduce resolution?"""
        # detector_state.resolution = blah
        pass


class VarianceFailure(FailureCase):
    """Detected object's location varies too much."""

    def test(self, detection, detector_state):
        pass

    def fix(self):
        pass
