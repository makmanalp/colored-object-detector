
class DetectorState(object):

    def __init__(self, resolution, brightness, fps):
        self.detections = []
        self.resolution = None
        self.brightness = None
        self.fps = None
        self.current_image = None

    def update_detections(self, detection):
        self.detections.append(detection)

    @property
    def last_detection(self):
        return self.detections[:-1]

    @property
    def current_detection(self):
        if self.last_detection.stale:
            return None
        else:
            return self.last_detection
