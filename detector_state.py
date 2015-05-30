import math

class DetectorState(object):

    def __init__(self, resolution, brightness, fps, camera_angle=68.0,
                 camera_height=70.0, camera_vertical_fov=43.30,
                 camera_horizontal_fov=70.42, approx_object_size=15.0):

        self.detections = []
        self.resolution = None
        self.brightness = None
        self.fps = None
        self.current_image = None
        self.camera_angle = camera_angle
        self.camera_height = camera_height
        self.camera_horizontal_fov = camera_horizontal_fov
        self.camera_vertical_fov = camera_vertical_fov
        self.approx_object_size = approx_object_size

    @property
    def image_size(self):
        """return width, height"""
        return self.current_image.shape[1], self.current_image.shape[0]

    @property
    def focal_length(self):
        height = self.image_size[1]
        return height / (2.0 * math.tan(math.radians(self.camera_vertical_fov / 2.0)))

    def find_blob_distance(self, pixel_y):
        image_height = self.image_size[1]
        pixel_resolution = self.camera_vertical_fov / float(image_height)
        image_center = image_height / 2.0
        pixel_angle = (image_center - pixel_y) * pixel_resolution
        theta_pixel = self.camera_angle + pixel_angle
        return self.camera_height * math.tan(math.radians(theta_pixel))

    def find_blob_size(self, size, real_y):
        return (size * real_y) / self.focal_length

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
