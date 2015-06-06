import time
from collections import namedtuple


STALENESS_THRESHOLD = 1  # seconds

Blob = namedtuple('Blob', ['x', 'y', 'contour', 'contour_hole_area', 'size', 'area', 'real_x', 'real_y', 'real_size'])


class Detection(object):

    def __init__(self, blobs=[]):
        self.blobs = blobs
        self.time = time.time()
        self.chosen_blobs = []

    def __iter__(self):
        return iter(self.blobs)

    def __len__(self):
        return len(self.blobs)

    @property
    def is_stale(self):
        return time.time() - self.time > STALENESS_THRESHOLD

    @property
    def since(self):
        return time.time() - self.time
