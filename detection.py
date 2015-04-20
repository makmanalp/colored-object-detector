import time
from collections import namedtuple


STALENESS_THRESHOLD = 1  # seconds

Blob = namedtuple('Blob', ['x', 'y', 'size'])


class Detection(object):

    def __init__(self, blobs=[]):
        self.blobs = blobs
        self.time = time.time()
        self.chosen_blob = None

    def __iter__(self):
        return iter(self.blobs)

    @property
    def is_stale(self):
        return time.time() - self.time > STALENESS_THRESHOLD

    @property
    def since(self):
        return time.time() - self.time
