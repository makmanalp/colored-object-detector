import time
from collections import deque


class Timed(object):

    def __init__(self):
        self.timings = deque([], maxlen=100)

    @property
    def average_time(self):
        return float(sum(self.timings)) / len(self.timings)

    def print_time(self):
        print "{} took {}".format(self.__class__.__name__, self.average_time)

    def start_timing(self):
        self.start_time = time.time()

    def end_timing(self):
        assert self.start_time is not None
        self.timings.append(time.time() - self.start_time)
        self.start_time = None
