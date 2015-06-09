import cv2
import cv2.cv as cv
import numpy as np

from collections import OrderedDict


class Threshold(object):

    def __init__(self, name, start_values):

        self.name = name

        cv.NamedWindow(name, flags=cv.CV_WINDOW_NORMAL)
        cv.MoveWindow(name, 20, 20)

        assert len(start_values) == 6

        self.values = OrderedDict()
        for channel, value in start_values.items():
            self.values[channel] = value
            cv.CreateTrackbar(channel, name, value, 255,
                              self.make_modifier(channel))

    def make_modifier(self, bar_name):
        def inner(x):
            self.values[bar_name] = x
        return inner

    def threshold(self, img):
        lower = np.array([
            self.values["h_low"],
            self.values["s_low"],
            self.values["v_low"],
        ])
        upper = np.array([
            self.values["h_high"],
            self.values["s_high"],
            self.values["v_high"],
        ])

        mask = cv2.inRange(img, lower, upper)
        return cv2.bitwise_and(img, img, mask=mask)
