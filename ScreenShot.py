import cv2
import numpy
from mss import mss

ScreenX = 1600
ScreenY = 900
window_size = (
    int(ScreenX / 2 - 200),
    int(ScreenY / 2 - 200),
    int(ScreenX / 2 + 200),
    int(ScreenY / 2 + 200))

Screenshot_value = mss()


def screenshot():
    img = Screenshot_value.grab(window_size)
    img = numpy.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

# while True:
#     cv2.imshow('a', numpy.array(screenshot()))
#     cv2.waitKey(1)
