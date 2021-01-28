import cv2 as cv
import numpy as np


def rescale_frame(frame, scale=0.75):
    height, width, _ = frame.shape # rows, cols, channels
    height *= scale
    width *= scale

    dimesions = (int(width), int(height))

    return cv.resize(frame, dimesions, interpolation=cv.INTER_AREA)
