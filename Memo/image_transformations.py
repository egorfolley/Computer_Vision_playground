import cv2 as cv
import numpy as np
from basic_functions.read_data import read_img, read_video
from basic_functions.helpers import show_img

if __name__ == "__main__":
    img_path = "data/images/city.jpg"

    # Reading image
    img = cv.imread(img_path)
