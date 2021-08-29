import cv2 as cv
import numpy as np
from basic_functions.read_data import read_img, read_video
from basic_functions.helpers import show_img

if __name__ == "__main__":
    img_path = "data/images/city.jpg"

    # Reading image
    img = cv.imread(img_path)

    # Blurring
    blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)

    img_blur = np.hstack((img, blur))
    show_img("Image and Blur image", img_blur)

    # Edge Cascade
    canny_img = cv.Canny(img, 125, 175)
    canny_blur = cv.Canny(blur, 125, 175)

    canny = np.hstack((canny_img, canny_blur))
    show_img("Canny edges on Image and Blur", canny)

    # Image dilation
    dilated = cv.dilate(canny_blur, (7, 7), iterations=3)

    canny_dilated = np.hstack((canny_blur, dilated))
    show_img("Dilated Canny edges", canny_dilated)

    # Image erosion
    eroded = cv.erode(dilated, (7, 7), iterations=3)

    results = np.hstack((np.hstack((canny_blur, dilated)), eroded))
    show_img("Blur - Dilated - Eroded", results)
