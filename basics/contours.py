import cv2 as cv
import numpy as np
from basic_functions.helpers import show_img


if __name__ == "__main__":
    img_path = "data/images/city.jpg"

    # Reading image as gray scale
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Empty image to draw contours
    blank = np.zeros(img.shape, dtype='uint8')

    # Exerting Gaussian blur
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    # Canny for edge detection
    canny = cv.Canny(blur, 125, 175)

    ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
    cv.imshow('Thresh', thresh)

    contours, hierarchies = cv.findContours(thresh,
                                            cv.RETR_LIST,
                                            cv.CHAIN_APPROX_SIMPLE)

    print(f"Founded countours: {len(contours)}")

    cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
    show_img(blank)
