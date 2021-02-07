import cv2 as cv
import numpy as np
from basic_functions.helpers import show_img


if __name__ == "__main__":
    img_path = "data/images/cat.jpg"
    img = cv.imread(img_path)

    # Averaging
    average = cv.blur(img, (3,3))
    img_average = np.hstack((img, average))

    show_img(img_average, "Image & Averaging")

    # Gaussian Blur
    gauss = cv.GaussianBlur(img, (5,5), 0)
    img_gauss = np.hstack((img, gauss))

    show_img(img_gauss, "Image & GaussianBlur")

    # Median Blur
    median = cv.medianBlur(img, 7)
    img_median = np.hstack((img, median))

    show_img(img_median, "Image & MedianBlur")

    # Bilateral
    bilateral = cv.bilateralFilter(img, 10, 35, 25)
    img_bilateral = np.hstack((img, bilateral))

    show_img(img_bilateral, "Image & Bilateral")
