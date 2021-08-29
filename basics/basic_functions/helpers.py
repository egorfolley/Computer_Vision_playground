import cv2 as cv
import numpy as np

def show_img(img, img_name: str = "Image"):
    cv.imshow(img_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
