import cv2 as cv
import numpy as np
from basic_functions.read_data import read_img, read_video
from basic_functions.transformations import translate, rotate
from basic_functions.helpers import show_img

if __name__ == "__main__":
    img_path = "data/images/city.jpg"

    # Reading image
    img = cv.imread(img_path)

    # Splitting into 3 colors
    b, g, r = cv.split(img)

    # distinct_colors = np.hstack((b, g, r))
    distinct_colors = cv.resize(distinct_colors, (1200, 400))
    show_img(distinct_colors, "Distinct colors")

    # Merging it back together
    img_merged = cv.merge((b, g, r))
    show_img(img_merged, "Merged")

    # Adding to images
    # Saturation of 2 images
    bg_add = cv.add(b, g)
    show_img(bg_add, "Blue and Green")

    gr_add = cv.add(g, r)
    show_img(gr_add, "Green and Red")

    # Weighted adding
    # result = img_1 * alpha + img_2 * beta + gamma
    bg_add_weighted = cv.addWeighted(b, .2, r, .5, 0)
    show_img(bg_add_weighted, "Weighted Addition")

    # Translation of an image
    translated = translate(img, -100, 23)
    show_img(translated, "Translated")

    # Rotation of an image
    rotated = rotate(img, 23)
    show_img(rotated, "Rotated")
