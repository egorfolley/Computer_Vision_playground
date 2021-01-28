import cv2 as cv
import numpy as np
from basic_functions.read_data import read_img, read_video

if __name__ == "__main__":
    img_path = "data/images/cat.jpg"
    vid_path = "data/videos/dog.mp4"

    read_img(img_path, scale=0.5, concat_gray=True)
    read_video(vid_path, scale=0.2, concat_gray=True)
