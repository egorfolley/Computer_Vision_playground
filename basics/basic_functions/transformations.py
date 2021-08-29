import cv2 as cv
import numpy as np

# X, Y
# (-x, -y, x, y)
# (left, up, right, down)

def translate(img, x: int, y: int):
    translation_matrix = np.float32([[1,0,x],
                                     [0,1,y]])
    dims = (img.shape[1], img.shape[0])

    return cv.warpAffine(img, translation_matrix, dims)

def rescale_frame(frame, scale=0.75):
    height, width, _ = frame.shape # rows, cols, channels
    height *= scale
    width *= scale

    dimesions = (int(width), int(height))

    return cv.resize(frame, dimesions, interpolation=cv.INTER_AREA)

def rotate(img, angle, rot_point = None):
    (height, width, _) = img.shape

    if not rot_point:
        rot_point = (width//2, height//2)

    rot_matrix = cv.getRotationMatrix2D(rot_point, angle, 1.0)
    dims = (width, height)

    return cv.warpAffine(img, rot_matrix, dims)
