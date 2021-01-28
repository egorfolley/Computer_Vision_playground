import cv2 as cv
import numpy as np
from basic_functions.rescaling import rescale_frame


def read_img(img_path: str,
             scale: float = 0.0,
             concat_gray: bool = False) -> None:
    '''
        Reading and showing image function
        Args:
            img_path: str - path to your image data
            scale: float - in order to rescale your image
            concat_gray: bool - if requires to show RGB and GRAY
                                in one window
    '''
    img = cv.imread(img_path)

    if scale:
        img = rescale_frame(img, scale)

    if concat_gray:
        # This method firstly transforms BGR to GRAY - 3D -> 1D
        # and then from 1D -> 3D but remains still gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_BGR = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        # Matrix concatenation
        img = np.hstack((img, gray_BGR))

    cv.imshow('Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_video(vid_path: str = None,
               scale: float = 0.0,
               concat_gray: bool = False) -> None:
    '''
       Reading and showing video function
       Args:
           vid_path: str - path to your video data
           scale: float - in order to rescale your image
           concat_gray: bool - if requires to show RGB and GRAY
                               in one window
    '''
    if not vid_path:
        vid_path = 0 # In order to use webcam

    cap = cv.VideoCapture(vid_path)

    if not cap.isOpened():
        print("No web-camera or video source provided")
        return

    while True:
        ret, frame = cap.read()
        if scale:
            frame = rescale_frame(frame, scale)

        if concat_gray:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray_BGR = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)


            frame = np.hstack((frame, gray_BGR))
        cv.imshow("Video", frame)

        if cv.waitKey(20) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()

            return
