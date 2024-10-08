from enum import Enum
from typing import Any

import cv2
import numpy as np

from video_capture_threaded import VideoCaptureThreaded


class ImageSource(Enum):
    Local = 1
    KITTI = 2


class ImageLoader:
    frame_count = 0
    capture: VideoCaptureThreaded
    right_prev: cv2.Mat | np.ndarray[Any, np.dtype]
    right_curr: cv2.Mat | np.ndarray[Any, np.dtype]
    left_prev: cv2.Mat | np.ndarray[Any, np.dtype]
    left_curr: cv2.Mat | np.ndarray[Any, np.dtype]

    # https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    def __init__(self, image_source: ImageSource, L_src: int, R_src: int):
        self.capture: VideoCaptureThreaded
        self.right_curr = np.zeros((720, 1280), dtype=np.uint8)
        self.left_curr = np.zeros((720, 1280), dtype=np.uint8)
        self.right_prev = np.zeros((720, 1280), dtype=np.uint8)
        self.left_prev = np.zeros((720, 1280), dtype=np.uint8)

        match image_source:
            case ImageSource.Local:
                self.capture = VideoCaptureThreaded(L_src, R_src, (1280, 960), (1280, 720))
                self.capture.start()
                print("Initialized Image Loader")

            case ImageSource.KITTI:
                pass

    def get_frame(self):
        left_frame, right_frame = self.capture.read()
        left_frame = left_frame[110:830, :]
        self.left_prev = self.left_curr.copy()
        self.right_prev = self.right_curr.copy()
        self.left_curr = left_frame.copy()
        self.right_curr = right_frame.copy()
        self.frame_count = self.frame_count + 1

    def close(self):
        self.capture.stop()
