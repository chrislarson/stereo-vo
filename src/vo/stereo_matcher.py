from enum import Enum

import cv2
import numpy as np


class StereoMatcherType(Enum):
    SGBM = 1
    BM = 2


def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    # Get focal length of x-axis for left camera
    f = k_left[0][0]

    # Calculate baseline of stereo pair
    if rectified:
        b = t_right[0] - t_left[0]
    else:
        b = t_left[0] - t_right[0]

    # Avoid instability and division by zero
    disp_left[disp_left == 0.0] = 0.01
    disp_left[disp_left == -1.0] = 0.01

    depth_map = f * b / disp_left
    return depth_map


class StereoMatcher:
    def __init__(self, kind: StereoMatcherType):
        sad_window = 6
        num_disparities = sad_window * 16
        block_size = 11
        match kind:
            case StereoMatcherType.SGBM:
                self._matcher = cv2.StereoSGBM.create(
                    numDisparities=num_disparities,
                    minDisparity=0,
                    blockSize=block_size,
                    P1=8 * 3 * sad_window**2,
                    P2=32 * 3 * sad_window**2,
                    mode=cv2.STEREO_SGBM_MODE_HH,
                )
            case StereoMatcherType.BM:
                self._matcher = cv2.StereoBM.create(
                    numDisparities=16,
                    blockSize=15,
                )

    def compute_disparity(self, img_left: cv2.Mat, img_right: cv2.Mat):
        disp = self._matcher.compute(img_left, img_right).astype(np.float32) / 16
        return disp
