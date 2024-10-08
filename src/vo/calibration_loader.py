import os
import pickle
from enum import Enum
from typing import Any, Sequence

import numpy as np


class CalibrationType(Enum):
    KITTI = 1
    Local = 2


class CalibrationLoader:
    k_left: np.ndarray[Any, np.dtype]
    k_right: np.ndarray[Any, np.dtype]

    k_left_udst: np.ndarray[Any, np.dtype]
    k_right_udst: np.ndarray[Any, np.dtype]

    t_left: np.ndarray[Any, np.dtype]
    t_right: np.ndarray[Any, np.dtype]

    dist_left: np.ndarray[Any, np.dtype]
    dist_right: np.ndarray[Any, np.dtype]

    roi_left: Sequence[int]
    roi_right: Sequence[int]

    def __init__(self, calib_type: CalibrationType):
        dir_path = os.path.dirname(__file__)
        dir_path = dir_path.split(os.sep)[:-2]
        dir_path = os.path.sep.join(dir_path)
        match calib_type:
            case CalibrationType.Local:
                # Instrinsic Matrix
                fp_right = os.path.join(dir_path, "calibration", "intrinsic_right.pkl")
                fp_left = os.path.join(dir_path, "calibration", "intrinsic_left.pkl")
                with open(fp_left, "rb") as fp_left:
                    kl = pickle.load(fp_left)
                    kl = np.array(kl, dtype=np.float64)
                    self.k_left = kl
                with open(fp_right, "rb") as fp_right:
                    kr = pickle.load(fp_right)
                    kr = np.array(kr, dtype=np.float64)
                    self.k_right = kr
                # Distortion Matrix
                fp_distright = os.path.join(dir_path, "calibration", "distortion_right.pkl")
                fp_distleft = os.path.join(dir_path, "calibration", "distortion_left.pkl")
                with open(fp_distleft, "rb") as fp_distleft:
                    dl = pickle.load(fp_distleft)
                    dl = np.array(dl, dtype=np.float64)
                    self.dist_left = dl
                with open(fp_distright, "rb") as fp_distright:
                    dr = pickle.load(fp_distright)
                    dr = np.array(dr, dtype=np.float64)
                    self.dist_right = dr

                # Translation Vector
                # self.t_right = np.array([[0], [0], [0]], dtype=np.float64)
                self.t_right = np.array([[0.1695], [0], [0]], dtype=np.float64)

                # self.t_right = np.array([[6.95], [0], [0]], dtype=np.float64)
                self.t_left = np.array([[0], [0], [0]], dtype=np.float64)

                # kl_opt, roi_l = cv2.getOptimalNewCameraMatrix(
                #     self.k_left, self.dist_left, (1280, 720), 1, (1280, 720)
                # )
                #
                # kr_opt, roi_r = cv2.getOptimalNewCameraMatrix(
                #     self.k_right, self.dist_right, (1280, 720), 1, (1280, 720)
                # )

                # self.k_left_udst = kl_opt
                # self.k_right_udst = kr_opt
                # self.roi_left = roi_l
                # self.roi_right = roi_r

                # Rotation Matrix
