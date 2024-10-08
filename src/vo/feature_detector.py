from enum import Enum

import cv2 as cv


class DetectorType(Enum):
    SIFT = 1
    ORB = 2
    AKAZE = 3

    def is_binary(self):
        return self in (DetectorType.ORB, DetectorType.AKAZE)


class FeatureDetector:
    _detector: cv.AKAZE | cv.SIFT | cv.ORB

    def __init__(self, kind: DetectorType):
        match kind:
            case kind.SIFT:
                self._detector = cv.SIFT.create()
            case kind.ORB:
                self._detector = cv.ORB.create(nfeatures=2000)
            case kind.AKAZE:
                self._detector = cv.AKAZE.create()

    def detect_and_compute(self, image: cv.Mat, mask: cv.Mat | None):
        return self._detector.detectAndCompute(image, mask)
