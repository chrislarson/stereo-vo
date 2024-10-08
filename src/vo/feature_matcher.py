from enum import Enum
from typing import Sequence

import cv2

from vo.feature_detector import DetectorType


class MatcherType(Enum):
    BruteForce = 1
    Flann = 2


def filter_matches(matches: Sequence[Sequence[cv2.DMatch]], dist_ratio: float):
    filtered = []
    for match in matches:
        match_1 = match[0]
        match_2 = match[1]
        if match_1.distance < dist_ratio * match_2.distance:
            filtered.append(match_1)
    return filtered


class FeatureMatcher:
    _matcher: cv2.DescriptorMatcher
    # https://datahacker.rs/feature-matching-methods-comparison-in-opencv/

    def __init__(self, kind: MatcherType, detector_kind: DetectorType):
        match kind:
            case kind.BruteForce:
                if detector_kind.is_binary():
                    self._matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)
                else:
                    self._matcher = cv2.BFMatcher.create(cv2.NORM_L2, crossCheck=False)
            case kind.Flann:
                if detector_kind.is_binary():
                    FLANN_INDEX_LSH = 6
                    index_params = dict(
                        algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1
                    )
                    search_params = dict(checks=50)
                    self._matcher = cv2.FlannBasedMatcher(index_params, search_params).create()
                else:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    self._matcher = cv2.FlannBasedMatcher(index_params, search_params).create()

    def match_one(self, query_descriptors, train_descriptors, mask):
        return self._matcher.match(query_descriptors, train_descriptors, mask)

    def match_knn(self, query_descriptors: cv2.Mat, train_descriptors: cv2.Mat, k: int):
        return self._matcher.knnMatch(query_descriptors, train_descriptors, k)
