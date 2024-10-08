import os
from typing import List, Any

import cv2
import numpy as np
from cv2 import Mat

from feature_detector import FeatureDetector, DetectorType
from feature_matcher import FeatureMatcher, MatcherType, filter_matches
from utils import visualize_paths
from vo.stereo_matcher import StereoMatcherType, StereoMatcher, calc_depth_map


def form_transform(r, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def load_calibration(kitti_path: str, seq: str, side: str):
    filepath = f"{kitti_path}/calibs/{seq}/calib.txt"
    with open(filepath, "r") as f:
        if side == "right":
            _ = f.readline()  # bypass left
        line = f.readline()
        line_edit = line.split(" ")[1:]
        params = np.array(line_edit, dtype=np.float64)
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P


def load_gt_poses(kitti_path: str, seq: str):
    poses = []
    filepath = f"{kitti_path}/poses/{seq}.txt"
    with open(filepath, "r") as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=" ")
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses


def load_image_sequence(kitti_path: str, seq: str, side: str, len=1000) -> List[Mat]:
    images = []
    if side == "left":
        filepath = f"{kitti_path}/sequences/{seq}/image_0"
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        for i in range(0, len):
            images.append(cv2.imread(image_paths[i]))
    else:
        filepath = f"{kitti_path}/sequences/{seq}/image_1"
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        for i in range(0, len):
            images.append(cv2.imread(image_paths[i]))
    return images


def decompose_projection_matrix(p: np.ndarray[Any, np.dtype[np.generic]]):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t


def main():
    # KITTI Sequence
    seq = "00"

    # Pose
    gt_poses = load_gt_poses("/Users/chris/kitti_dataset", seq)
    est_poses = []
    H_tot = np.eye(4)

    # Camera Properties
    _, p_left = load_calibration("/Users/chris/kitti_dataset", seq, "left")
    _, p_right = load_calibration("/Users/chris/kitti_dataset", seq, "right")
    k_left, r_left, t_left = decompose_projection_matrix(p_left)
    k_right, r_right, t_right = decompose_projection_matrix(p_right)

    detector_name = "SIFT"
    ratio = 0.4
    # Algorithm(s) Configuration
    max_depth = 5000
    detector = FeatureDetector(DetectorType.SIFT)
    matcher = FeatureMatcher(MatcherType.BruteForce, DetectorType.SIFT)
    stereo_matcher = StereoMatcher(StereoMatcherType.SGBM)

    # Images
    images_left = load_image_sequence("/Users/chris/kitti_dataset", seq, "left", 250)
    images_right = load_image_sequence("/Users/chris/kitti_dataset", seq, "right", 250)

    images_left = [cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) for img_l in images_left]
    images_right = [cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) for img_r in images_right]

    # Algorithm
    for i in range(len(images_left) - 1):
        disp_map = stereo_matcher.compute_disparity(images_left[i], images_right[i])
        depth_map = calc_depth_map(disp_map, k_left, t_left, t_right, rectified=True)

        # TODO: Apply mask over stereo blind spot
        kp_0, des_0 = detector.detect_and_compute(images_left[i], None)
        kp_1, des_1 = detector.detect_and_compute(images_left[i + 1], None)
        matches = matcher.match_knn(des_0, des_1, k=2)
        filtered_matches = filter_matches(matches, dist_ratio=0.4)

        unfiltered_matched_img = cv2.drawMatchesKnn(
            images_left[i],
            kp_0,
            images_left[i + 1],
            kp_1,
            matches,
            outImg=None,
            flags=2,
        )

        filtered_matched_img = cv2.drawMatchesKnn(
            images_left[i],
            kp_0,
            images_left[i + 1],
            kp_1,
            [filtered_matches],
            outImg=None,
            flags=2,
        )

        left_left_img = np.hstack((images_left[i], images_left[i + 1]))

        norm_image = cv2.normalize(disp_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imshow("Left Camera Frames, times t=i and t=i+1", left_left_img)
        cv2.imshow("Disparity Map", norm_image)
        cv2.imshow("Unfiltered Matches", unfiltered_matched_img)
        cv2.imshow("Filtered Matches", filtered_matched_img)
        cv2.waitKey(50)

        img_kp_0 = np.array([kp_0[m.queryIdx].pt for m in filtered_matches], dtype=np.float32)
        img_kp_1 = np.array([kp_1[m.trainIdx].pt for m in filtered_matches], dtype=np.float32)

        cx = k_left[0, 2]
        cy = k_left[1, 2]
        fx = k_left[0, 0]
        fy = k_left[1, 1]
        object_points = np.zeros((0, 3))
        delete = []
        for k, (u, v) in enumerate(img_kp_0):
            z = depth_map[int(v), int(u)]
            if z > max_depth:
                delete.append(k)
                continue
            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            object_points = np.vstack([object_points, np.array([x, y, z])])

        img_kp_1 = np.delete(img_kp_1, delete, 0)

        # Use PnP algorithm with RANSAC for robustness to outliers
        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, img_kp_1, k_left, np.array([]))
        rmat = cv2.Rodrigues(rvec)[0]

        H = form_transform(rmat, tvec.T)
        H_tot = H_tot @ np.linalg.inv(H)
        est_poses.append(H_tot)

    cv2.destroyAllWindows()
    gt_path = []
    est_path = []

    for i, est_pose in enumerate(est_poses):
        est_path.append((est_pose[0, 3], est_pose[2, 3]))
        gt_pose = gt_poses[i]
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))

    title = f"Stereo VO  ---   KITTI Sequence: {seq}  ---   Feature Detector: {detector_name}  ---   Lowe's Distance Ratio: {ratio}"
    visualize_paths(
        gt_path,
        est_path,
        title=title,
        file_out=os.path.basename(__file__) + f"{seq}{detector_name}.html",
    )


if __name__ == "__main__":
    main()
