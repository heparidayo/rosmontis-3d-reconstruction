"""대응점 중복/방향, 모델 track 파싱, 카메라 gate의 분석 오류를 방지한다."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("diagnose", Path(__file__).resolve().parents[1]/"src/diagnose_sfm.py")
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


class DiagnosisTests(unittest.TestCase):
    def test_reversed_pair_deduplication_and_ambiguity(self):
        # query ID=3; stored pairs put query keypoint in the second column.
        reg = {1: {"pids": np.array([10, 11, 12])},
               2: {"pids": np.array([10, 12, 13])}}
        pairs = {(1, 3): {"verified": 15, "data": np.array([[0, 0], [1, 1], [2, 2]])},
                 (2, 3): {"verified": 15, "data": np.array([[0, 0], [1, 1], [2, 3]])}}
        summary, clean = d.candidates(3, reg, {i: {} for i in range(10, 14)}, pairs)
        self.assertEqual(summary["candidate_keypoints"], 4)
        self.assertEqual(summary["candidate_links"], 5)
        self.assertEqual(summary["ambiguous_keypoints"], 1)
        self.assertEqual(clean, [(0, 10), (2, 12), (3, 13)])

    def test_same_3d_point_cannot_inflate_clean_count(self):
        reg = {2: {"pids": np.array([10, 10])}}
        pair = {(1, 2): {"verified": 15, "data": np.array([[0, 0], [1, 1]])}}
        summary, clean = d.candidates(1, reg, {10: {}}, pair)
        self.assertEqual(summary["candidate_keypoints"], 2)
        self.assertEqual(summary["clean_unique_correspondences"], 1)

    def test_camera_gate_uses_absolute_distortion(self):
        limits = dict(min_focal_length_ratio=.1, max_focal_length_ratio=10., max_extra_param=1.)
        camera = dict(model="SIMPLE_RADIAL", width=7008, height=4672,
                      params=[13625.6, 3504, 2336, -4.463])
        self.assertEqual(d.camera_gate(camera, limits)["reasons"], ["extra_param"])
        camera["params"][-1] = .505
        self.assertTrue(d.camera_gate(camera, limits)["valid"])

    def test_graph_connectivity_does_not_depend_on_registration(self):
        pairs = {(1, 2): {"verified": 50}, (2, 3): {"verified": 15}}
        self.assertEqual(d.components([1, 2, 3], pairs, 15), [[1, 2, 3]])
        self.assertEqual(d.components([1, 2, 3], pairs, 30), [[1, 2], [3]])

    def test_binary_model_with_untriangulated_feature(self):
        root = Path(__file__).resolve().parents[1]/"results"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="diagnose_test_") as tmp:
            p = Path(tmp)
            self.assertTrue(p.resolve().is_relative_to(root.resolve()))
            (p/"cameras.bin").write_bytes(struct.pack("<QiiQQdddd", 1, 1, 2, 100, 80, 90, 50, 40, 0))
            head = struct.pack("<Qidddddddi", 1, 7, 1, 0, 0, 0, 1, 2, 3, 1)
            obs = struct.pack("<Qddqddq", 2, 20, 30, 42, 40, 50, -1)
            (p/"images.bin").write_bytes(head+b"photo.JPG\0"+obs)
            (p/"points3D.bin").write_bytes(struct.pack("<QQdddBBBdQii", 1, 42, 1, 2, 3, 10, 20, 30, .5, 1, 7, 0))
            cameras, images, points = d.read_model(p)
            self.assertEqual(cameras[1]["model"], "SIMPLE_RADIAL")
            self.assertEqual(images[7]["pids"].tolist(), [42, -1])
            self.assertEqual(points[42]["track"].tolist(), [[7, 0]])
            np.testing.assert_allclose(-d.rotation(images[7]["q"]).T@images[7]["t"], [-1, -2, -3])

    def test_pnp_with_known_pose_and_outliers(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV 없음")
        rng = np.random.default_rng(5)
        xyz = rng.uniform(-1, 1, (80, 3))
        xyz[:, 2] += 6
        cam = dict(model="SIMPLE_RADIAL", params=[800, 320, 240, .02])
        k, dist = d.camera_cv(cam)
        xy = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, dist)[0][:, 0]
        xy[-15:] += 150
        clean = [(i, i) for i in range(80)]
        result = d.pnp_check(clean, xy, {i: {"xyz": x} for i, x in enumerate(xyz)}, cam, cv2)
        self.assertGreaterEqual(result["pnp_inliers"], 60)
        self.assertLessEqual(result["pnp_inliers"], 65)
        self.assertEqual(result["pnp_positive_depth_ratio"], 1.)

    def test_mapper_log_counts_repeated_attempts_and_reasons(self):
        root = Path(__file__).resolve().parents[1]/"results"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix="diagnose_log_test_") as tmp:
            p = Path(tmp)
            self.assertTrue(p.resolve().is_relative_to(root.resolve()))
            log = p/"mapper.log"
            log.write_text(
                "Registering image #6 (num_reg_frames=17)\n"
                "=> Image sees 25 / 80 points\n"
                "incremental_mapper.cc:343] Insufficient number of 2D-3D correspondences for registration (25 < 30)\n"
                "=> Could not register, trying another image.\n"
                "Registering image #6 (num_reg_frames=17)\n"
                "=> Image sees 25 / 80 points\n"
                "Registering image with structure-less fallback\n"
                "=> Could not register, trying another image.\n", encoding="utf-8")
            result = d.read_log(log)
            self.assertEqual(result["failed_attempts"], 2)
            self.assertEqual(result["failure_with_at_least_30_visible_points"], 0)
            self.assertEqual(sum(result["failure_reason_counts"].values()), 1)
            self.assertTrue(result["attempts"][1]["fallback"])


if __name__ == "__main__":
    unittest.main()
