"""
Pose Detector for PT Rehab Coach
Uses MediaPipe Pose to extract body landmarks, compute joint angles,
and count reps via angle thresholds -- all on CPU at ~30fps.
"""

import math
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    logger.warning("mediapipe not installed -- pose-based rep counting disabled")


# MediaPipe landmark indices (subset we care about)
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
}


def _angle_between(a, b, c) -> float:
    """Compute angle at point b given three (x, y) landmarks. Returns degrees."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.hypot(*ba)
    mag_bc = math.hypot(*bc)
    if mag_ba * mag_bc == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def _midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class AngleRepCounter:
    """Counts reps by watching a joint angle cross up/down thresholds."""

    def __init__(self, down_threshold: float, up_threshold: float):
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.reps = 0
        self._state = "up"  # start assuming extended/standing

    def update(self, angle: float) -> bool:
        """Feed a new angle value. Returns True if a rep just completed."""
        if self._state == "up" and angle < self.down_threshold:
            self._state = "down"
        elif self._state == "down" and angle > self.up_threshold:
            self._state = "up"
            self.reps += 1
            return True
        return False

    def reset(self):
        self.reps = 0
        self._state = "up"


class PoseDetector:
    """Wraps MediaPipe Pose for lightweight skeleton detection."""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self._pose = None
            return
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # fastest
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._rep_counter: Optional[AngleRepCounter] = None
        self._joint_keys: Optional[tuple] = None

    @property
    def available(self) -> bool:
        return self._pose is not None

    def configure_exercise(self, joint_keys: tuple[str, str, str],
                           down_threshold: float, up_threshold: float):
        """Set which joint angle to track and the rep thresholds."""
        self._joint_keys = joint_keys
        self._rep_counter = AngleRepCounter(down_threshold, up_threshold)

    def clear_exercise(self):
        self._joint_keys = None
        self._rep_counter = None

    @property
    def reps(self) -> int:
        return self._rep_counter.reps if self._rep_counter else 0

    def reset_reps(self):
        if self._rep_counter:
            self._rep_counter.reset()

    def process_frame(self, bgr_frame: np.ndarray) -> dict:
        """
        Run pose detection on a BGR frame.
        Returns dict with angle, rep_completed, total_reps, landmarks visibility.
        Runs in ~5-15ms on CPU.
        """
        if not self.available:
            return {"pose_detected": False}

        rgb = bgr_frame[:, :, ::-1]
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            return {"pose_detected": False}

        lms = results.pose_landmarks.landmark
        h, w = bgr_frame.shape[:2]

        def pt(name):
            idx = LM.get(name)
            if idx is None:
                return None
            lm = lms[idx]
            return (lm.x * w, lm.y * h) if lm.visibility > 0.3 else None

        result = {"pose_detected": True}

        if self._joint_keys and self._rep_counter:
            a_name, b_name, c_name = self._joint_keys
            a, b, c = pt(a_name), pt(b_name), pt(c_name)

            if a and b and c:
                angle = _angle_between(a, b, c)
                rep_completed = self._rep_counter.update(angle)
                result.update({
                    "angle": round(angle, 1),
                    "rep_completed": rep_completed,
                    "total_reps": self._rep_counter.reps,
                })
            else:
                result.update({"angle": None, "rep_completed": False,
                               "total_reps": self._rep_counter.reps})
        return result

    def close(self):
        if self._pose:
            self._pose.close()
            self._pose = None
