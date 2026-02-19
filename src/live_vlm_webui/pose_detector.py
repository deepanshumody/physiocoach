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


# MediaPipe landmark indices
LM = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]

# Joint angle mapping: (joint, movement) -> (landmark_a, landmark_b, landmark_c)
# Angle is measured at landmark_b
ROM_JOINT_MAP = {
    ("knee", "flexion"): ("hip", "knee", "ankle"),
    ("hip", "flexion"): ("shoulder", "hip", "knee"),
    ("elbow", "flexion"): ("shoulder", "elbow", "wrist"),
    ("shoulder", "abduction"): ("hip", "shoulder", "wrist"),
    ("ankle", "plantarflexion"): ("knee", "ankle", "foot_index"),
    ("neck", "rotation"): ("left_shoulder", "nose", "right_shoulder"),
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


def compute_rom_angle(landmarks: dict, joint: str, movement: str, side: str) -> Optional[float]:
    """Compute ROM angle from MediaPipe landmarks."""
    triplet = ROM_JOINT_MAP.get((joint, movement))
    if not triplet:
        return None
    
    # Special case for neck rotation
    if joint == "neck" and movement == "rotation":
        ls = landmarks.get("left_shoulder")
        nose = landmarks.get("nose")
        rs = landmarks.get("right_shoulder")
        if ls and nose and rs:
            return _angle_between(ls, nose, rs)
        return None
    
    a_base, b_base, c_base = triplet
    
    # For "both" sides, measure both and average
    sides_to_check = ["left", "right"] if side == "both" else [side]
    angles = []
    
    for s in sides_to_check:
        a = landmarks.get(f"{s}_{a_base}")
        b = landmarks.get(f"{s}_{b_base}")
        c = landmarks.get(f"{s}_{c_base}")
        if a and b and c:
            angles.append(_angle_between(a, b, c))
    
    # Fallback for shoulder abduction when hip not visible
    if not angles and joint == "shoulder" and movement == "abduction":
        for s in sides_to_check:
            other_side = "right" if s == "left" else "left"
            shoulder = landmarks.get(f"{s}_shoulder")
            other_shoulder = landmarks.get(f"{other_side}_shoulder")
            wrist = landmarks.get(f"{s}_wrist")
            if shoulder and other_shoulder and wrist:
                # Measure angle from shoulder line to arm
                raw_angle = _angle_between(other_shoulder, shoulder, wrist)
                # Map to 0-90 range (subtract 90 since horizontal = 90°)
                mapped_angle = max(0, min(90, raw_angle - 90))
                angles.append(mapped_angle)
    
    return sum(angles) / len(angles) if angles else None


def get_tracked_joint_for_display(landmarks: dict, joint: str, movement: str, side: str):
    """Get the actual joint points being tracked for visualization."""
    triplet = ROM_JOINT_MAP.get((joint, movement))
    if not triplet:
        return None
    
    # Special case for neck
    if joint == "neck" and movement == "rotation":
        ls = landmarks.get("left_shoulder")
        nose = landmarks.get("nose")
        rs = landmarks.get("right_shoulder")
        if ls and nose and rs:
            return (ls, nose, rs), ("left_shoulder", "nose", "right_shoulder")
        return None
    
    a_base, b_base, c_base = triplet
    sides_to_check = ["left", "right"] if side == "both" else [side]
    
    # Try standard tracking first
    for s in sides_to_check:
        a = landmarks.get(f"{s}_{a_base}")
        b = landmarks.get(f"{s}_{b_base}")
        c = landmarks.get(f"{s}_{c_base}")
        if a and b and c:
            return (a, b, c), (f"{s}_{a_base}", f"{s}_{b_base}", f"{s}_{c_base}")
    
    # Fallback for shoulder abduction
    if joint == "shoulder" and movement == "abduction":
        for s in sides_to_check:
            other_side = "right" if s == "left" else "left"
            shoulder = landmarks.get(f"{s}_shoulder")
            other_shoulder = landmarks.get(f"{other_side}_shoulder")
            wrist = landmarks.get(f"{s}_wrist")
            if shoulder and other_shoulder and wrist:
                return (other_shoulder, shoulder, wrist), (f"{other_side}_shoulder", f"{s}_shoulder", f"{s}_wrist")
    
    return None


import cv2

def draw_skeleton(frame: np.ndarray, landmarks: dict, tracked_joint=None,
                  angle: float = None, joint_keys: tuple = None, 
                  rom_angles: list = None) -> np.ndarray:
    """Draw skeleton with highlighted tracked limb and ROM angles."""
    overlay = frame.copy()
    
    def _pt(name):
        p = landmarks.get(name)
        return (int(p[0]), int(p[1])) if p else None
    
    # Build set of tracked landmark names
    tracked_names = set(joint_keys) if joint_keys else set()
    
    # Draw all skeleton connections
    for a_name, b_name in SKELETON_CONNECTIONS:
        pa, pb = _pt(a_name), _pt(b_name)
        if pa and pb:
            is_tracked = (a_name in tracked_names and b_name in tracked_names)
            color = (0, 255, 0) if is_tracked else (150, 150, 150)  # Green for tracked, gray for others
            thickness = 5 if is_tracked else 2
            cv2.line(overlay, pa, pb, color, thickness, cv2.LINE_AA)
    
    # Draw joint dots
    for name in landmarks:
        p = _pt(name)
        if p:
            is_tracked = name in tracked_names
            color = (0, 255, 0) if is_tracked else (200, 200, 200)
            radius = 7 if is_tracked else 4
            cv2.circle(overlay, p, radius, color, -1, cv2.LINE_AA)
    
    # Draw angle at tracked joint
    if tracked_joint and angle is not None:
        a, b, c = tracked_joint
        bi = (int(b[0]), int(b[1]))
        ai = (int(a[0]), int(a[1]))
        ci = (int(c[0]), int(c[1]))
        
        # Draw angle lines (bright green)
        cv2.line(overlay, bi, ai, (0, 255, 0), 5, cv2.LINE_AA)
        cv2.line(overlay, bi, ci, (0, 255, 0), 5, cv2.LINE_AA)
        cv2.circle(overlay, bi, 9, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Skip arc for neck (obtuse angle looks confusing)
        is_neck = joint_keys == ("left_shoulder", "nose", "right_shoulder")
        if not is_neck:
            arc_r = 35
            ang_a = math.degrees(math.atan2(ai[1] - bi[1], ai[0] - bi[0]))
            ang_c = math.degrees(math.atan2(ci[1] - bi[1], ci[0] - bi[0]))
            start = min(ang_a, ang_c)
            end = max(ang_a, ang_c)
            if end - start > 180:
                start, end = end, start + 360
            cv2.ellipse(overlay, bi, (arc_r, arc_r), 0, start, end, (0, 255, 0), 3, cv2.LINE_AA)
        
        # Angle label
        label = f"{int(angle)}\u00b0"
        tx, ty = bi[0] + 14, bi[1] - 14
        cv2.putText(overlay, label, (tx + 1, ty + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 3, cv2.LINE_AA)
    
    # ROM angles overlay (top-right corner)
    if rom_angles:
        parts = []
        for r in rom_angles:
            label_text = r.get("label", "").replace("_", " ").title()
            ang = r.get("angle")
            if ang is not None and label_text:
                parts.append(f"{label_text}: {int(ang)}\u00b0")
        
        if parts:
            txt = "  |  ".join(parts)
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            w = overlay.shape[1]
            rx = max(10, w - tw - 20)
            ry = 54
            cv2.rectangle(overlay, (rx - 8, ry - th - 6), (min(w - 8, rx + tw + 8), ry + 8), (0, 0, 0), -1)
            cv2.rectangle(overlay, (rx - 8, ry - th - 6), (min(w - 8, rx + tw + 8), ry + 8), (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, txt, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


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
        Returns dict with angle, rep_completed, total_reps, landmarks, tracked_joint.
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

        landmarks = {name: pt(name) for name in LM}
        result = {"pose_detected": True, "landmarks": landmarks}

        if self._joint_keys and self._rep_counter:
            a_name, b_name, c_name = self._joint_keys
            a, b, c = pt(a_name), pt(b_name), pt(c_name)
            joint_keys_used = self._joint_keys
            
            # Auto-detect which arm for shoulder/elbow exercises
            if not (a and b and c) or (b_name in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"] and "wrist" in c_name):
                # Detect which arm is active
                left_wrist = pt("left_wrist")
                right_wrist = pt("right_wrist")
                left_shoulder = pt("left_shoulder")
                right_shoulder = pt("right_shoulder")
                left_elbow = pt("left_elbow")
                right_elbow = pt("right_elbow")
                
                use_side = "left"  # default
                
                if left_wrist and right_wrist and left_shoulder and right_shoulder:
                    # For shoulder exercises: check wrist height
                    if "shoulder" in b_name:
                        left_raised = left_wrist[1] < left_shoulder[1]
                        right_raised = right_wrist[1] < right_shoulder[1]
                        
                        if left_raised and not right_raised:
                            use_side = "left"
                        elif right_raised and not left_raised:
                            use_side = "right"
                        elif left_raised and right_raised:
                            use_side = "left" if left_wrist[1] < right_wrist[1] else "right"
                    
                    # For elbow exercises: check which elbow is more bent
                    elif "elbow" in b_name and left_elbow and right_elbow:
                        # Calculate elbow angles
                        left_angle = _angle_between(pt("left_shoulder"), left_elbow, left_wrist) if pt("left_shoulder") and left_elbow and left_wrist else 180
                        right_angle = _angle_between(pt("right_shoulder"), right_elbow, right_wrist) if pt("right_shoulder") and right_elbow and right_wrist else 180
                        # Smaller angle = more bent = active
                        use_side = "left" if left_angle < right_angle else "right"
                
                # Try with detected side
                if "hip" in a_name:
                    a = pt(f"{use_side}_hip")
                elif "shoulder" in a_name and "elbow" not in b_name:
                    # Shoulder exercise - use other shoulder or hip
                    a = pt(f"{use_side}_hip") or pt(f"{'right' if use_side == 'left' else 'left'}_shoulder")
                else:
                    a = pt(f"{use_side}_{a_name.split('_')[-1]}")
                
                b = pt(f"{use_side}_{b_name.split('_')[-1]}")
                c = pt(f"{use_side}_{c_name.split('_')[-1]}")
                
                if a and b and c:
                    a_key = f"{use_side}_hip" if "hip" in a_name else (f"{'right' if use_side == 'left' else 'left'}_shoulder" if "shoulder" in a_name and "elbow" not in b_name else f"{use_side}_{a_name.split('_')[-1]}")
                    joint_keys_used = (a_key, f"{use_side}_{b_name.split('_')[-1]}", f"{use_side}_{c_name.split('_')[-1]}")
                else:
                    # Fallback: use shoulder-line angle for shoulder exercises
                    other_side = "right" if use_side == "left" else "left"
                    a = pt(f"{other_side}_shoulder")
                    b = pt(f"{use_side}_shoulder")
                    c = pt(f"{use_side}_wrist")
                    if a and b and c:
                        joint_keys_used = (f"{other_side}_shoulder", f"{use_side}_shoulder", f"{use_side}_wrist")

            if a and b and c:
                angle = _angle_between(a, b, c)
                # For shoulder abduction fallback (shoulder-line), map to 0-90 range
                if joint_keys_used and "shoulder" in joint_keys_used[0] and "shoulder" in joint_keys_used[1]:
                    angle = max(0, min(90, angle - 90))
                
                rep_completed = self._rep_counter.update(angle)
                result.update({
                    "angle": round(angle, 1),
                    "rep_completed": rep_completed,
                    "total_reps": self._rep_counter.reps,
                    "tracked_joint": (a, b, c),
                    "joint_keys": joint_keys_used,
                })
            else:
                result.update({
                    "angle": None,
                    "rep_completed": False,
                    "total_reps": self._rep_counter.reps,
                    "tracked_joint": None,
                    "joint_keys": self._joint_keys,
                })
        return result

    def close(self):
        if self._pose:
            self._pose.close()
            self._pose = None
