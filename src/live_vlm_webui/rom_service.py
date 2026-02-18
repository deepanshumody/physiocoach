# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Range of Motion (ROM) Measurement Service
Uses VLM to estimate joint angles from video frames — no OpenCV pose detection.
Provides clinical-grade ROM tracking with history and progress measurement.
"""

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Clinical ROM reference ranges (in degrees) based on AMA Guides / AAOS standards
# Format: { joint_name: { movement: (min_normal, max_normal, unit) } }
ROM_REFERENCE = {
    "shoulder": {
        "flexion": (0, 180, "°"),
        "extension": (0, 60, "°"),
        "abduction": (0, 180, "°"),
        "adduction": (0, 45, "°"),
        "internal_rotation": (0, 90, "°"),
        "external_rotation": (0, 90, "°"),
    },
    "elbow": {
        "flexion": (0, 150, "°"),
        "extension": (0, 0, "°"),
    },
    "wrist": {
        "flexion": (0, 80, "°"),
        "extension": (0, 70, "°"),
        "radial_deviation": (0, 20, "°"),
        "ulnar_deviation": (0, 30, "°"),
    },
    "hip": {
        "flexion": (0, 120, "°"),
        "extension": (0, 30, "°"),
        "abduction": (0, 45, "°"),
        "adduction": (0, 30, "°"),
        "internal_rotation": (0, 40, "°"),
        "external_rotation": (0, 45, "°"),
    },
    "knee": {
        "flexion": (0, 135, "°"),
        "extension": (0, 0, "°"),
    },
    "ankle": {
        "dorsiflexion": (0, 20, "°"),
        "plantarflexion": (0, 50, "°"),
        "inversion": (0, 35, "°"),
        "eversion": (0, 15, "°"),
    },
    "neck": {
        "flexion": (0, 50, "°"),
        "extension": (0, 60, "°"),
        "lateral_flexion": (0, 45, "°"),
        "rotation": (0, 80, "°"),
    },
    "lumbar_spine": {
        "flexion": (0, 60, "°"),
        "extension": (0, 25, "°"),
        "lateral_flexion": (0, 25, "°"),
        "rotation": (0, 30, "°"),
    },
}


@dataclass
class ROMMeasurement:
    """Single ROM measurement data point"""
    joint: str
    movement: str
    angle: float
    target_angle: float
    timestamp: float
    confidence: str  # "high", "medium", "low"
    side: str  # "left", "right", "bilateral"
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "joint": self.joint,
            "movement": self.movement,
            "angle": self.angle,
            "target_angle": self.target_angle,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "side": self.side,
            "notes": self.notes,
            "percent_of_normal": round((self.angle / self.target_angle) * 100, 1)
            if self.target_angle > 0
            else 0,
        }


class ROMService:
    """
    ROM measurement service that uses VLM for joint angle estimation.
    Tracks measurement history and provides clinical progress feedback.
    """

    def __init__(self):
        self.enabled = False
        self.current_joint = "knee"
        self.current_side = "right"
        self.measurements: list[ROMMeasurement] = []
        self.session_start = time.time()

    def get_rom_prompt(self, joint: str, side: str = "right") -> str:
        """Generate a specialized VLM prompt for ROM measurement of a specific joint."""
        ref = ROM_REFERENCE.get(joint, {})
        movements = list(ref.keys())
        movements_str = ", ".join(movements)
        normal_ranges = "; ".join(
            f"{m}: 0-{ref[m][1]}°" for m in movements
        )

        return f"""You are a clinical Range of Motion (ROM) assessment assistant. Analyze the person's {side} {joint} joint in this image.

TASK: Estimate the visible joint angles in degrees for the {joint} joint.

AVAILABLE MOVEMENTS for {joint}: {movements_str}
NORMAL RANGES: {normal_ranges}

RESPOND ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "joint": "{joint}",
  "side": "{side}",
  "visible": true,
  "measurements": [
    {{
      "movement": "<movement_name>",
      "angle_degrees": <number>,
      "confidence": "<high|medium|low>"
    }}
  ],
  "body_position": "<brief description of person's position>",
  "notes": "<any clinical observations>"
}}

If the {joint} joint is NOT clearly visible, respond with:
{{
  "joint": "{joint}",
  "side": "{side}",
  "visible": false,
  "measurements": [],
  "body_position": "not visible",
  "notes": "Cannot see {side} {joint} clearly. Please adjust position."
}}

GUIDELINES:
- Only measure movements you can confidently estimate from the visible angle
- Angles are measured from anatomical zero position (0° = fully extended/neutral)
- Report confidence: high (clear view, obvious angle), medium (partially visible), low (estimated)
- Be conservative — underestimate rather than overestimate capability"""

    def parse_vlm_response(self, response_text: str) -> Optional[dict]:
        """Parse VLM response into structured ROM data."""
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                logger.warning(f"No JSON found in VLM response: {response_text[:200]}")
                return None

            data = json.loads(json_match.group())

            if not isinstance(data, dict):
                return None
            if "joint" not in data or "measurements" not in data:
                return None

            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ROM JSON: {e}, response: {response_text[:200]}")
            return None

    def record_measurement(self, parsed_data: dict) -> list[dict]:
        """Record measurements from parsed VLM response and return enriched results."""
        if not parsed_data or not parsed_data.get("visible", False):
            return []

        joint = parsed_data.get("joint", self.current_joint)
        side = parsed_data.get("side", self.current_side)
        results = []

        for m in parsed_data.get("measurements", []):
            movement = m.get("movement", "unknown")
            angle = m.get("angle_degrees", 0)
            confidence = m.get("confidence", "low")

            ref = ROM_REFERENCE.get(joint, {})
            movement_ref = ref.get(movement)
            target = movement_ref[1] if movement_ref else 0

            measurement = ROMMeasurement(
                joint=joint,
                movement=movement,
                angle=float(angle),
                target_angle=float(target),
                timestamp=time.time(),
                confidence=confidence,
                side=side,
                notes=parsed_data.get("notes", ""),
            )
            self.measurements.append(measurement)

            result = measurement.to_dict()
            result["body_position"] = parsed_data.get("body_position", "")
            result["status"] = self._get_status_label(angle, target)
            result["color"] = self._get_status_color(angle, target)
            results.append(result)

        return results

    def _get_status_label(self, angle: float, target: float) -> str:
        if target <= 0:
            return "N/A"
        pct = (angle / target) * 100
        if pct >= 90:
            return "Excellent"
        elif pct >= 70:
            return "Good"
        elif pct >= 50:
            return "Limited"
        else:
            return "Restricted"

    def _get_status_color(self, angle: float, target: float) -> str:
        if target <= 0:
            return "#999"
        pct = (angle / target) * 100
        if pct >= 90:
            return "#76B900"  # NVIDIA green
        elif pct >= 70:
            return "#8BC34A"
        elif pct >= 50:
            return "#FFA726"
        else:
            return "#EF5350"

    def get_progress(self, joint: Optional[str] = None, movement: Optional[str] = None) -> dict:
        """Calculate progress over time for a specific joint/movement."""
        filtered = self.measurements
        if joint:
            filtered = [m for m in filtered if m.joint == joint]
        if movement:
            filtered = [m for m in filtered if m.movement == movement]

        if not filtered:
            return {"has_data": False}

        first = filtered[0]
        last = filtered[-1]
        best = max(filtered, key=lambda x: x.angle)

        return {
            "has_data": True,
            "first_angle": first.angle,
            "latest_angle": last.angle,
            "best_angle": best.angle,
            "improvement": round(last.angle - first.angle, 1),
            "total_measurements": len(filtered),
            "time_span_seconds": last.timestamp - first.timestamp,
            "target": last.target_angle,
        }

    def get_history(
        self,
        joint: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Get measurement history, optionally filtered by joint."""
        filtered = self.measurements
        if joint:
            filtered = [m for m in filtered if m.joint == joint]
        return [m.to_dict() for m in filtered[-limit:]]

    def get_reference_data(self, joint: Optional[str] = None) -> dict:
        """Get clinical ROM reference data."""
        if joint and joint in ROM_REFERENCE:
            return {joint: {k: {"min": v[0], "max": v[1], "unit": v[2]} for k, v in ROM_REFERENCE[joint].items()}}
        return {
            j: {k: {"min": v[0], "max": v[1], "unit": v[2]} for k, v in movements.items()}
            for j, movements in ROM_REFERENCE.items()
        }

    def clear_history(self):
        """Clear all measurement history."""
        self.measurements.clear()
        self.session_start = time.time()

    def get_session_summary(self) -> dict:
        """Get a summary of the current session."""
        if not self.measurements:
            return {"has_data": False, "duration_seconds": time.time() - self.session_start}

        joints_measured = set(m.joint for m in self.measurements)
        return {
            "has_data": True,
            "total_measurements": len(self.measurements),
            "joints_measured": list(joints_measured),
            "duration_seconds": round(time.time() - self.session_start, 1),
            "session_start": self.session_start,
        }
