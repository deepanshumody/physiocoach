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
Video Track Processor
Handles video frames, adds text overlays, and manages VLM processing
"""

import asyncio
import cv2
import numpy as np
from PIL import Image
from aiortc import VideoStreamTrack
from typing import Optional
import logging
import time
import av

from .vlm_service import VLMService
from .pose_detector import PoseDetector

# Enable swscaler warnings to track hardware acceleration status
# TODO: Implement hardware-accelerated color space conversion on Jetson using NVMM/VPI
av.logging.set_level(av.logging.WARNING)

logger = logging.getLogger(__name__)


class VideoProcessorTrack(VideoStreamTrack):
    """
    Video track that receives frames, sends them to VLM for analysis,
    and overlays responses on the video before sending back
    """

    # Class variable for frame processing interval (can be updated dynamically)
    process_every_n_frames = 30
    # Coaching mode uses a faster interval
    coaching_frame_interval = 15
    # Max allowed latency before dropping frames (in seconds, 0 = disabled)
    max_frame_latency = 0.0
    # Whether we are in active coaching mode
    _coaching_active = False
    # How often to run pose detection (every Nth frame, cheap ~10ms)
    pose_every_n_frames = 3

    def __init__(self, track: VideoStreamTrack, vlm_service: VLMService,
                 text_callback=None, pose_callback=None, camera_role: str = "front"):
        super().__init__()
        self.track = track
        self.vlm_service = vlm_service
        self.text_callback = text_callback
        self.pose_callback = pose_callback
        self.camera_role = camera_role
        self.camera_id = 1 if camera_role == "front" else 2
        self.coaching_prompt = None  # Per-track prompt set by server on session start
        self.pose_detector = PoseDetector()
        self.last_frame: Optional[np.ndarray] = None
        self.frame_count = 0
        self.dropped_frames = 0
        self.first_frame_pts = None  # Track first frame PTS to calculate relative time
        self.first_frame_time = None  # Wall clock time of first frame
        self.frame_time_base = None  # Time base for PTS conversion (e.g., 1/90000)

    async def recv(self):
        """
        Receive frame from input track, process it, and return with text overlay
        """
        try:
            # Get frame from incoming track
            frame = await self.track.recv()

            # Initialize timing on first frame
            if self.first_frame_pts is None and frame.pts is not None:
                self.first_frame_pts = frame.pts
                self.first_frame_time = time.time()
                # Store time_base for PTS conversion (e.g., 1/90000 for 90kHz clock)
                self.frame_time_base = float(frame.time_base)
                logger.info(
                    f"Latency tracking initialized: PTS={frame.pts}, time_base={frame.time_base} ({self.frame_time_base}s per tick)"
                )

            # Calculate actual frame age (latency) using PTS and time_base
            # Note: Some streams (like RTSP) may not have PTS set, so skip latency checks
            frame_latency = 0.0
            if frame.pts is not None and self.first_frame_pts is not None:
                # PTS is in time_base units, convert to seconds: pts * time_base
                frame_time_offset = (frame.pts - self.first_frame_pts) * self.frame_time_base
                expected_wall_time = self.first_frame_time + frame_time_offset
                current_time = time.time()
                frame_latency = current_time - expected_wall_time

            # Check for accumulated latency and drop old frames if needed (only if max_latency > 0)
            max_latency = self.__class__.max_frame_latency
            if max_latency > 0 and frame_latency > max_latency and frame.pts is not None:
                logger.warning(
                    f"Frame is {frame_latency:.2f}s behind, dropping frames (threshold: {max_latency}s)"
                )

                # Drop frames until we get a fresh one
                dropped_count = 0
                while frame_latency > max_latency:
                    self.dropped_frames += 1
                    dropped_count += 1

                    # Get next frame
                    frame = await self.track.recv()

                    # Recalculate latency for new frame (using time_base for correct conversion)
                    if frame.pts is not None and self.first_frame_pts is not None:
                        frame_time_offset = (
                            frame.pts - self.first_frame_pts
                        ) * self.frame_time_base
                        expected_wall_time = self.first_frame_time + frame_time_offset
                        frame_latency = time.time() - expected_wall_time
                    else:
                        # If PTS becomes unavailable, stop dropping frames
                        break

                    # Prevent infinite loop
                    if dropped_count > 100:
                        logger.error(
                            f"Dropped {dropped_count} frames, but still behind. Resetting timing."
                        )
                        if frame.pts is not None:
                            self.first_frame_pts = frame.pts
                            self.first_frame_time = time.time()
                            self.frame_time_base = float(frame.time_base)
                        break

                if dropped_count > 0:
                    logger.info(
                        f"Dropped {dropped_count} frames, now at {frame_latency:.2f}s latency"
                    )

            # Increment frame counter
            self.frame_count += 1

            cls = self.__class__
            coaching = cls._coaching_active
            vlm_interval = cls.coaching_frame_interval if coaching else cls.process_every_n_frames

            # Determine what work to do this frame
            need_vlm = (self.frame_count % vlm_interval == 0)
            need_pose = coaching and self.pose_detector.available and (self.frame_count % cls.pose_every_n_frames == 0)
            need_conversion = need_vlm or need_pose or (self.frame_count == 1)

            if need_conversion:
                t1 = time.time()
                img = frame.to_ndarray(format="bgr24")
                t2 = time.time()
                self.last_frame = img.copy()

                if self.frame_count % 300 == 0:
                    logger.info(f"Frame conversion: to_ndarray={1000*(t2-t1):.1f}ms")

                if self.frame_count == 1:
                    logger.info(f"First frame received: {img.shape}")

                # Pose detection (runs ~5-15ms on CPU)
                if need_pose:
                    pose_result = self.pose_detector.process_frame(img)
                    if self.pose_callback and pose_result.get("pose_detected"):
                        pose_result["camera_role"] = self.camera_role
                        self.pose_callback(pose_result)

                # VLM analysis (async, non-blocking, slow but smart)
                if need_vlm:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    prompt = self.coaching_prompt if (self._coaching_active and self.coaching_prompt) else None
                    asyncio.create_task(self.vlm_service.process_frame(pil_img, prompt=prompt))
                    if self.frame_count % 150 == 0:
                        logger.info(f"Frame {self.frame_count}: Sending to VLM (interval={vlm_interval})")

            # Get current response (may be old if VLM is still processing)
            response, is_processing = self.vlm_service.get_current_response()

            # Get metrics
            metrics = self.vlm_service.get_metrics()

            # Send text update via callback (for WebSocket)
            if self.text_callback:
                self.text_callback(response, metrics)

            # Return original frame directly - zero-copy passthrough!
            # This avoids expensive BGR→YUV conversion
            return frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            raise

    def _add_text_overlay(self, img: np.ndarray, text: str, status: str = "") -> np.ndarray:
        """
        Add text overlay to image

        Args:
            img: Input image (BGR format)
            text: Text to overlay (VLM response)
            status: Optional status text

        Returns:
            Image with text overlay
        """
        img_copy = img.copy()
        height, width = img_copy.shape[:2]

        # Prepare text
        full_text = f"{text} {status}" if status else text

        # Text wrapping - split long captions
        max_chars_per_line = 60
        words = full_text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_chars_per_line:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(" ".join(current_line))

        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White
        bg_color = (0, 0, 0)  # Black background
        padding = 10
        line_height = 30

        # Calculate total height needed
        total_text_height = len(lines) * line_height + 2 * padding

        # Create semi-transparent overlay at bottom
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, height - total_text_height), (width, height), bg_color, -1)

        # Blend overlay with original image
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

        # Add text lines
        y_position = height - total_text_height + padding + line_height
        for line in lines:
            cv2.putText(
                img_copy,
                line,
                (padding, y_position),
                font,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )
            y_position += line_height

        return img_copy
