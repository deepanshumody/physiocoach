# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications copyright (c) 2026 PhysioCoach team
# (Deepanshu Mody, Anagha Palandye, Taruni Nugooru).
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
PhysioCoach - Real-time AI physical-therapy coaching.

Built on NVIDIA's open-source live-vlm-webui (Apache-2.0), which provides the
WebRTC/VLM streaming server, GPU monitoring, and RTSP support. PhysioCoach adds
the physical-therapy layer: MediaPipe pose estimation, rep counting, range-of-
motion measurement, an exercise library, and a real-time coaching pipeline.
"""

__version__ = "0.1.0"
__author__ = "PhysioCoach team (Deepanshu Mody, Anagha Palandye, Taruni Nugooru)"
__license__ = "Apache-2.0"

from . import server
from . import video_processor
from . import gpu_monitor
from . import vlm_service
from . import exercise_library
from . import session_manager

__all__ = [
    "server",
    "video_processor",
    "gpu_monitor",
    "vlm_service",
    "exercise_library",
    "session_manager",
]
