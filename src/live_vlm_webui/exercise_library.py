"""
Exercise Library for PT Rehab Coach
Defines exercises with form criteria, phase definitions, and VLM prompt templates.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ROMTarget:
    """ROM measurement target for an exercise."""
    joint: str
    movement: str
    side: str
    target_angle: float
    min_angle: float = 0.0


@dataclass
class Exercise:
    id: str
    name: str
    category: str  # "lower", "upper", "core", "stretch"
    description: str
    correct_form: str
    common_mistakes: list[str]
    phases: list[str]
    rep_start_phase: str
    rep_end_phase: str
    # MediaPipe joint angle tracking for rep counting
    # Tuple of (landmark_a, landmark_b, landmark_c) -- angle measured at b
    rom_targets: list[ROMTarget] = field(default_factory=list)
    primary_joint: Optional[tuple[str, str, str]] = None
    rep_down_threshold: float = 90.0
    rep_up_threshold: float = 150.0
    expected_objects: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def _rom_prompt_section(self) -> str:
        if not self.rom_targets:
            return ""
        targets_desc = []
        for rt in self.rom_targets:
            targets_desc.append(
                f"{rt.joint} {rt.movement} ({rt.side} side): target {rt.target_angle}\u00b0"
            )
        targets_str = "; ".join(targets_desc)
        rom_fields = ', '.join(
            f'"{rt.joint}_{rt.movement}_angle": <degrees>'
            for rt in self.rom_targets
        )
        return (
            f"\nROM MEASUREMENT \u2014 ALWAYS estimate the current joint angle in degrees, even if the form is wrong or incomplete:\n"
            f"  {targets_str}\n"
            f"IMPORTANT: Report the patient's ACTUAL current angle, not the ideal angle. "
            f"If a patient's knee is only bent to 40\u00b0, report 40, not the target.\n"
            f"Include in your JSON: {rom_fields}\n"
            f"Also include angle coaching in your feedback, e.g. 'Your knee is at 40\u00b0 \u2014 try to reach 90\u00b0. You need 50\u00b0 more.'\n"
        )

    def build_vlm_prompt(self) -> str:
        mistakes_str = "\n".join(f"  - {m}" for m in self.common_mistakes)
        phases_str = ", ".join(self.phases)
        rom_section = self._rom_prompt_section()
        objects_section = ""
        if self.expected_objects:
            objects = ", ".join(self.expected_objects)
            objects_section = (
                f"Expected objects/equipment for this exercise: {objects}.\n"
                "Identify whether these objects are visible and being used correctly.\n"
                "If a needed object is missing or used unsafely, include one clear correction.\n"
            )
        rom_fields = ""
        if self.rom_targets:
            rom_fields = ", " + ", ".join(
                f'"{rt.joint}_{rt.movement}_angle": <number or null>'
                for rt in self.rom_targets
            )
        return (
            f"You are an expert physical therapy coach. Analyze this image of a patient performing: {self.name}.\n"
            f"\n"
            f"Exercise: {self.name}\n"
            f"Description: {self.description}\n"
            f"Correct form: {self.correct_form}\n"
            f"Movement phases (in order): {phases_str}\n"
            f"Common mistakes to watch for:\n{mistakes_str}\n"
            f"{objects_section}"
            f"{rom_section}"
            f"\n"
            f"Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
            f'{{"exercise_detected": true/false, "phase": "<one of: {phases_str}>", '
            f'"form_score": <1-10>, "corrections": ["<specific correction>"], '
            f'"rep_boundary": true/false, "feedback": "<clear user-facing coaching message>"'
            f'{rom_fields}}}\n'
            f"\n"
            f'Feedback quality requirements when "exercise_detected" is true:\n'
            f"- Write 2-4 short sentences in plain language.\n"
            f"- Start with one concrete positive observation.\n"
            f"- Then give one highest-priority correction with a body part and direction.\n"
            f"- End with one next-step cue for the next rep.\n"
            f"- Mention object/equipment usage when relevant (e.g., wall distance, chair support, weight control).\n"
            f"- Be specific and actionable; avoid vague praise.\n"
            f"- Avoid repeating the exact same wording as prior responses unless necessary.\n"
            f"\n"
            f'Set "rep_boundary" to true ONLY when the person transitions from "{self.rep_end_phase}" back to "{self.rep_start_phase}" (one full rep just completed).\n'
            f'If you cannot see the person clearly or they are not exercising, set "exercise_detected" to false and '
            f'provide a short repositioning cue in "feedback" (for example: "Please step back so I can see your full body from head to feet.").'
        )


EXERCISES: list[Exercise] = [
    Exercise(
        id="general",
        name="General Coach (Auto-Detect)",
        category="general",
        description="The AI coach will automatically detect what exercise you are performing and provide posture tips and encouragement.",
        correct_form="Maintain good posture, controlled movements, and full range of motion.",
        common_mistakes=["Poor posture", "Rushing through movements", "Limited range of motion"],
        phases=["active"],
        rep_start_phase="start",
        rep_end_phase="end",
        primary_joint=None,
    ),
    Exercise(
        id="squat",
        name="Bodyweight Squat",
        category="lower",
        description="Stand with feet shoulder-width apart, lower hips back and down as if sitting in a chair, then return to standing.",
        correct_form="Feet shoulder-width apart, weight on heels, chest up, back straight, knees tracking over toes, thighs parallel to floor at bottom.",
        common_mistakes=[
            "Knees caving inward",
            "Heels lifting off the ground",
            "Leaning too far forward / rounding back",
            "Not going deep enough (thighs not reaching parallel)",
            "Knees extending past toes excessively",
        ],
        phases=["standing", "descending", "bottom", "ascending"],
        rep_start_phase="standing",
        rep_end_phase="ascending",
        rom_targets=[
            ROMTarget(joint="knee", movement="flexion", side="both", target_angle=135),
            ROMTarget(joint="hip", movement="flexion", side="both", target_angle=120),
        ],
        primary_joint=("left_hip", "left_knee", "left_ankle"),
        rep_down_threshold=100,
        rep_up_threshold=155,
    ),
    Exercise(
        id="lunge",
        name="Forward Lunge",
        category="lower",
        description="Step forward with one leg, lower hips until both knees are bent at about 90 degrees, then push back to starting position.",
        correct_form="Step forward with a long stride, front knee over ankle (not past toes), back knee approaching floor, torso upright, core engaged.",
        common_mistakes=[
            "Front knee extending past toes",
            "Torso leaning forward",
            "Back knee not bending enough",
            "Losing balance / wobbling side to side",
            "Stance too narrow (feet in a line)",
        ],
        phases=["standing", "stepping", "lowered", "returning"],
        rep_start_phase="standing",
        rep_end_phase="returning",
        rom_targets=[
            ROMTarget(joint="knee", movement="flexion", side="both", target_angle=90),
            ROMTarget(joint="hip", movement="flexion", side="both", target_angle=90),
        ],
        primary_joint=("left_hip", "left_knee", "left_ankle"),
        rep_down_threshold=100,
        rep_up_threshold=155,
    ),
    Exercise(
        id="wall_pushup",
        name="Wall Push-Up",
        category="upper",
        description="Stand facing a wall at arm's length, place palms flat on wall at shoulder height, bend elbows to bring chest toward wall, then push back.",
        correct_form="Arms shoulder-width apart on wall, body in a straight line from head to heels, elbows bending to about 90 degrees, controlled movement.",
        common_mistakes=[
            "Sagging hips (body not in straight line)",
            "Flaring elbows out to the sides",
            "Not bending elbows enough",
            "Head dropping forward",
            "Standing too close or too far from wall",
        ],
        expected_objects=["wall"],
        phases=["extended", "lowering", "chest_near_wall", "pushing_back"],
        rep_start_phase="extended",
        rep_end_phase="pushing_back",
        rom_targets=[
            ROMTarget(joint="elbow", movement="flexion", side="both", target_angle=90),
        ],
        primary_joint=("left_shoulder", "left_elbow", "left_wrist"),
        rep_down_threshold=100,
        rep_up_threshold=155,
    ),
    Exercise(
        id="shoulder_raise",
        name="Shoulder Raise",
        category="upper",
        description="Stand with arms at sides, raise both arms out to the sides (lateral raise) or forward until shoulder height, then lower slowly.",
        correct_form="Stand tall, slight bend in elbows, raise arms smoothly to shoulder height, palms facing down, controlled lowering.",
        common_mistakes=[
            "Shrugging shoulders up toward ears",
            "Swinging arms / using momentum",
            "Raising arms above shoulder height",
            "Arching the back",
            "Bending the wrists",
        ],
        expected_objects=["dumbbells (optional)"],
        phases=["arms_down", "raising", "arms_up", "lowering"],
        rep_start_phase="arms_down",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="shoulder", movement="abduction", side="both", target_angle=90),
        ],
        primary_joint=("left_hip", "left_shoulder", "left_wrist"),
        rep_down_threshold=30,
        rep_up_threshold=70,
    ),
    Exercise(
        id="calf_raise",
        name="Calf Raise",
        category="lower",
        description="Stand with feet hip-width apart (optionally holding a chair for balance), rise up onto the balls of your feet, then lower back down.",
        correct_form="Feet hip-width apart, rise straight up (no leaning), full extension at the top, slow controlled descent, knees slightly soft.",
        common_mistakes=[
            "Rolling ankles outward",
            "Not rising high enough",
            "Bending at the hips or knees",
            "Going too fast (no control on the way down)",
            "Leaning forward",
        ],
        expected_objects=["chair (optional support)"],
        phases=["flat", "rising", "top", "lowering"],
        rep_start_phase="flat",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="ankle", movement="plantarflexion", side="both", target_angle=50),
        ],
        primary_joint=("left_hip", "left_knee", "left_ankle"),
        rep_down_threshold=160,
        rep_up_threshold=172,
    ),
    Exercise(
        id="seated_knee_ext",
        name="Seated Knee Extension",
        category="lower",
        description="Sit on a chair with back supported, straighten one leg out in front until fully extended, hold briefly, then lower slowly.",
        correct_form="Sit tall with back against chair, extend leg fully squeezing the quad at the top, slow controlled return, keep thigh on chair.",
        common_mistakes=[
            "Leaning back or slouching",
            "Swinging the leg (using momentum)",
            "Not fully extending the knee",
            "Lifting the thigh off the chair",
            "Lowering too quickly",
        ],
        expected_objects=["chair"],
        phases=["knee_bent", "extending", "fully_extended", "lowering"],
        rep_start_phase="knee_bent",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="knee", movement="extension", side="right", target_angle=0, min_angle=90),
        ],
        primary_joint=("left_hip", "left_knee", "left_ankle"),
        rep_down_threshold=100,
        rep_up_threshold=155,
    ),
    Exercise(
        id="leg_raise",
        name="Side-Lying Leg Raise",
        category="lower",
        description="Lie on your side with legs stacked, raise the top leg toward the ceiling, then lower it back down with control.",
        correct_form="Body in a straight line, hips stacked (don't roll forward/back), raise leg to about 45 degrees, slow and controlled movement.",
        common_mistakes=[
            "Rolling hips forward or backward",
            "Bending the knee of the raised leg",
            "Raising the leg too high (above 45 degrees)",
            "Using momentum / jerky movements",
            "Not keeping the body in a straight line",
        ],
        phases=["legs_together", "raising", "top", "lowering"],
        rep_start_phase="legs_together",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="hip", movement="abduction", side="right", target_angle=45),
        ],
        primary_joint=("left_shoulder", "left_hip", "left_ankle"),
        rep_down_threshold=155,
        rep_up_threshold=170,
    ),
    Exercise(
        id="hip_abduction",
        name="Standing Hip Abduction",
        category="lower",
        description="Stand holding a chair for balance, lift one leg out to the side keeping it straight, then return to center.",
        correct_form="Stand tall, keep hips level, raise leg to the side 30-45 degrees, toes pointing forward, controlled return.",
        common_mistakes=[
            "Leaning the torso to the opposite side",
            "Rotating the hip / turning the toes up",
            "Bending the knee of the moving leg",
            "Moving too fast / swinging",
            "Not standing tall (slouching)",
        ],
        expected_objects=["chair (optional support)"],
        phases=["legs_together", "lifting", "leg_out", "returning"],
        rep_start_phase="legs_together",
        rep_end_phase="returning",
        rom_targets=[
            ROMTarget(joint="hip", movement="abduction", side="right", target_angle=45),
        ],
        primary_joint=("left_shoulder", "left_hip", "left_ankle"),
        rep_down_threshold=155,
        rep_up_threshold=170,
    ),
    Exercise(
        id="bicep_curl",
        name="Bicep Curl",
        category="upper",
        description="Stand or sit with arms at sides holding light weights (or no weight), curl forearms up toward shoulders by bending at the elbows, then lower.",
        correct_form="Elbows pinned at sides, curl with control, full range of motion, don't swing the body, squeeze at the top.",
        common_mistakes=[
            "Swinging the body to generate momentum",
            "Elbows drifting forward or backward",
            "Not using full range of motion",
            "Lowering too fast (no eccentric control)",
            "Shrugging shoulders",
        ],
        expected_objects=["dumbbells or resistance bands (optional)"],
        phases=["arms_extended", "curling", "top", "lowering"],
        rep_start_phase="arms_extended",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="elbow", movement="flexion", side="both", target_angle=150),
        ],
        primary_joint=("left_shoulder", "left_elbow", "left_wrist"),
        rep_down_threshold=50,
        rep_up_threshold=140,
    ),
    Exercise(
        id="tennis_ball_squeeze",
        name="Seated Tennis Ball Squeeze",
        category="lower",
        description="Sit tall in a chair with a tennis ball between your knees, gently squeeze the ball, then release with control.",
        correct_form="Keep feet flat, knees aligned with hips, torso upright, and squeeze from the inner thighs without leaning or holding breath.",
        common_mistakes=[
            "Collapsing posture or leaning forward",
            "Feet lifting off the floor",
            "Pressing with the ankles instead of the knees",
            "Holding the breath during the squeeze",
            "Releasing too quickly without control",
        ],
        expected_objects=["chair", "tennis ball"],
        phases=["relaxed", "squeezing", "hold", "releasing"],
        rep_start_phase="relaxed",
        rep_end_phase="releasing",
        primary_joint=None,
    ),
    Exercise(
        id="wall_slide_towel",
        name="Wall Slide with Towel",
        category="upper",
        description="Stand facing a wall with forearms on a towel, slide arms upward while keeping light pressure on the wall, then return to start.",
        correct_form="Keep ribs down, shoulders relaxed, elbows and forearms in contact with the towel, and move slowly without arching the lower back.",
        common_mistakes=[
            "Arching the lower back while reaching overhead",
            "Shrugging shoulders toward the ears",
            "Losing forearm contact with the wall",
            "Moving too quickly or bouncing",
            "Standing too far from the wall",
        ],
        expected_objects=["wall", "towel"],
        phases=["arms_low", "sliding_up", "arms_high", "sliding_down"],
        rep_start_phase="arms_low",
        rep_end_phase="sliding_down",
        rom_targets=[
            ROMTarget(joint="shoulder", movement="flexion", side="both", target_angle=150),
        ],
        primary_joint=("left_hip", "left_shoulder", "left_wrist"),
        rep_down_threshold=45,
        rep_up_threshold=120,
    ),
    Exercise(
        id="water_bottle_press",
        name="Seated Water Bottle Overhead Press",
        category="upper",
        description="Sit on a chair holding water bottles at shoulder level, press arms overhead, then lower back down slowly.",
        correct_form="Sit tall with core engaged, wrists stacked over elbows, press straight up without locking out forcefully, and lower with control.",
        common_mistakes=[
            "Arching the back during the press",
            "Pressing one arm faster than the other",
            "Shrugging shoulders excessively",
            "Dropping elbows too quickly on the way down",
            "Holding breath through each rep",
        ],
        expected_objects=["chair", "water bottles (or light dumbbells)"],
        phases=["rack_position", "pressing", "overhead", "lowering"],
        rep_start_phase="rack_position",
        rep_end_phase="lowering",
        rom_targets=[
            ROMTarget(joint="shoulder", movement="flexion", side="both", target_angle=170),
            ROMTarget(joint="elbow", movement="extension", side="both", target_angle=170),
        ],
        primary_joint=("left_shoulder", "left_elbow", "left_wrist"),
        rep_down_threshold=90,
        rep_up_threshold=160,
    ),
    Exercise(
        id="neck_rotation",
        name="Neck Rotation",
        category="stretch",
        description="Slowly turn your head to look over one shoulder, hold briefly, return to center, then turn to the other side.",
        correct_form="Sit or stand tall, move slowly, turn head until gentle stretch is felt, keep shoulders relaxed and still, breathe normally.",
        common_mistakes=[
            "Moving too quickly / jerking the head",
            "Raising the shoulders",
            "Tilting the head (chin up or down) instead of pure rotation",
            "Forcing past comfortable range",
            "Holding breath",
        ],
        phases=["center", "turning_right", "right", "returning_center", "turning_left", "left", "returning_center_2"],
        rep_start_phase="center",
        rep_end_phase="returning_center_2",
        rom_targets=[
            ROMTarget(joint="neck", movement="rotation", side="both", target_angle=80),
        ],
    ),
]

EXERCISE_MAP: dict[str, Exercise] = {ex.id: ex for ex in EXERCISES}


def get_exercise(exercise_id: str) -> Optional[Exercise]:
    return EXERCISE_MAP.get(exercise_id)


def get_all_exercises() -> list[dict]:
    return [ex.to_dict() for ex in EXERCISES]


def get_exercises_by_category(category: str) -> list[dict]:
    return [ex.to_dict() for ex in EXERCISES if ex.category == category]
