"""
Exercise Library for PT Rehab Coach
Defines exercises with form criteria, phase definitions, and VLM prompt templates.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


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

    def to_dict(self) -> dict:
        return asdict(self)

    def build_vlm_prompt(self) -> str:
        mistakes_str = "\n".join(f"  - {m}" for m in self.common_mistakes)
        phases_str = ", ".join(self.phases)
        return (
            f"You are an expert physical therapy coach. Analyze this image of a patient performing: {self.name}.\n"
            f"\n"
            f"Exercise: {self.name}\n"
            f"Description: {self.description}\n"
            f"Correct form: {self.correct_form}\n"
            f"Movement phases (in order): {phases_str}\n"
            f"Common mistakes to watch for:\n{mistakes_str}\n"
            f"\n"
            f"Respond ONLY with a valid JSON object (no markdown, no extra text):\n"
            f'{{"exercise_detected": true/false, "phase": "<one of: {phases_str}>", '
            f'"form_score": <1-10>, "corrections": ["<specific correction>"], '
            f'"rep_boundary": true/false, "feedback": "<brief encouraging or corrective message>"}}\n'
            f"\n"
            f'Set "rep_boundary" to true ONLY when the person transitions from "{self.rep_end_phase}" back to "{self.rep_start_phase}" (one full rep just completed).\n'
            f"If you cannot see the person or they are not exercising, set exercise_detected to false."
        )


EXERCISES: list[Exercise] = [
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
        phases=["extended", "lowering", "chest_near_wall", "pushing_back"],
        rep_start_phase="extended",
        rep_end_phase="pushing_back",
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
        phases=["arms_down", "raising", "arms_up", "lowering"],
        rep_start_phase="arms_down",
        rep_end_phase="lowering",
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
        phases=["flat", "rising", "top", "lowering"],
        rep_start_phase="flat",
        rep_end_phase="lowering",
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
        phases=["knee_bent", "extending", "fully_extended", "lowering"],
        rep_start_phase="knee_bent",
        rep_end_phase="lowering",
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
        phases=["legs_together", "lifting", "leg_out", "returning"],
        rep_start_phase="legs_together",
        rep_end_phase="returning",
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
        phases=["arms_extended", "curling", "top", "lowering"],
        rep_start_phase="arms_extended",
        rep_end_phase="lowering",
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
    ),
]

EXERCISE_MAP: dict[str, Exercise] = {ex.id: ex for ex in EXERCISES}


def get_exercise(exercise_id: str) -> Optional[Exercise]:
    return EXERCISE_MAP.get(exercise_id)


def get_all_exercises() -> list[dict]:
    return [ex.to_dict() for ex in EXERCISES]


def get_exercises_by_category(category: str) -> list[dict]:
    return [ex.to_dict() for ex in EXERCISES if ex.category == category]
