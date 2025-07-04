# Joint names and angle triplets

JOINT_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_pinky","right_pinky","left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle",
    "left_heel","right_heel","left_foot_index","right_foot_index"
]

ANGLE_JOINTS = [
    ("left_shoulder","left_elbow","left_wrist"),
    ("right_shoulder","right_elbow","right_wrist"),
    ("left_elbow","left_wrist","left_index"),
    ("right_elbow","right_wrist","right_index"),
    ("left_hip","left_knee","left_ankle"),
    ("right_hip","right_knee","right_ankle"),
    ("left_knee","left_ankle","left_foot_index"),
    ("right_knee","right_ankle","right_foot_index"),
    ("right_hip","right_shoulder","right_elbow"),
    ("left_hip","left_shoulder","left_elbow"),
    ("right_hip","left_hip","left_knee"),
    ("left_hip","right_hip","right_knee")
]

def get_angle_indices():
    from constants import JOINT_NAMES, ANGLE_JOINTS
    return [(JOINT_NAMES.index(a), JOINT_NAMES.index(b), JOINT_NAMES.index(c))
            for (a,b,c) in ANGLE_JOINTS]