"""ColosseumV2 dual-arm Panda modality configs for GR00T N1.7 (NEW_EMBODIMENT).

Camera profiles: one camera alone (to see which view can carry the VLM path), and multi-view sets.
State: arm joints (sin/cos encoded) and finger widths. Action (16-step chunks at 20 Hz = 0.8 s):
arm joints relative to the current joint state, gripper command absolute (-1 close, 1 open).
Register one profile per run with a cv2_modality_<profile>.py file (--modality-config-path).
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

VIDEO_KEYS = {
    "ext1": ["external1"],
    "ext2": ["external2"],
    "lhand": ["left_hand"],
    "rhand": ["right_hand"],
    "ext12": ["external1", "external2"],
    "all4": ["external1", "external2", "left_hand", "right_hand"],
}
STATE = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
ACTION = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
ACTION_HORIZON = 16


def cv2_modality_config(profile: str) -> dict:
    rel = ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
    ab = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS[profile]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=STATE, sin_cos_embedding_keys=["left_arm", "right_arm"]),
        "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=ACTION,
                                 action_configs=[rel, ab, rel, ab]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def register(profile: str) -> None:
    register_modality_config(cv2_modality_config(profile), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
