"""GR1 multi-view modality configs for GR00T N1.7 fine-tuning (cam1 / cam3* / cam5*).

Everything except the video keys copies N1.7's ``robocasa_gr1_tabletop`` config
(gr00t/configs/data/embodiment_configs.py): arms, hands and waist as state (sin/cos
encoded) and action, an 8-step action target, arms and hands relative, waist absolute,
language from the dataset's task string. N1.7 registers that tag already, so these are
registered under NEW_EMBODIMENT, which shares its projector slot (10). Video keys are the
eval env's obs keys without "video." (multiview_tasks/cameras.py, CAMERA_PROFILES).

Used through the per-profile files next to this one, e.g.
    --embodiment-tag NEW_EMBODIMENT --modality-config-path multiview/finetune_n17/modality_cam5r.py
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

EGO = "ego_view_bg_crop_pad_res256_freq20"
VIDEO_KEYS = {
    "cam1": [EGO],
    # HoldDoorTakeBlockV3
    "cam3": [EGO, "left_wrist_under_view", "right_wrist_under_view"],
    "cam5d": [EGO, "left_wrist_under_view", "right_wrist_under_view", "door_low_right", "door_high_right"],
    # ColorButtonBoxV3
    "cam3b": [EGO, "left_wrist_inner", "right_wrist_inner"],
    "cam5b": [EGO, "left_wrist_inner", "right_wrist_inner", "side_left", "side_right"],
    # StopRollingCanTable / StopRollingBallTable
    "cam3r": [EGO, "roll_diag_left", "roll_diag_right"],
    "cam5r": [EGO, "roll_diag_left", "roll_diag_right", "roll_side_left", "roll_side_right"],
    # PushDrawerBoxV3 (cam3 = wrist-under pair as above; HoldDoorTakeFruitV3 uses cam3 / cam5d)
    "cam5w": [EGO, "left_wrist_under_view", "right_wrist_under_view", "side_left", "side_right"],
}
PARTS = ["left_arm", "right_arm", "left_hand", "right_hand", "waist"]
ACTION_HORIZON = 8  # N1.7 robocasa_gr1_tabletop


def gr1_modality_config(profile: str) -> dict:
    rel = ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
    ab = ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=VIDEO_KEYS[profile]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=PARTS, sin_cos_embedding_keys=PARTS),
        "action": ModalityConfig(delta_indices=list(range(ACTION_HORIZON)), modality_keys=PARTS,
                                 action_configs=[rel, rel, rel, rel, ab]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def register(profile: str) -> None:
    register_modality_config(gr1_modality_config(profile), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
