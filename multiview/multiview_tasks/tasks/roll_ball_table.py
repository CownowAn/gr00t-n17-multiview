"""Stop the ball rolling toward you across one table: StopRollingCanTable with a ball (tennis
ball size, orange) instead of the can. Same table, start positions, directions, speed,
body cameras and false-start rule; the ball can roll off the blocking hand sideways, which
the can (lying on its side) cannot.

Success: as for the can (the ball stays on the table, on its own side of the midline, and
stops moving along its path; no hand moved before it started rolling).
"""

import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import array_to_string

from multiview_tasks.tasks.roll_can import CAN_RADIUS, SURFACE
from multiview_tasks.tasks.roll_can_table import PREFIX, StopRollingCanTable
from multiview_tasks.variant import make_level_classes

BALL_RADIUS = CAN_RADIUS  # 0.033 m (a tennis ball is 0.0335); the push's rolling spin uses CAN_RADIUS
BALL_MASS = 0.057
BALL_RGBA = "0.95 0.45 0.05 1"


class StopRollingBallTable(StopRollingCanTable):
    TASK_NAME = "StopRollingBallTable"

    def _add_can(self, name, start):
        """The ball, under the can's body and joint names (the rest of the task uses them)."""
        self.can_start[name] = start
        body = f"{PREFIX}can_{name}"
        ball = ET.SubElement(self.model.worldbody, "body", name=body, pos=array_to_string(start))
        ET.SubElement(ball, "freejoint", name=f"{body}_joint")
        i = 0.4 * BALL_MASS * BALL_RADIUS**2  # solid sphere
        ET.SubElement(ball, "inertial", pos="0 0 0", mass=str(BALL_MASS), diaginertia=f"{i} {i} {i}")
        ET.SubElement(ball, "geom", name=f"{body}_geom", type="sphere", size=f"{BALL_RADIUS}", rgba=BALL_RGBA,
                      group="1", **SURFACE)

    def _can_quat(self, name):
        return [1.0, 0.0, 0.0, 0.0]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = "stop the rolling ball"
        return ep_meta


ENV_NAMES = make_level_classes(StopRollingBallTable, __name__)
