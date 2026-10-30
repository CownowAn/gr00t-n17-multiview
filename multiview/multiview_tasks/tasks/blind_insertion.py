"""Task 1: Blind Receptacle Insertion.

Base task: PnPBottleToCabinetClose (pick the bottle, place it in the cabinet, close it).
Knob: how far the cabinet door starts open. A narrower gap hides the cabinet interior
and the arm/door clearance from egoview during insertion, while the wrist camera
still sees the hand, object and interior as it passes the opening.
"""

import numpy as np
from robocasa.environments.tabletop.tabletop_cabinet_pnp import (
    PnPBottleToCabinetClose,
    TabletopCabinetPnPClose,
)
from robocasa.utils.object_utils import obj_inside_of

from multiview_tasks.camera_mixin import MultiViewCameraMixin
from multiview_tasks.variant import VariantMixin, make_level_classes


class BlindInsertionBottleCabinet(
    VariantMixin, MultiViewCameraMixin, PnPBottleToCabinetClose
):
    TASK_NAME = "BlindInsertionBottleCabinet"
    # Initial door state range (fraction open). L0 is the original task.
    LEVELS = {
        0: dict(door_open=(0.90, 1.00)),
        1: dict(door_open=(0.60, 0.70)),
        2: dict(door_open=(0.45, 0.55)),
        3: dict(door_open=(0.30, 0.40)),
    }

    def _reset_internal(self):
        # Mirrors TabletopCabinetPnPClose._reset_internal with a configurable door range,
        # consuming the rng identically so L0 reproduces the original episodes.
        # The door starts open only in the "close" behavior that this task uses.
        assert self.behavior == "close"
        super(TabletopCabinetPnPClose, self)._reset_internal()
        lo, hi = self.variant_params["door_open"]
        self.cabinet.set_door_state(min=lo, max=hi, env=self, rng=self.rng)
        self._init_diagnostics()

    def _is_grasped(self):
        return any(
            self._check_grasp(gripper=self.robots[0].gripper[side], object_geoms=self.objects["obj"])
            for side in ("left", "right")
        )

    def _is_inserted(self):
        return obj_inside_of(env=self, obj_name="obj", fixture_id=self.cabinet, partial_check=True)

    def phase_signals(self):
        """Per-step phase flags; "inserted" uses the same inside check as _check_success."""
        return dict(
            grasped=self._is_grasped(),
            inserted=self._is_inserted(),
            door_closed=self.is_door_closed(),
        )

    # ------------------------------------------------------------------
    # Failure diagnostics (privileged state). All flags only count events that
    # happen before the bottle first enters the cabinet.
    # ------------------------------------------------------------------
    def _init_diagnostics(self):
        model = self.sim.model
        cab = self.cabinet.name
        body_names = [model.body_id2name(model.geom_bodyid[g]) or "" for g in range(model.ngeom)]
        self._robot_geoms = {
            g for g, b in enumerate(body_names) if b.startswith(("robot0_", "gripper0_"))
        }
        self._door_geoms = {
            g for g, b in enumerate(body_names) if b.startswith((f"{cab}_hingedoor", f"{cab}_door_"))
        }
        self._frame_geoms = {g for g, b in enumerate(body_names) if b == f"{cab}_main"}
        obj_body = self.objects["obj"].root_body
        self._obj_geoms = {g for g, b in enumerate(body_names) if b == obj_body}
        self._initial_door = self.cabinet.get_door_state(env=self)["door"]
        self._was_grasped = False
        self._was_inserted = False

    def _contact_pairs(self):
        data = self.sim.data
        for i in range(data.ncon):
            c = data.contact[i]
            yield c.geom1, c.geom2

    def _touching(self, geoms_a, geoms_b):
        return any(
            (g1 in geoms_a and g2 in geoms_b) or (g2 in geoms_a and g1 in geoms_b)
            for g1, g2 in self._contact_pairs()
        )

    def _obj_distance_to_interior(self):
        p0, px, py, pz = self.cabinet.get_int_sites(relative=False)
        obj = self.sim.data.body_xpos[self.obj_body_id[self.objects["obj"].name]]
        nearest = p0.copy()
        for edge in (px - p0, py - p0, pz - p0):
            t = np.clip(np.dot(obj - p0, edge) / np.dot(edge, edge), 0.0, 1.0)
            nearest += t * edge
        return float(np.linalg.norm(obj - nearest))

    def diagnostic_signals(self):
        grasped = self._is_grasped()
        self._was_grasped |= grasped
        self._was_inserted |= self._is_inserted()
        if self._was_inserted:
            return {}
        cabinet_geoms = self._door_geoms | self._frame_geoms
        obj_on_other_surface = any(
            (g1 in self._obj_geoms and g2 not in self._robot_geoms | cabinet_geoms | self._obj_geoms)
            or (g2 in self._obj_geoms and g1 not in self._robot_geoms | cabinet_geoms | self._obj_geoms)
            for g1, g2 in self._contact_pairs()
        )
        door = self.cabinet.get_door_state(env=self)["door"]
        return dict(
            robot_door_contact=self._touching(self._robot_geoms, self._door_geoms),
            robot_frame_contact=self._touching(self._robot_geoms, self._frame_geoms),
            obj_door_contact=self._touching(self._obj_geoms, self._door_geoms),
            obj_frame_contact=self._touching(self._obj_geoms, self._frame_geoms),
            door_pushed=door < self._initial_door - 0.1,
            reached_opening=self._obj_distance_to_interior() < 0.08,
            dropped=self._was_grasped and not grasped and obj_on_other_surface,
        )


ENV_NAMES = make_level_classes(BlindInsertionBottleCabinet, __name__)
