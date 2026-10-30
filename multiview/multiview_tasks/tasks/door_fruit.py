"""Hold the Door, Take the Fruit (the HoldDoorTakeBlockV3 family, with groceries).

The HoldDoorTakeBlockV3 scene (left hand holding the full-height door open, one row of three
items 20 cm apart inside the cabinet) with the colored blocks replaced by three different
fruits and vegetables from RoboCasa's objaverse object set, drawn per episode from FRUITS
(category and mesh instance). The instruction names one by category: "take the <fruit> out
of the cabinet". The right hand takes it out and puts it down on the work surface.

Visibility and cameras are those of HoldDoorTakeBlockV3: the held door hides the interior
from the head camera; the wrist-under cameras see it once the hand is at the opening, the
right-side door cameras from the start. What has to be perceived is which item is which
(shape, color, texture) rather than a block color.

Grasping: as for the blocks, a closed right hand with an item's center near its grasp point
holds the item by a weld.
"""

import numpy as np

from multiview_tasks.tasks.door_hold import HoldDoorTakeBlockV3
from multiview_tasks.variant import make_level_classes

# Roughly round items of similar size (5-8 cm) that the palm-down hand closes around.
FRUITS = ("apple", "orange", "lemon", "lime", "peach", "potato")


class HoldDoorTakeFruitV3(HoldDoorTakeBlockV3):
    TASK_NAME = "HoldDoorTakeFruitV3"
    # The row 3 cm further from the hinge (V3: 0.10 / 0.30 / 0.50) and 18 cm apart: with a 6-7 cm
    # item in the hinge-side slot the open fingers met the door and the hand came down on the
    # item's top, 5-6 cm from its center (no grasp).
    SLOTS = ((0.07, 0.13), (0.07, 0.31), (0.07, 0.49))
    SCRIPTED_SLOT_RANGES = {2: dict(outside=((-0.14, 0.02, 0.0), (-0.10, 0.05, 0.0)))}

    def _add_scene(self):
        self._fruit_objs, self._fruit_rest = {}, {}
        super()._add_scene()

    def _sample_items(self) -> list:
        return [str(c) for c in self.rng.choice(FRUITS, size=3, replace=False)]

    def _add_item(self, wb, fruit, floor_point, quat) -> str:
        """Add a sampled RoboCasa object of category `fruit`, resting on `floor_point`."""
        paths = self.layout.setdefault("mjcf", {})
        cfg = dict(name=f"mvfruit_{fruit}", obj_groups=fruit, obj_registries=("objaverse",))
        if fruit in paths:  # replaying a recorded episode: the same mesh instance
            cfg["info"] = dict(mjcf_path=paths[fruit])
        obj, info = self._create_obj(cfg)
        paths[fruit] = info["mjcf_path"]
        self.model.merge_objects([obj])
        self._fruit_objs[fruit] = obj
        # Rest pose, set at every reset (_place_items): bottom on the cabinet floor.
        self._fruit_rest[fruit] = (np.asarray(floor_point, dtype=float), np.asarray(quat, dtype=float))
        return obj.root_body

    def _create_obj(self, cfg):
        # Tabletop._create_obj samples from every registry; restrict to objaverse meshes.
        registries = cfg.pop("obj_registries", None)
        if registries is not None and "info" not in cfg:
            saved = self.obj_registries
            self.obj_registries = registries
            try:
                return super()._create_obj(cfg)
            finally:
                self.obj_registries = saved
        return super()._create_obj(cfg)

    def _center_offset(self, fruit) -> np.ndarray:
        """Bounding-box center relative to the object's body origin (object frame)."""
        obj = self._fruit_objs[fruit]
        return (np.asarray(obj.bottom_offset) + np.asarray(obj.top_offset)) / 2

    def item_half_height(self, fruit) -> float:
        obj = self._fruit_objs[fruit]
        return float(obj.top_offset[2] - obj.bottom_offset[2]) / 2

    def item_noun(self, fruit) -> str:
        return fruit

    def _item_body_id(self, fruit) -> int:
        return self.sim.model.body_name2id(self._fruit_objs[fruit].root_body)

    def block_pos(self, fruit) -> np.ndarray:
        """Item bounding-box center (world)."""
        data, b = self.sim.data, self._item_body_id(fruit)
        return data.body_xpos[b] + data.body_xmat[b].reshape(3, 3) @ self._center_offset(fruit)

    def _reset_internal(self):
        super()._reset_internal()
        self._place_items()

    def _place_items(self):
        sim = self.sim
        for fruit, obj in self._fruit_objs.items():
            floor, quat = self._fruit_rest[fruit]
            pos = floor - np.array([0.0, 0.0, obj.bottom_offset[2] - 0.002])
            sim.data.set_joint_qpos(obj.joints[0], np.concatenate([pos, quat]))
        sim.forward()


ENV_NAMES = make_level_classes(HoldDoorTakeFruitV3, __name__)
