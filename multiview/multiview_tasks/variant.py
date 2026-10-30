"""Difficulty levels for task variants.

A variant is an existing robocasa task plus one observability knob. Level 0 must
reproduce the original task exactly (same layout, language and success check), so
the original benchmark number is the reference point for every sweep.

robosuite only forwards a fixed set of constructor kwargs, so each level becomes
its own robosuite env class (``<Task>_L<level>``) carrying its parameters as a
class attribute. Defining the class registers it with robosuite.
"""

import sys


class VariantMixin:
    """Exposes the level parameters to the task and records them in ep_meta."""

    TASK_NAME: str = ""
    LEVELS: dict = {}  # level -> parameter dict
    LEVEL: int = 0

    @property
    def variant_params(self) -> dict:
        return self.LEVELS[self.LEVEL]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["mv_variant"] = dict(
            task=self.TASK_NAME, level=self.LEVEL, **self.variant_params
        )
        return ep_meta


def make_level_classes(task_cls: type, module_name: str) -> dict:
    """Create and register one robosuite env class per level of ``task_cls``.

    Returns {level: robosuite env name}.
    """
    module = sys.modules[module_name]
    env_names = {}
    for level in task_cls.LEVELS:
        name = f"{task_cls.TASK_NAME}_L{level}"
        level_cls = type(name, (task_cls,), {"LEVEL": level, "__module__": module_name})
        setattr(module, name, level_cls)
        env_names[level] = name
    return env_names
