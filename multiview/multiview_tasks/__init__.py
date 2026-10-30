"""Multi-view observability task variants for RoboCasa GR1 tabletop tasks.

Importing this package registers the robosuite env classes and the
``gr1_mv/<Task>_L<level>_<profile>`` gym IDs.
"""

from multiview_tasks.registry import register_variants
from multiview_tasks.tasks import TASK_ENV_NAMES

REGISTERED_IDS = [
    gym_id for env_names in TASK_ENV_NAMES.values() for gym_id in register_variants(env_names)
]
