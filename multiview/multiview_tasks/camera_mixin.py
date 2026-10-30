from multiview_tasks.cameras import MOUNTED_CAMERAS


class MultiViewCameraMixin:
    """Adds the wrist and shoulder cameras to a robocasa Tabletop scene.

    Tabletop attaches every entry of ``self._cam_configs`` that has a ``parent_body``
    to that body when it edits the model XML, so registering them here is enough.
    Cameras do not affect physics, so the scene is identical across camera profiles.
    """

    def set_cameras(self):
        super().set_cameras()
        for spec in MOUNTED_CAMERAS:
            self._cam_configs[spec.name] = dict(
                pos=list(spec.pos),
                quat=list(spec.quat),
                parent_body=spec.parent_body,
                camera_attribs=dict(fovy=str(spec.fovy)),
            )
