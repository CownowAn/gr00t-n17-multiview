"""Task 6 (prototype): Catch the Ball Rolling in from the Side (field-of-view limit).

Same catching task as CatchRollingBall (one of two identical balls, left or right, is let
go at a random moment and rolls at a fixed speed; the hand on that side has to close on
it as it passes), but nothing hides the balls: each rolls toward a catch point beside
the robot (forward 0.43 m, 0.30 m to the side) from a ramp 0.6 m away, and the level sets
the direction it comes from, from ahead (30 deg) to behind the side (120 deg). The head camera looks forward
and has a limited field of view, so a ball coming from the side enters its image only
shortly before it arrives; cameras that look further out see it start. The waist is
locked, so the robot cannot turn to look.

Where the path runs is random (a sideways offset of the ball on its ramp, up to
PATH_OFFSET, so the ramp does not show it). A ball that is not caught rolls off the
surface's near edge or into a gap along the midline, away from the other hand.
"""

import numpy as np

from multiview_tasks.tasks import catch_ball as cb
from multiview_tasks.tasks.catch_ball import PREFIX, SIDES, SURFACE, CatchRollingBall
from multiview_tasks.variant import make_level_classes

PLATFORM_FORWARD = (0.18, 1.05)
PLATFORM_SIDE = (0.05, 1.02)  # |right| extent of each half; the gap between the halves catches stray balls
# Behind the near edge, beside the robot (clear of its arms), for balls coming from behind the side.
PLATFORM_BACK = (-0.10, 0.18)
PLATFORM_BACK_SIDE = (0.36, 1.02)

CATCH_POINT = (0.43, 0.30)  # (forward, |right|) the paths pass through at zero offset
START_DISTANCE = 0.60  # ball start from the catch point, along the path
PATH_OFFSET = 0.11  # max sideways offset of the path (m)
RAMP_DEG = 5.0
RAMP_LENGTH = 0.20
RAMP_START = 0.14  # the ball starts this far up the ramp (sets the speed, ~0.35 m/s)
RAMP_HALF_WIDTH = 0.11


class CatchSideBall(CatchRollingBall):
    TASK_NAME = "CatchSideBall"
    # The ball passes under the hand, out of every head-level camera's sight, in its last
    # ~0.25 s; a 0.5 s closing window lets a hand close on the last sight of it.
    GRASP_WINDOW_STEPS = 10
    GRASP_ON_CONTACT = True
    # approach_deg: the direction the ball comes from, measured at the catch point from
    # the robot's forward axis toward its side (90 = straight from the side).
    LEVELS = {
        0: dict(approach_deg=30.0),
        1: dict(approach_deg=60.0),
        2: dict(approach_deg=90.0),
        3: dict(approach_deg=120.0),
    }

    def _sample_layout(self):
        if "catch_layout" in self._ep_meta:  # replaying a recorded episode
            return dict(self._ep_meta["catch_layout"])
        offsets = self.rng.uniform(-PATH_OFFSET, PATH_OFFSET, size=2)
        return dict(
            offset_left=float(offsets[0]), offset_right=float(offsets[1]),
            side=str(self.rng.choice(list(SIDES))),
            release_step=int(self.rng.integers(cb.RELEASE_STEPS[0], cb.RELEASE_STEPS[1] + 1)),
        )

    def path(self, side, offset=None):
        """(point on the path nearest the catch point, unit direction of travel) in (forward, right)."""
        sgn = SIDES[side]
        th = np.deg2rad(self.variant_params["approach_deg"])
        u = np.array([np.cos(th), sgn * np.sin(th)])  # from the catch point toward the start
        n = np.array([-np.sin(th), sgn * np.cos(th)])  # sideways (positive: away from the robot / forward)
        off = self.layout[f"offset_{side}"] if offset is None else offset
        c = np.array([CATCH_POINT[0], sgn * CATCH_POINT[1]]) + off * n
        return c, -u

    def _add_scene(self):
        self.layout = lay = self._sample_layout()
        self.frame = self._frame()
        wb = self.model.worldbody
        f0, f1 = PLATFORM_FORWARD
        s0, s1 = PLATFORM_SIDE
        for side, sgn in SIDES.items():
            self._box(wb, f"{PREFIX}platform_{side}", self._world((f0 + f1) / 2, sgn * (s0 + s1) / 2, -cb.PLATFORM_HEIGHT / 2),
                      ((f1 - f0) / 2, (s1 - s0) / 2, cb.PLATFORM_HEIGHT / 2), "0.62 0.52 0.40 1", **SURFACE)
            b0, b1 = PLATFORM_BACK
            r0, r1 = PLATFORM_BACK_SIDE
            self._box(wb, f"{PREFIX}platform_back_{side}", self._world((b0 + b1) / 2, sgn * (r0 + r1) / 2, -cb.PLATFORM_HEIGHT / 2),
                      ((b1 - b0) / 2, (r1 - r0) / 2, cb.PLATFORM_HEIGHT / 2), "0.62 0.52 0.40 1", **SURFACE)
        a = np.deg2rad(RAMP_DEG)
        th = self.variant_params["approach_deg"]
        t = 0.004  # ramp plank half thickness
        mid = RAMP_LENGTH / 2
        low = START_DISTANCE - RAMP_START * np.cos(a)  # ramp's low edge from the catch point
        self.ball_start = {}
        for side, sgn in SIDES.items():
            c0, d = self.path(side, offset=0.0)
            u = -d
            # Ramp: rises along u, centered on the zero-offset path (so it does not show the offset).
            along = low + mid * np.cos(a) + t * np.sin(a)
            p = c0 + along * u
            self._box(wb, f"{PREFIX}ramp_{side}", self._world(p[0], p[1], mid * np.sin(a) - t * np.cos(a)),
                      (mid, RAMP_HALF_WIDTH, t), "0.80 0.80 0.76 1", pitch_deg=RAMP_DEG, turn_deg=sgn * th, **SURFACE)
            hi = RAMP_LENGTH * np.sin(a)
            q = c0 + (low + RAMP_LENGTH * np.cos(a) + 0.01) * u
            self._box(wb, f"{PREFIX}ramp_post_{side}", self._world(q[0], q[1], hi / 2), (0.01, RAMP_HALF_WIDTH, hi / 2),
                      "0.80 0.80 0.76 1", turn_deg=sgn * th)
            c, _ = self.path(side)
            b = c + (low + RAMP_START * np.cos(a) - cb.BALL_RADIUS * np.sin(a)) * u
            self._add_ball(side, self._world(b[0], b[1], RAMP_START * np.sin(a) + cb.BALL_RADIUS * np.cos(a)))
        self._add_grasp_welds()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if hasattr(self, "layout"):
            ep_meta["lang"] = "catch the rolling ball"
        return ep_meta


ENV_NAMES = make_level_classes(CatchSideBall, __name__)
