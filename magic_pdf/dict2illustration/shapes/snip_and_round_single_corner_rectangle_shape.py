import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Line, Arc


class SnipAndRoundSingleCornerRectangleShape(DefinitedBasicShape):
    NAME = 'SNIP_ROUND_RECTANGLE'
    NUM_EDGES = 5
    NUM_CURVES = 1
    TRANSFORMS = [
        # [theta, flip_h, flip_v],
        [0, False, False],
        [0, False, True],
        [0, True, False],
        [0, True, True],
        [np.pi / 2, False, False],
        [np.pi / 2, False, True],
        [np.pi / 2, True, False],
        [np.pi / 2, True, True],
    ]

    def set_adjustable_params(self):
        self.adjustments = [0.0, 0.0]
        return self.adjustments

    def set_bounds(self):
        bounds = [
            (0, 0.5),  # r1
            (0, 0.5),  # r2
        ]
        return bounds

    def set_constraints(self):
        return []

    def generate_segments(self, params):
        assert len(params) == 2

        if self.normalize:
            w, h = self.wr, self.hr
        else:
            w, h = self.w, self.h

        s1 = params[0] * min(w, h)
        s2 = params[1] * min(w, h)

        V1 = np.array([-w / 2, -h / 2])
        V2 = np.array([w / 2, -h / 2])
        V3 = np.array([w / 2, h / 2 - s2])
        V4 = np.array([w / 2 - s2, h / 2])
        V5 = np.array([-w / 2 + s1, h / 2])
        V6 = np.array([-w / 2, h / 2 - s1])

        L1 = Line(V1, V2)
        L2 = Line(V2, V3)
        L3 = Line(V3, V4, weight=5)
        L4 = Line(V4, V5)
        A5 = Arc(
            np.pi / 2, np.pi, np.array([-w / 2 + s1, h / 2 - s1]), s1, s1, weight=5
        )
        L6 = Line(V6, V1)

        return [L1, L2, L3, L4, A5, L6]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(
            MSO_SHAPE.SNIP_ROUND_RECTANGLE, left, top, width, height
        )

        shape.adjustments[0] = params[0]
        shape.adjustments[1] = params[1]

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            pass
            # shape.flip_horizontally()

        if self.flip_v:
            pass
            # shape.flip_vertically()

        return shape
