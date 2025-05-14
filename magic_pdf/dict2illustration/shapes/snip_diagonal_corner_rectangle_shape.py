import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Line


class SnipDiagonalCornerRectangleShape(DefinitedBasicShape):
    NAME = 'SNIP_2_DIAG_RECTANGLE'
    NUM_EDGES = 8
    NUM_CURVES = 0
    TRANSFORMS = [
        # [theta, flip_h, flip_v],
        [0, False, False],
        [0, False, True],
        [0, True, False],
        [np.pi / 2, False, False],
        [np.pi / 2, True, False],
        [np.pi / 2, False, True],
    ]

    def set_adjustable_params(self):
        self.adjustments = [0.5, 0.5]
        return self.adjustments

    def set_bounds(self):
        bounds = [
            (0, 0.5),  # s1
            (0, 0.5),  # s2
        ]
        return bounds

    def set_constraints(self):
        return []

    def generate_segments(self, params):
        assert len(params) == 2

        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        s1 = params[0] * min(w, h)
        s2 = params[1] * min(w, h)

        V1 = np.array([-w / 2 + s2, -h / 2])
        V2 = np.array([w / 2 - s1, -h / 2])
        V3 = np.array([w / 2, -h / 2 + s1])
        V4 = np.array([w / 2, h / 2 - s2])
        V5 = np.array([w / 2 - s2, h / 2])
        V6 = np.array([-w / 2 + s1, h / 2])
        V7 = np.array([-w / 2, h / 2 - s1])
        V8 = np.array([-w / 2, -h / 2 + s2])

        L1 = Line(V1, V2)
        L2 = Line(V2, V3, weight=2)
        L3 = Line(V3, V4)
        L4 = Line(V4, V5, weight=2)
        L5 = Line(V5, V6)
        L6 = Line(V6, V7, weight=2)
        L7 = Line(V7, V8)
        L8 = Line(V8, V1, weight=2)

        return [L1, L2, L3, L4, L5, L6, L7, L8]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(
            MSO_SHAPE.SNIP_2_DIAG_RECTANGLE,  # snip same side corner rectangle
            left,
            top,
            width,
            height,
        )

        shape.adjustments[0] = params[0]
        shape.adjustments[1] = params[1]

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            # shape.flip_horizontally()
            pass

        if self.flip_v:
            # shape.flip_vertically()
            pass

        return shape
