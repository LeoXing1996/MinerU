import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Line, Arc


class RoundedRectangleShape(DefinitedBasicShape):
    NAME = 'ROUNDED_RECTANGLE'
    NUM_EDGES = 4
    NUM_CURVES = 4
    TRANSFORMS = [
        # [theta, flip_h, flip_v],
        [0, False, False],
        [np.pi / 2, False, False],
    ]

    def set_adjustable_params(self):
        self.adjustments = [0.1]
        return self.adjustments

    def set_bounds(self):
        bounds = [
            (0, 0.5),
        ]
        return bounds

    def set_constraints(self):
        return []

    def generate_segments(self, params):
        assert len(params) == 1

        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        r = params[0] * min(w, h)

        V1 = np.array([-w / 2 + r, -h / 2])
        V2 = np.array([w / 2 - r, -h / 2])
        V3 = np.array([w / 2, -h / 2 + r])
        V4 = np.array([w / 2, h / 2 - r])
        V5 = np.array([w / 2 - r, h / 2])
        V6 = np.array([-w / 2 + r, h / 2])
        V7 = np.array([-w / 2, h / 2 - r])
        V8 = np.array([-w / 2, -h / 2 + r])

        L1 = Line(V1, V2)
        A2 = Arc(-np.pi / 2, 0, np.array([w / 2 - r, -h / 2 + r]), r, r, weight=5)
        L3 = Line(V3, V4)
        A4 = Arc(0, np.pi / 2, np.array([w / 2 - r, h / 2 - r]), r, r, weight=5)
        L5 = Line(V5, V6)
        A6 = Arc(np.pi / 2, np.pi, np.array([-w / 2 + r, h / 2 - r]), r, r, weight=5)
        L7 = Line(V7, V8)
        A8 = Arc(-np.pi, -np.pi / 2, np.array([-w / 2 + r, -h / 2 + r]), r, r, weight=5)

        return [L1, A2, L3, A4, L5, A6, L7, A8]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
        )

        shape.adjustments[0] = params[0]

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            # shape.flip_horizontally()
            pass

        if self.flip_v:
            # shape.flip_vertically()
            pass

        return shape
