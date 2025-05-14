import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Line


class RightTriangleShape(DefinitedBasicShape):
    NAME = 'RIGHT_TRIANGLE'
    NUM_EDGES = 3
    NUM_CURVES = 0
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
        self.adjustments = []
        return self.adjustments

    def set_bounds(self):
        return []

    def set_constraints(self):
        return []

    def generate_segments(self, params):
        assert len(params) == 0

        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        V1 = np.array([-w / 2, -h / 2])  # (0,0)
        V2 = np.array([w / 2, -h / 2])  # (w,0)
        V3 = np.array([-w / 2, h / 2])  # (0,h)

        L1 = Line(V1, V2)
        L2 = Line(V2, V3)
        L3 = Line(V3, V1)

        return [L1, L2, L3]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_TRIANGLE, left, top, width, height
        )

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            # shape.flip_horizontally()
            pass

        if self.flip_v:
            # shape.flip_vertically()
            pass

        return shape
