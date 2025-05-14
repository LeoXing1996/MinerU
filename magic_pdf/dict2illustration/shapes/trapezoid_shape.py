import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Line


class TrapezoidShape(DefinitedBasicShape):
    NAME = 'TRAPEZOID'
    NUM_EDGES = 4
    NUM_CURVES = 0
    TRANSFORMS = [
        # [theta, flip_h, flip_v],
        [0, False, False],
        [0, False, True],
        [np.pi / 2, False, False],
        [np.pi / 2, True, False],
    ]

    def set_adjustable_params(self):
        self.adjustments = [0.1]
        return self.adjustments

    def set_bounds(self):
        bounds = [(0, 2)]
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

        # s = (params[0] * 0.5) * w
        # 如果s太小，会导致上底特别窄，
        s = (params[0]) * w

        V1 = np.array([-w / 2, -h / 2])
        V2 = np.array([w / 2, -h / 2])
        V3 = np.array([w / 2 - s, h / 2])
        V4 = np.array([-w / 2 + s, h / 2])

        L1 = Line(V1, V2)
        L2 = Line(V2, V3, weight=2)
        L3 = Line(V3, V4, weight=2)
        L4 = Line(V4, V1, weight=2)

        return [L1, L2, L3, L4]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(MSO_SHAPE.TRAPEZOID, left, top, width, height)

        shape.adjustments[0] = params[0]

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            # shape.flip_horizontally()
            pass

        if self.flip_v:
            # shape.flip_vertically()
            pass

        return shape
