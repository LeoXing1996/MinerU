import numpy as np
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt

from .basic_shape import DefinitedBasicShape
from .types import Arc


class OvalShape(DefinitedBasicShape):
    NAME = 'OVAL'
    NUM_EDGES = 0
    NUM_CURVES = 1
    TRANSFORMS = [
        # [theta, flip_h, flip_v],
        [0, False, False],
        [np.pi / 2, False, False],
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
            rx = self.wr * 0.5
            ry = self.hr * 0.5
        else:
            rx = self.w * 0.5
            ry = self.h * 0.5

        A = Arc(-np.pi, np.pi, np.array([0, 0]), rx, ry)
        return [A]

    def draw_helper(self, prs, slide, params):
        # 定义圆形的尺寸和位置
        left = Pt(self.cx - self.w * 0.5)
        top = Pt(self.cy - self.h * 0.5)
        width = Pt(self.w)
        height = Pt(self.h)

        # 在幻灯片上添加一个矩形
        shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, width, height)

        shape.rotation = self.theta * 180 / np.pi

        if self.flip_h:
            # shape.flip_horizontally()
            pass

        if self.flip_v:
            # shape.flip_vertically()
            pass

        return shape
