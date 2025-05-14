import numpy as np
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.util import Pt
from pptx.oxml import parse_xml
from .basic_shape import DefinitedBasicShape
from .types import Line


class ArrowBaseShape(DefinitedBasicShape):
    NAME = 'ARROW_SHAPE'
    NUM_EDGES = 7  # 假设复杂箭头有8条边
    NUM_CURVES = 0  # 根据实际箭头结构调整
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
        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        # 定义箭头的关键点
        # 箭头的长度和宽度
        arrow_length = h * 0.2  # 箭头长度占总高度的30%
        arrow_width = w  # 箭头宽度占总宽度的20%

        # 主线段
        V1 = np.array([-w / 2, 0])  # 起点
        V2 = np.array([w / 2 - arrow_length, 0])  # 终点（箭头前）

        # 箭头部分
        V3 = np.array([w / 2 - arrow_length, -arrow_width / 2])  # 箭头左起点
        V4 = np.array([w / 2, 0])  # 箭头尖
        V5 = np.array([w / 2 - arrow_length, arrow_width / 2])  # 箭头右起点

        # 创建线段
        L1 = Line(V1, V2)  # 主线段
        L2 = Line(V2, V3)  # 箭头左斜线
        L3 = Line(V3, V4)  # 箭头左尖
        L4 = Line(V4, V5)  # 箭头右尖
        L5 = Line(V5, V2)  # 箭头右斜线

        return [L1, L2, L3, L4, L5]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        width = Pt(self.w)
        height = Pt(self.h)

        shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, width, height)

        return shape


class ArrowBar(DefinitedBasicShape):
    def __init__(self, arrow_end=True):
        super().__init__()
        self.arrow_end = arrow_end

    NAME = 'ARROW_BAR'
    NUM_EDGES = 1
    NUM_CURVES = 0
    TRANSFORMS = [
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
        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        V1 = np.array([-w / 2, -h / 2])
        V2 = np.array([w / 2, h / 2])
        L1 = Line(V1, V2)
        return [L1]

    def draw_helper(self, prs, slide, params):
        # left = Pt(self.cx - self.w/2)
        # top = Pt(self.cy - self.h/2)
        # width = Pt(self.w)
        # height = Pt(self.h)

        # shape = slide.shapes.add_shape(
        #     MSO_SHAPE.LINE_INVERSE,
        #     left, top,
        #     width, height
        # )

        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        right = Pt(self.cx + self.w / 2)
        bottom = Pt(self.cy + self.h / 2)

        shape = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, left, top, right, bottom
        )

        if self.arrow_end:
            # shape.line.tailEnd = ('type', 'arrow')
            line_elem = shape.line._get_or_add_ln()
            line_elem.append(
                parse_xml("""
                    <a:tailEnd type="arrow" size="sm" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
            """)
            )

        # line_elem = shape.line._get_or_add_ln()
        # line_elem.append(parse_xml("""
        #         <a:tailEnd type="arrow" size="sm" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>
        # """))

        return shape


class LineConnector(DefinitedBasicShape):
    NAME = 'LINE_CONNECTOR'
    NUM_EDGES = 1
    NUM_CURVES = 0
    TRANSFORMS = [
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
        if self.normalize:
            w = self.wr
            h = self.hr
        else:
            w = self.w
            h = self.h

        V1 = np.array([-w / 2, -h / 2])
        V2 = np.array([w / 2, h / 2])
        L1 = Line(V1, V2)
        return [L1]

    def draw_helper(self, prs, slide, params):
        left = Pt(self.cx - self.w / 2)
        top = Pt(self.cy - self.h / 2)
        right = Pt(self.cx + self.w / 2)
        bottom = Pt(self.cy + self.h / 2)

        shape = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, left, top, right, bottom
        )

        return shape
