import numpy as np
from abc import ABC, abstractmethod
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Pt
from pptx.shapes.connector import Connector

from .util import (
    get_bounding_rectangle,
    sample_segment,
    sample_arc,
    allocate_segment_points,
)
from .types import Line, Arc
from .color import parse_color
from .stroke import parse_stroke_width, parse_stroke_dashstyle


class BasicShape(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_init_params(self, target_points):
        pass

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def set_constraints(self):
        pass

    @abstractmethod
    def generate_segments(self, params):
        pass

    def generate(self, params, num_points=200):
        """
        根据shape的模型函数产生采样点

        参数：
        params = [cx, cy, w, h, theta, ...]
        cx, cy：shape的外接矩形中心坐标
        w, h：shape的外接矩形的宽和高
        theta：全局旋转角（弧度，逆时针）
        num_points：采样点总数，默认 200 个。
        返回：
        一个 (N x 2) 的数组，表示shape上的采样点。
        """
        cx, cy, theta = params[0], params[1], params[4]

        segments = self.generate_segments(params)
        segment_points = allocate_segment_points(segments, num_points)

        sampled_segments = []
        for i in range(len(segments)):
            s = segments[i]
            if isinstance(s, Line):
                seg = sample_segment(
                    s.start, s.end, segment_points[i], include_endpoint=True
                )
            elif isinstance(s, Arc):
                seg = sample_arc(
                    s.start,
                    s.end,
                    segment_points[i],
                    s.center,
                    s.radius,
                    include_endpoint=True,
                )
            else:
                raise TypeError(f'Unsupported Type {type(s)} of s')
            sampled_segments.append(seg)
        l_pts = np.concatenate(sampled_segments, axis=0)

        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        l_pts = (R @ l_pts.T).T
        g_pts = l_pts * np.array([1, -1]) + np.array([cx, cy])

        return g_pts

    @abstractmethod
    def draw_helper(self, prs, slide, params, ratio):
        pass

    def draw(self, params, save_name):
        # 创建一个新的演示文稿
        prs = Presentation()

        # 添加一个幻灯片（使用空白布局）
        slide_layout = prs.slide_layouts[5]  # 5 是空白幻灯片的布局索引
        slide = prs.slides.add_slide(slide_layout)

        shape = self.draw_helper(prs, slide, params)

        # 设置圆形的填充颜色和边框
        fill = shape.fill
        fill.solid()  # 设置填充为纯色
        fill.fore_color.rgb = RGBColor(255, 255, 255)  # 红色填充

        line = shape.line
        line.color.rgb = RGBColor(255, 0, 0)  # 蓝色边框
        line.width = Pt(2)  # 边框宽度为 2 磅

        # 保存演示文稿
        if save_name is not None:
            prs.save(save_name)
            print(f"已生成并保存为 '{save_name}'")


class DefinitedBasicShape(ABC):
    NAME = 'BASIC'
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

    def __init__(self):
        self.cx = 0
        self.cy = 0
        self.w = 0
        self.h = 0
        self.wr = 0
        self.hr = 0
        self.normalize = False
        self.adjustments = []
        self.attributes = None

        self._counter = 0

    def set_draw_params(
        self, cx, cy, w, h, rotation=0, flip_h=False, flip_v=False, normalize=False
    ):
        self.normalize = normalize
        self.cx = cx
        self.cy = cy
        self.w, self.h = w, h
        self.theta = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v

    def set_definited_params(
        self, target_points, rotation=0, flip_h=False, flip_v=False, normalize=False
    ):
        self.normalize = normalize

        cx, cy, w, h = get_bounding_rectangle(target_points)

        self.cx = cx
        self.cy = cy
        self.theta = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v

        if rotation == 0:
            self.w, self.h = w, h
        elif rotation in [np.pi / 2, 1.57, 1.5]:
            self.w, self.h = h, w
        else:
            raise ValueError(
                f'Unsupported rotation angle {rotation}. Supported values: {{0, np.pi/2}}'
            )

        if self.normalize:
            self.wr = self.w / max(self.w, self.h)
            self.hr = self.h / max(self.w, self.h)

    @abstractmethod
    def set_adjustable_params(self):
        pass

    def set_params(self, params):
        self.adjustments = params

    def set_transform(self, rotation, flip_h, flip_v):
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v

    def set_attributes(self, attributes):
        self.attributes = attributes

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def set_constraints(self):
        pass

    def to_dict(self):
        # 简化attributes，只保留draw函数中使用的属性
        simplified_attributes = {}

        if self.attributes:
            # 需要保留的属性列表
            important_attrs = [
                'fill',
                'fill-opacity',
                'stroke',
                'stroke-width',
                'stroke-opacity',
                'stroke-dasharray',
                'stroke-dashoffset',
            ]

            # 处理transform中的缩放因子
            if 'transform' in self.attributes and 'stroke-width' in self.attributes:
                transform = self.attributes['transform']
                if transform.startswith('matrix('):
                    try:
                        # 提取缩放因子
                        scale = float(transform.split(',')[0].replace('matrix(', ''))
                        # 直接应用缩放因子到stroke-width
                        stroke_width = float(self.attributes['stroke-width']) * scale
                        simplified_attributes['stroke-width'] = str(stroke_width)
                    except (ValueError, IndexError):
                        # 如果解析失败，保留原始值
                        if self.attributes['stroke-width'].lower() != 'none':
                            simplified_attributes['stroke-width'] = self.attributes[
                                'stroke-width'
                            ]
            elif (
                'stroke-width' in self.attributes
                and self.attributes['stroke-width'].lower() != 'none'
            ):
                simplified_attributes['stroke-width'] = self.attributes['stroke-width']

            # 复制其他重要属性，但忽略值为"none"的属性
            for attr in important_attrs:
                if (
                    attr in self.attributes and attr != 'stroke-width'
                ):  # 已经处理过stroke-width
                    attr_value = self.attributes[attr]
                    # 只有当属性值不是"none"时才添加该属性
                    if isinstance(attr_value, str) and attr_value.lower() != 'none':
                        simplified_attributes[attr] = attr_value
                    # elif not isinstance(attr_value, str):
                    #     simplified_attributes[attr] = attr_value

        return dict(
            name=self.NAME,
            params=dict(
                cx=self.cx,
                cy=self.cy,
                w=self.w,
                h=self.h,
                rotation=self.rotation,
                flip_v=self.flip_v,
                flip_h=self.flip_h,
                adjustments=self.adjustments,
            ),
            attributes=simplified_attributes,
        )

    @abstractmethod
    def generate_segments(self, params):
        pass

    def generate(self, params, num_points=200):
        """
        根据shape的模型函数产生采样点

        参数：
        params = [cx, cy, w, h, theta, ...]
        cx, cy：shape的外接矩形中心坐标
        w, h：shape的外接矩形的宽和高
        theta：全局旋转角（弧度，逆时针）
        num_points：采样点总数，默认 200 个。
        返回：
        一个 (N x 2) 的数组，表示shape上的采样点。
        """
        # 以下操作发生在(cx, cy) = (0, 0)的笛卡尔坐标系
        segments = self.generate_segments(params)
        segment_points = allocate_segment_points(segments, num_points)

        sampled_segments = []
        for i in range(len(segments)):
            s = segments[i]
            if isinstance(s, Line):
                seg = sample_segment(
                    s.start, s.end, segment_points[i], include_endpoint=True
                )
            elif isinstance(s, Arc):
                seg = sample_arc(
                    s.start,
                    s.end,
                    segment_points[i],
                    s.center,
                    s.rx,
                    s.ry,
                    include_endpoint=True,
                )
            else:
                raise TypeError(f'Unsupported Type {type(s)} of s')
            sampled_segments.append(seg)
        l_pts = np.concatenate(sampled_segments, axis=0)

        ############################################################
        # rotation, inverse-clockwise
        # R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
        #               [np.sin(self.theta),  np.cos(self.theta)]])
        # l_pts = (R @ l_pts.T).T
        ############################################################

        # rotation, clockwise
        R = np.array(
            [
                [np.cos(self.theta), np.sin(self.theta)],  # 原 -sin 改为 +sin
                [-np.sin(self.theta), np.cos(self.theta)],
            ]
        )  # 原 +sin 改为 -sin
        l_pts = (R @ l_pts.T).T

        # flip
        if self.flip_h:
            l_pts[:, 0] = l_pts[:, 0] * (-1)
        if self.flip_v:
            l_pts[:, 1] = l_pts[:, 1] * (-1)

        # 转换到图像坐标系并平移到真正的(cx, cy)
        if self.normalize:
            g_pts = l_pts * np.array([1, -1])
        else:
            g_pts = l_pts * np.array([1, -1]) + np.array([self.cx, self.cy])

        return g_pts

    @abstractmethod
    def draw_helper(self, prs, slide, params, ratio):
        pass

    def draw(self, prs, slide, save_name=None, params=None, attributes=None):
        params = params or self.adjustments
        attributes = attributes or self.attributes

        if prs is None:
            prs = Presentation()

        if slide is None:
            slide_layout = prs.slide_layouts[5]
            slide = prs.slides.add_slide(slide_layout)

        # 调整stroke-width
        if attributes and 'transform' in attributes and 'stroke-width' in attributes:
            transform = attributes['transform']
            if transform.startswith('matrix('):
                # 提取缩放因子
                scale = float(transform.split(',')[0].replace('matrix(', ''))
                # 调整stroke-width
                attributes['stroke-width'] = str(
                    float(attributes['stroke-width']) * scale
                )

        shape = self.draw_helper(prs, slide, params)
        if self.flip_h or self.flip_v:
            shape.rotation += np.pi * 180.0 / np.pi
        shape.shadow.inherit = False

        if attributes is None:
            # setup default attributes
            fill = shape.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(255, 255, 255)

            line = shape.line
            line.color.rgb = RGBColor(255, 0, 0)
            line.width = Pt(2)
        else:
            # print(attributes)

            fill = attributes.get('fill', None)
            if fill is None or fill.lower() == 'none':
                if isinstance(shape, Connector):
                    shape.line.color.rgb = RGBColor(0, 0, 0)
                else:
                    shape.fill.background()
                # shape.fill.background()
            else:
                r, g, b = parse_color(fill)
                if isinstance(shape, Connector):
                    shape.line.color.rgb = RGBColor(r, g, b)
                    fill_opacity = attributes.get('fill-opacity', 1.0)
                    shape.line.fill.transparency = fill_opacity
                else:
                    shape.fill.solid()
                    shape.fill.fore_color.rgb = RGBColor(r, g, b)
                    fill_opacity = attributes.get('fill-opacity', 1.0)
                    shape.fill.transparency = fill_opacity

            stroke = attributes.get('stroke', None)
            if stroke is None or stroke.lower() == 'none':
                shape.line.fill.background()
            else:
                r, g, b = parse_color(stroke)
                shape.line.color.rgb = RGBColor(r, g, b)

                stroke_width = attributes.get('stroke-width', Pt(1))
                shape.line.width = parse_stroke_width(str(stroke_width))

                if attributes.get('stroke-opacity', None) == '0':
                    shape.line.fill.background()

                # stroke_linecap = attributes.get("stroke-linecap", "butt")
                # stroke_linejoin = attributes.get("stroke-linejoin", "miter")
                # stroke_miterlimit = attributes.get("stroke-miterlimit", 4)
                # line.linecap = parse_stroke_linecap(stroke_linecap)
                # line.linejoin = parse_stroke_linejoin(stroke_linejoin)
                # line.miter_limit = parse_stroke_miterlimit(stroke_miterlimit)

                stroke_dasharray = attributes.get('stroke-dasharray', None)
                stroke_dashoffset = attributes.get('stroke-dashoffset', 0)
                shape.line.dash_style = parse_stroke_dashstyle(stroke_dasharray)

                # line.dash_array = parse_stroke_dasharray(stroke_dasharray)
                # line.dash_offset = parse_stroke_dashoffset(stroke_dashoffset)

        # 保存演示文稿
        if save_name is not None:
            prs.save(save_name)
            print(f"已生成并保存为 '{save_name}'")

        return prs, slide

    def plot(self, points):
        # points: 图像坐标系

        if self.normalize:
            points = points * np.array([1, -1]) * max(self.w, self.h) + np.array(
                [self.cx, self.cy]
            )
        else:
            points = (points - np.array([self.cx, self.cy])) * np.array(
                [1, -1]
            ) + np.array([self.cx, self.cy])

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 8))  # 设置图形大小
        plt.scatter(
            points[:, 0], points[:, 1], color='blue', marker='o', label='Data Points'
        )  # 绘制散点

        plt.title('Scatter Plot of Points')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')

        plt.savefig(
            f'plot-{self._counter:06d}.png', dpi=300, bbox_inches='tight'
        )  # 保存为 PNG 文件

        self._counter += 1
