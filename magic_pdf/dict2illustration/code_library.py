# from .code_to_pptx import text_code_to_pptx, image_code_to_pptx
from typing import Optional
from .shapes import *
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Pt
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
import re
from svgpathtools import svg2paths2


shape_pool = [
    OvalShape,
    IsoscelesTriangleShape,
    RightTriangleShape,
    RectangleShape,
    RoundedRectangleShape,
    SnipSingleCornerRectangleShape,
    SnipSameSideCornerRectangleShape,
    SnipDiagonalCornerRectangleShape,
    SnipAndRoundSingleCornerRectangleShape,
    ArrowBar,
    LineConnector,
    ParallelogramShape,
    TrapezoidShape,
]
WIDTH = 500
HEIGHT = 300


def generate_path_code(
    path: str,
    prs=None,
    slide=None,
    save_path: Optional[str] = None,
    sample_points: int = 100,
) -> tuple:
    """为LLM提供的函数调用工具，用于生成包含路径元素的PPT

    Args:
        path (str): SVG文件路径或SVG路径字符串
        prs: PowerPoint演示文稿对象，如果为None则使用全局对象
        slide: PowerPoint幻灯片对象，如果为None则使用全局对象
        save_path (str, optional): PPT保存路径
        sample_points (int, optional): 路径采样点数量，默认100

    Returns:
        tuple: (prs, slide, success)，其中success表示是否成功导入路径

    Example:
        >>> prs, slide, _ = generate_path_code(path="path/to/svg/file.svg", sample_points=100)
    """
    try:
        # 如果没有提供prs和slide，使用全局对象
        global _prs, _slide
        if prs is None:
            prs = _prs
        if slide is None:
            slide = _slide

        # 如果全局对象也不存在，创建新的
        if prs is None or slide is None:
            prs, slide = init_presentation()

        def hex_to_rgb(hex_color):
            hex_color = hex_color.strip()
            if hex_color.startswith('#'):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            elif hex_color.startswith('rgb'):
                nums = map(int, re.findall(r'\d+', hex_color))
                return tuple(nums)
            else:
                # fallback to black
                return (0, 0, 0)

        # 读取SVG文件
        paths, attributes, svg_attributes = svg2paths2(path)

        # 处理每个路径
        for path, attr in zip(paths, attributes):
            builder = slide.shapes.build_freeform()

            # 设置起始点
            start_point = path[0].start
            start_x, start_y = start_point.real, start_point.imag
            start_x, start_y = Pt(start_x), Pt(start_y)
            builder.move_to(start_x, start_y)

            # 采样路径点
            following_points = []
            for segment in path:
                points = [segment.point(t) for t in np.linspace(0, 1, sample_points)]
                for p in points:
                    x, y = p.real, p.imag
                    x, y = Pt(x), Pt(y)
                    following_points.append((x, y))

            # 添加路径点并转换为形状
            builder.add_line_segments(following_points, close=True)
            shape = builder.convert_to_shape()

            # 设置填充
            shape.fill.solid()
            fill_color = attr.get('fill', '#000000')
            if fill_color != 'none':
                fill_color = hex_to_rgb(fill_color)
                shape.fill.fore_color.rgb = RGBColor(*fill_color)

            # 禁用阴影
            shape.shadow.inherit = False

            # 设置边框
            shape.line.color.rgb = RGBColor(*fill_color)
            shape.line.width = Pt(0)
            shape.line.fill.background()

        # 保存PPT
        if save_path is not None:
            prs.save(save_path)
            print(f"已生成并保存为 '{save_path}'")

        return prs, slide, True

    except Exception as e:
        print(f'生成SVG路径时发生错误: {str(e)}')
        return prs, slide, False


def rot_from_angle(value):
    """Return positive integer rotation in 60,000ths of a degree corresponding
    to *value* expressed as a float number of degrees.
    """
    DEGREE_INCREMENTS = 60000
    THREE_SIXTY = 360 * DEGREE_INCREMENTS
    # modulo normalizes negative and >360 degree values
    return int(round(value * DEGREE_INCREMENTS)) % THREE_SIXTY


def rotate_point(point, center, angle):
    """Rotate a point around a given center by a specific angle (in
    degrees).
    """
    angle_rad = np.radians(angle)  # Convert angle to radians
    px, py = point
    cx, cy = center

    # Translate point to origin
    translated_x = px - cx
    translated_y = py - cy

    # Perform rotation
    rotated_x = translated_x * np.cos(angle_rad) - translated_y * np.sin(angle_rad)
    rotated_y = translated_x * np.sin(angle_rad) + translated_y * np.cos(angle_rad)

    # Translate back to original position
    new_x = rotated_x + cx
    new_y = rotated_y + cy

    return new_x, new_y


def get_rotated_bbox(bbox, center, theta):
    """Get the bounding box of a rotated rectangle.

    Args:
    - bbox: Tuple (x_min, y_min, x_max, y_max) representing the original bounding box.
    - center: Tuple (cx, cy) representing the rotation center.
    - theta: Rotation angle in degrees.

    Returns:
    - Tuple (x_min, y_min, x_max, y_max) representing the rotated bounding box.
    """
    x_min, y_min, x_max, y_max = bbox

    # Define the four corner points of the bounding box
    corners = [
        (x_min, y_min),  # Bottom-left
        (x_min, y_max),  # Top-left
        (x_max, y_min),  # Bottom-right
        (x_max, y_max),  # Top-right
    ]

    # Rotate each corner
    rotated_corners = [rotate_point(corner, center, theta) for corner in corners]

    # Get the new bounding box
    rotated_x = [p[0] for p in rotated_corners]
    rotated_y = [p[1] for p in rotated_corners]

    new_x_min = min(rotated_x)
    new_y_min = min(rotated_y)
    new_x_max = max(rotated_x)
    new_y_max = max(rotated_y)

    return new_x_min, new_y_min, new_x_max, new_y_max


def generate_text_code(
    bbox: list[float],
    text: str,
    font: str,
    size: float,
    style: int,
    color: list[int],
    rotate: float,
    prs: Optional[Presentation] = None,
    slide: Optional[object] = None,
    save_path: Optional[str] = None,
) -> Presentation:
    # 使用全局对象
    global _prs, _slide
    if prs is None:
        prs = _prs
    if slide is None:
        slide = _slide

    # 如果全局对象也不存在，创建新的
    if prs is None:
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _prs, _slide = prs, slide
    elif slide is None:
        slide = prs.slides[0]
        _slide = slide

    if float(rotate) != 0:
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        bbox = get_rotated_bbox(bbox, (cx, cy), -rotate)

    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min

    txbox = slide.shapes.add_textbox(Pt(x_min), Pt(y_min), Pt(width), Pt(height))
    txbox.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    xfrm = txbox._element.spPr.get_or_add_xfrm()
    xfrm.set('rot', str(rot_from_angle(rotate)))

    p = txbox.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text

    font_obj = run.font
    font_obj.name = font
    font_obj.size = Pt(size)
    if style == 1:
        font_obj.italic = True
    elif style == 4:
        font_obj.bold = True

    font_obj.color.rgb = RGBColor(*color)

    if save_path:
        prs.save(save_path)

    return prs


def generate_image_code(
    bbox: list[float],
    rotate: float = 0.0,
    placeholder_text: str = 'Image Placeholder',
    prs: Optional[Presentation] = None,
    slide: Optional[object] = None,
    save_path: Optional[str] = None,
) -> tuple[Presentation, object, bool]:
    # 使用全局对象
    global _prs, _slide
    if prs is None:
        prs = _prs
    if slide is None:
        slide = _slide

    # 如果全局对象也不存在，创建新的
    if prs is None:
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _prs, _slide = prs, slide
    elif slide is None:
        slide = prs.slides[0]
        _slide = slide

    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min

    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Pt(x_min), Pt(y_min), Pt(width), Pt(height)
    )

    # 设置文本框属性
    placeholder_text = '<IMG>'
    text_frame = shape.text_frame
    text_frame.clear()
    text_frame.word_wrap = True
    text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

    p = text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = placeholder_text
    run.font.color.rgb = RGBColor(0, 0, 0)

    # 根据边界框大小自适应字体大小
    # 基本思路：根据宽度和高度的较小值来计算合适的字体大小
    # 这里使用一个简单的启发式方法，可以根据需要调整系数
    font_size = min(width, height) / 10  # 字体大小为边界框较小边的1/10
    font_size = max(6, min(font_size, 36))  # 限制字体大小在6-36pt之间
    run.font.size = Pt(font_size)

    shape.shadow.inherit = False
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 255, 255)
    line = shape.line
    line.color.rgb = RGBColor(0, 0, 0)
    line.width = Pt(1)

    xfrm = shape._element.spPr.get_or_add_xfrm()
    xfrm.set('rot', str(rot_from_angle(rotate)))

    if save_path:
        prs.save(save_path)

    return prs, slide, True


def generate_shape_code(
    name: str,
    params: dict,
    attributes: Optional[dict] = None,
    prs: Optional[Presentation] = None,
    slide: Optional[object] = None,
    save_path: Optional[str] = None,
) -> tuple[Presentation, object, bool]:
    # 使用全局对象
    global _prs, _slide
    if prs is None:
        prs = _prs
    if slide is None:
        slide = _slide

    # 如果全局对象也不存在，创建新的
    if prs is None:
        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        _prs, _slide = prs, slide
    elif slide is None:
        slide = prs.slides[0]
        _slide = slide

    try:
        shape_class = [cls for cls in shape_pool if cls.__dict__['NAME'] == name][0]
        shape = shape_class()
    except:
        print(f'Shape class {name} not found.')
        return prs, slide, False

    shape.set_draw_params(
        cx=params['cx'],
        cy=params['cy'],
        w=params['w'],
        h=params['h'],
        rotation=params.get('rotation', 0),
        flip_h=params.get('flip_h', False),
        flip_v=params.get('flip_v', False),
        normalize=False,
    )

    if attributes:
        shape.set_attributes(attributes)

    shape.set_adjustable_params()
    if 'adjustments' in params:
        shape.set_params(params['adjustments'])

    try:
        prs, slide = shape.draw(prs, slide)
        if save_path:
            prs.save(save_path)
        return prs, slide, True
    except:
        print(f'Shape {name} draw failed.')
        return prs, slide, False


def generate_svg_path_code(
    path_data: str,
    fill_color: str = '#000000',
    prs=None,
    slide=None,
    save_path: Optional[str] = None,
    sample_points: int = 100,
) -> tuple:
    """为LLM提供的函数调用工具，用于处理SVG路径数据并在PPT中生成图形

    Args:
        path_data (str): SVG路径数据（d属性的值）
        fill_color (str): 填充颜色（十六进制或rgb格式）
        prs: PowerPoint演示文稿对象，如果为None则使用全局对象
        slide: PowerPoint幻灯片对象，如果为None则使用全局对象
        save_path (str, optional): PPT保存路径
        sample_points (int, optional): 路径采样点数量，默认100

    Returns:
        tuple: (prs, slide, success)，其中success表示是否成功导入路径
    """
    try:
        # 使用全局对象
        global _prs, _slide
        if prs is None:
            prs = _prs
        if slide is None:
            slide = _slide

        # 如果全局对象也不存在，创建新的
        if prs is None:
            prs = Presentation()
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            _prs, _slide = prs, slide
        elif slide is None:
            slide = prs.slides[0]
            _slide = slide

        def hex_to_rgb(hex_color):
            hex_color = hex_color.strip()
            if hex_color.startswith('#'):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            elif hex_color.startswith('rgb'):
                nums = map(int, re.findall(r'\d+', hex_color))
                return tuple(nums)
            else:
                # fallback to black
                return (0, 0, 0)

        # 创建临时SVG文件
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as temp_file:
            temp_path = temp_file.name
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"><defs/><path fill="{fill_color}" d="{path_data}"/></svg>'
            temp_file.write(svg_content.encode('utf-8'))

        # 使用临时文件调用generate_path_code
        try:
            result = generate_path_code(
                path=temp_path,
                prs=prs,
                slide=slide,
                save_path=save_path,
                sample_points=sample_points,
            )

            # 删除临时文件
            import os

            os.unlink(temp_path)

            return result
        except Exception as e:
            # 确保删除临时文件
            import os

            os.unlink(temp_path)
            raise e

    except Exception as e:
        print(f'处理SVG路径数据时发生错误: {str(e)}')
        return prs, slide, False
