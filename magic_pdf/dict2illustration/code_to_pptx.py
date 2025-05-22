import os.path as osp
import tempfile
import os
from typing import Optional, List, Dict

import numpy as np
from PIL import Image
from pptx import Presentation, presentation
from pptx.dml.color import RGBColor
from pptx.slide import Slide
from pptx.util import Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

from .shape_optimizer import ShapeOptimizer
from .filter_shapes import remove_out_of_bounds_paths
from .code_library import shape_pool, generate_path_code, generate_svg_path_code


def rot_from_angle(value):
    """Return positive integer rotation in 60,000ths of a degree corresponding
    to *value* expressed as a float number of degrees."""
    DEGREE_INCREMENTS = 60000
    THREE_SIXTY = 360 * DEGREE_INCREMENTS
    # modulo normalizes negative and >360 degree values
    return int(round(value * DEGREE_INCREMENTS)) % THREE_SIXTY


def rotate_point(point, center, angle):
    """Rotate a point around a given center by a specific angle (in
    degrees)."""
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


def text_code_to_pptx(
    text_info_list: list[dict],
    prs: Optional[presentation.Presentation] = None,
    save_path: Optional[str] = None,
) -> presentation.Presentation:
    """v1: do not consider latex rendering."""

    if prs is None:
        prs = Presentation()
        slide: Slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    else:
        slide = prs.slides[0]  # only one slide

    for text_code_info in text_info_list:
        theta = text_code_info['rotate']
        bbox = text_code_info['bbox']

        if float(theta) != 0:
            # rotate bbox frist
            cent_x = (bbox[0] + bbox[2]) / 2
            cent_y = (bbox[1] + bbox[3]) / 2

            bbox = get_rotated_bbox(
                bbox,
                (cent_x, cent_y),
                -theta,
            )

        x_min, y_min, x_max, y_max = bbox
        cx = x_max - x_min
        cy = y_max - y_min

        txbox = slide.shapes.add_textbox(
            Pt(x_min),
            Pt(y_min),
            Pt(cx),
            Pt(cy),
        )
        txbox.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE

        xfrm = txbox._element.spPr.get_or_add_xfrm()
        xfrm.set('rot', str(rot_from_angle(theta)))

        # write text
        text_frame = txbox.text_frame
        text_frame.clear()

        p = text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER  # center alignment
        run = p.add_run()
        run.text = text_code_info['text']

        # handle font, size, style and color
        font = run.font
        font.name = text_code_info['font']
        #############################################
        # font.size = Pt(text_code_info['size'])
        font_size = max(1, text_code_info['size'])
        font.size = Pt(font_size)
        #############################################
        text_style = text_code_info['style']
        if text_style == 1:
            font.italic = True
        elif text_style == 2:
            pass  # serifed
        elif text_style == 3:
            pass  # monospaced
        elif text_style == 4:
            font.bold = True
        rgb = text_code_info['color']
        font.color.rgb = RGBColor(*rgb)

    if save_path is not None:
        prs.save(save_path)

    return prs


def image_code_to_pptx(
    image_info_list: list[dict],
    prs: Optional[presentation.Presentation] = None,
) -> presentation.Presentation:
    if prs is None:
        prs = Presentation()
        slide: Slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    else:
        slide = prs.slides[0]  # only one slide

    for image in image_info_list:
        img = image['image']
        rot = image['rotate']

        # handle rotation
        if isinstance(img, Image.Image):
            processed_img = img.convert('RGBA').rotate(-rot, expand=True)
        elif isinstance(img, str):
            img = os.path.abspath(img)
            assert osp.exists(img), f'Image file {img} does not exist.'
            processed_img = Image.open(img).convert('RGBA').rotate(-rot, expand=True)
        else:
            raise ValueError(f'Unsupported image type: {type(img)}.')

        # handle bbox
        bbox = image['bbox']
        x_min, y_min, x_max, y_max = bbox
        cx = x_max - x_min
        cy = y_max - y_min

        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            processed_img.save(f.name)
            slide.shapes.add_picture(
                f.name,
                Pt(x_min),
                Pt(y_min),
                Pt(cx),
                Pt(cy),
            )

    return prs


def shape_code_to_pptx(
    shape_info_list: list[dict],
    prs: Optional[presentation.Presentation] = None,
) -> presentation.Presentation:
    if prs is None:
        prs = Presentation()
        slide: Slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    else:
        slide = prs.slides[0]  # only one slide

    for shape in shape_info_list:
        name = shape['name']
        params = shape['params']
        attributes = shape['attributes']

        try:
            shape_class = [
                shape for shape in shape_pool if shape.__dict__['NAME'] == name
            ][0]
            shape = shape_class()
        except:
            print(f'Shape class {name} not found.')
            continue
        # 提取参数
        cx = params['cx']
        cy = params['cy']
        w = params['w']
        h = params['h']
        rotation = params.get('rotation', 0)
        flip_h = params.get('flip_h', False)
        flip_v = params.get('flip_v', False)
        adjustments = params.get('adjustments', [])

        # 计算外接矩形的左上和右下点坐标
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

        # 设置形状参数
        shape.set_draw_params(
            cx=cx,
            cy=cy,
            w=w,
            h=h,
            rotation=rotation,
            flip_h=flip_h,
            flip_v=flip_v,
            normalize=False,
        )

        # 设置形状属性
        if attributes:
            shape.set_attributes(attributes)

        # 设置可调整参数
        shape.set_adjustable_params()
        if adjustments:
            shape.set_params(adjustments)

        # 绘制形状
        try:
            prs, slide = shape.draw(prs, slide)
        except:
            print(f'Shape {name} draw failed.')
            continue

    return prs


def svg_path_code_to_pptx(
    path_data_list: List[Dict],
    save_path: Optional[str] = None,
    slide_width: int = 500,
    slide_height: int = 300,
    sample_points: int = 100,
) -> bool:
    """
    将多个SVG路径数据转换为PPT文件

    Args:
        path_data_list (List[Dict]): SVG路径数据列表，每个字典包含：
            - path_data: SVG路径数据（d属性值）
            - fill_color: 填充颜色（默认"#000000"）
        save_path (str, optional): PPT保存路径
        slide_width (int): 幻灯片宽度，默认500
        slide_height (int): 幻灯片高度，默认300
        sample_points (int): 路径采样点数，默认100

    Returns:
        bool: 是否成功生成PPT

    Example:
        >>> svg_path_code_to_pptx(
        ...     path_data_list=[
        ...         {
        ...             "path_data": "M10,10 L90,10 L90,90 L10,90 Z",
        ...             "fill_color": "#FF0000"
        ...         },
        ...         {
        ...             "path_data": "M110,10 A40,40 0 1,1 190,10 A40,40 0 1,1 110,10 Z",
        ...             "fill_color": "#0000FF"
        ...         }
        ...     ],
        ...     save_path="output.pptx"
        ... )
    """
    try:
        # 创建新的PPT和幻灯片
        from pptx import Presentation

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[6])  # 空白布局

        # 处理每个SVG路径
        success = True
        for path_item in path_data_list:
            path_data = path_item['path_data']
            fill_color = path_item.get('fill_color', '#000000')

            # 使用generate_svg_path_code函数处理单个路径
            prs, slide, path_success = generate_svg_path_code(
                path_data=path_data,
                fill_color=fill_color,
                prs=prs,
                slide=slide,
                save_path=None,  # 不保存单个路径的结果
                sample_points=sample_points,
            )

            # 如果任何路径处理失败，将总体结果标记为失败
            success = success and path_success

        # 保存最终的PPT
        if save_path is not None:
            prs.save(save_path)
            print(f"已生成并保存为 '{save_path}'")

        return success

    except Exception as e:
        print(f'生成SVG路径PPT时发生错误: {str(e)}')
        return False


def code_to_pptx(
    text_info_list: Optional[list[dict]] = None,
    shape_info_list: Optional[list[dict]] = None,
    image_info_list: Optional[list[dict]] = None,
    svg_path_file: Optional[str] = None,
    save_path: Optional[str] = None,
) -> presentation.Presentation:
    # records_dir = save_path.split('/')[0] + '/' + save_path.split('/')[1]
    records_dir = os.path.join('/'.join(save_path.split('/')[:-2]))
    """Convert all codes to one PPTX."""
    assert (
        text_info_list is not None
        or shape_info_list is not None
        or image_info_list is not None
    ), 'At least one of text_info_list, shape_info_list, image_info_list should be provided.'

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    if svg_path_file is not None:
        prs, slide, _ = generate_path_code(svg_path_file, prs=prs, slide=slide)

    if shape_info_list is not None:
        if isinstance(shape_info_list, list):
            prs = shape_code_to_pptx(shape_info_list, prs=prs)

        elif shape_info_list.endswith('.svg'):
            remove_out_of_bounds_paths(shape_info_list, shape_info_list)
            optimizer = ShapeOptimizer()
            prs, slide = optimizer.process_file(
                shape_info_list, prs=prs, slide=slide, records_dir=records_dir
            )

    if text_info_list is not None:
        prs = text_code_to_pptx(text_info_list, prs=prs)

    if image_info_list is not None:
        prs = image_code_to_pptx(image_info_list, prs=prs)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        prs.save(save_path)
    return prs
