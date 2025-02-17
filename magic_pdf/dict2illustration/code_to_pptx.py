import os.path as osp
import tempfile
from typing import Optional

import numpy as np
from PIL import Image
from pptx import Presentation, presentation
from pptx.dml.color import RGBColor
from pptx.slide import Slide
from pptx.util import Pt


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
        xfrm = txbox._element.spPr.get_or_add_xfrm()
        xfrm.set('rot', str(rot_from_angle(theta)))

        # write text
        text_frame = txbox.text_frame
        text_frame.clear()

        p = text_frame.paragraphs[0]
        run = p.add_run()
        run.text = text_code_info['text']

        # handle font, size, style and color
        font = run.font
        font.name = text_code_info['font']
        font.size = Pt(text_code_info['size'])
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


def code_to_pptx(
    text_info_list: Optional[list[dict]] = None,
    shape_info_list: Optional[list[dict]] = None,
    image_info_list: Optional[list[dict]] = None,
    save_path: Optional[str] = None,
) -> presentation.Presentation:
    """Convert all codes to one PPTX."""
    assert (
        text_info_list is not None
        or shape_info_list is not None
        or image_info_list is not None
    ), 'At least one of text_info_list, shape_info_list, image_info_list should be provided.'

    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    if text_info_list is not None:
        prs = text_code_to_pptx(text_info_list, prs=prs)

    if shape_info_list is not None:
        pass

    if image_info_list is not None:
        prs = image_code_to_pptx(image_info_list, prs=prs)

    if save_path is not None:
        prs.save(save_path)
    return prs
