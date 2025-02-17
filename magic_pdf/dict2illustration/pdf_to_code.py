import math
from typing import Optional

import fitz
import numpy as np
from fitz import sRGB_to_rgb
from PIL import Image
from pymupdf import Document


def parse_text_block(block: dict) -> list[dict]:
    """v1: do not support latex parsing, all formulars are treated as text with up/down-scripts."""

    text_code_info_list = []
    for line in block['lines']:
        cos, sin = line['dir']
        spans = line['spans']

        # handle rotate
        theta = np.arcsin(sin) / np.pi * 180  # degree

        for span in spans:
            bbox = span['bbox']

            text_style = span['flags']
            rgb = sRGB_to_rgb(span['color'])

            text_code_info = {
                'bbox': bbox,
                'text': span['text'],
                'font': span['font'],
                'size': span['size'],
                'style': text_style,
                'color': rgb,
                'rotate': theta,
            }
            text_code_info_list.append(text_code_info)

    return text_code_info_list


def text_to_code(pdf: Optional[Document]) -> Optional[list[dict]]:
    """Extract the text from cropped illustration part and convert it to
    trainable format."""
    # if pdf is None:
    #     return None

    assert pdf.page_count == 1, 'Only support one page pdf.'

    text_page = pdf[0].get_textpage()
    text_blocks = text_page.extractDICT()['blocks']
    text_info_list = []
    for block in text_blocks:
        text_info_list += parse_text_block(block)

    return text_info_list


def parse_shape_block():
    pass


def shape_to_code(pdf: Optional[Document]) -> Optional[list[dict]]:
    import lxml.etree as le

    shapes_text_and_image: str = pdf[0].get_svg_image(text_as_path=False)
    shapes_text_and_image_svg = le.fromstring(shapes_text_and_image)
    namespace = '{' + shapes_text_and_image_svg.nsmap.get(None) + '}'
    for text_element in shapes_text_and_image_svg.findall('.//' + namespace + 'text'):
        parent = text_element.getparent()
        parent.remove(text_element)
    for image_element in shapes_text_and_image_svg.findall('.//' + namespace + 'image'):
        parent = image_element.getparent()
        parent.remove(image_element)

    # TODO: support high-level code
    return {'svg': le.tostring(shapes_text_and_image_svg).decode()}


NO_FLIP = 0
HORIZONTAL_FLIP = 1
VERTICAL_FLIP = 2
FLIP_UNKNOWN = 3


def parse_rot_and_flip_from_trans(transform: list[float]) -> int:
    """Parse rotation and flip from the transform matrix."""
    a, b, c, d, e, f = transform
    rotation_scale_matrix = np.array([[a, b], [c, d]])
    determinant = np.linalg.det(rotation_scale_matrix)
    has_flip = determinant < 0

    if not has_flip:
        return NO_FLIP
        # return translation, approx_rotation_angle, "no_flip"

    # 特征值和特征向量分析
    eigenvalues, eigenvectors = np.linalg.eig(rotation_scale_matrix)
    for i in range(len(eigenvalues)):
        if eigenvalues[i].real < 0:  # 查找实部为负的特征值
            negative_eigenvalue_eigenvector = eigenvectors[
                :, i
            ].real  # 取实部作为特征向量
            break  # 找到第一个负实部特征值就停止，假设通常只有一个

    if negative_eigenvalue_eigenvector is not None:
        v_flip_axis_normalized = negative_eigenvalue_eigenvector / np.linalg.norm(
            negative_eigenvalue_eigenvector
        )  # 归一化

        # 计算与水平和竖直轴的夹角 (使用点积计算夹角余弦，再用 arccos 求角度)
        horizontal_axis = np.array([1.0, 0.0])
        vertical_axis = np.array([0.0, 1.0])

        cos_angle_horizontal = np.dot(v_flip_axis_normalized, horizontal_axis)
        cos_angle_vertical = np.dot(v_flip_axis_normalized, vertical_axis)

        angle_horizontal = math.degrees(
            math.acos(np.clip(abs(cos_angle_horizontal), -1.0, 1.0))
        )  # 弧度转角度，并取绝对值和clip防止超出acos定义域
        angle_vertical = math.degrees(
            math.acos(np.clip(abs(cos_angle_vertical), -1.0, 1.0))
        )

        angle_threshold = 45.0  # 角度阈值，可以根据需要调整

        if angle_vertical < angle_threshold:
            return HORIZONTAL_FLIP
        elif angle_horizontal < angle_threshold:
            return VERTICAL_FLIP
        else:
            return FLIP_UNKNOWN
    else:
        return FLIP_UNKNOWN


def parse_image_block(images: list[dict], doc: Document) -> list[dict]:
    """
    v1: do not summarize image to text, just return the image itself and corresponding bbox.

    **IMPORTANT**: How we handle flip and rotation
        * Flip: we save the flip the original image according to the transform matrix, and
            the flip the image inplace.
        * Rotation: we calculate the rotation degree according to the transform matrix, and
            save the rotation degree as trainable code.
        * Code to PPT: we rotate the image first, and paste it to the slide according to the bbox.
    """

    image_code_info_list = []

    for image in images:
        xref = image['xref']
        if xref == 0:
            continue

        bbox = image['bbox']
        transform = image['transform']
        a, b, c, d, e, f = transform
        rotation_scale = np.array([[a, b], [c, d]])

        pix = fitz.Pixmap(doc, xref)

        img_pil = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)

        # NOTE: only consider flip and rotation, and ignore scale
        flip_status = parse_rot_and_flip_from_trans(transform)
        if flip_status == HORIZONTAL_FLIP:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            rotation_scale = np.array([[-1, 0], [0, 1]]) @ rotation_scale

        if flip_status == VERTICAL_FLIP:
            img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
            rotation_scale = np.array([[1, 0], [0, -1]]) @ rotation_scale

        angle = np.degrees(np.arctan2(rotation_scale[0, 1], rotation_scale[0, 0]))
        # print("flip_status:", flip_status)

        image_code_info = {'image': img_pil, 'bbox': bbox, 'rotate': angle}
        image_code_info_list.append(image_code_info)

    return image_code_info_list


def image_to_code(pdf: Document) -> list[dict]:
    """Extract the image from cropped illustration part and convert it to
    trainable format."""
    assert pdf.page_count == 1, 'Only support one page pdf.'
    images = pdf[0].get_image_info(xrefs=True)
    image_info_list = parse_image_block(images, pdf)
    return image_info_list
