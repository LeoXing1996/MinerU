import numpy as np
import re
import lxml.etree as le
from shapely.geometry import Polygon, LineString
from svgpath2mpl import parse_path


def parse_svg(svg_path):
    """解析 SVG 文件并返回 tree 和 root"""
    with open(svg_path, 'r') as f:
        tree = le.fromstring(f.read())
    return tree


def get_svg_bounds(svg_root):
    """获取 SVG 的 viewBox 或 width/height 作为画面边界"""
    viewBox = svg_root.get('viewBox')
    if viewBox:
        min_x, min_y, width, height = map(float, viewBox.split())
    else:
        width = float(svg_root.get('width', '0').replace('px', ''))
        height = float(svg_root.get('height', '0').replace('px', ''))
        min_x, min_y = 0, 0  # 默认 (0,0)

    return min_x, min_y, min_x + width, min_y + height  # (min_x, min_y, max_x, max_y)


def parse_transform(transform_str):
    """解析 SVG transform 字符串并返回 3x3 变换矩阵"""
    matrix = np.eye(3)  # 初始化单位矩阵

    if not transform_str:
        return matrix  # 无变换，返回单位矩阵

    # 匹配 transform 指令
    transform_regex = re.findall(r'(\w+)\(([^)]+)\)', transform_str)
    for transform_type, values in transform_regex:
        values = list(map(float, values.replace(',', ' ').split()))

        if transform_type == 'translate':
            tx, ty = values if len(values) > 1 else (values[0], 0)
            matrix = np.dot(np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]]), matrix)

        elif transform_type == 'scale':
            sx, sy = values if len(values) > 1 else (values[0], values[0])
            matrix = np.dot(np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]]), matrix)

        elif transform_type == 'rotate':
            angle = np.radians(values[0])
            cx, cy = values[1:] if len(values) > 1 else (0, 0)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotate_matrix = np.array(
                [
                    [cos_a, -sin_a, cx - cx * cos_a + cy * sin_a],
                    [sin_a, cos_a, cy - cx * sin_a - cy * cos_a],
                    [0, 0, 1],
                ]
            )
            matrix = np.dot(rotate_matrix, matrix)

        elif transform_type == 'skewX':
            angle = np.radians(values[0])
            matrix = np.dot(
                np.array([[1, np.tan(angle), 0], [0, 1, 0], [0, 0, 1]]), matrix
            )

        elif transform_type == 'skewY':
            angle = np.radians(values[0])
            matrix = np.dot(
                np.array([[1, 0, 0], [np.tan(angle), 1, 0], [0, 0, 1]]), matrix
            )

        elif transform_type == 'matrix':
            a, b, c, d, e, f = values
            matrix = np.dot(np.array([[a, c, e], [b, d, f], [0, 0, 1]]), matrix)

    return matrix


def apply_transform(vertices, transform_str):
    """对路径点应用 transform 变换"""
    matrix = parse_transform(transform_str)  # 获取变换矩阵
    vertices = np.column_stack(
        [vertices, np.ones(vertices.shape[0])]
    )  # 转换为齐次坐标 (x, y, 1)
    transformed_vertices = np.dot(vertices, matrix.T)[:, :2]  # 应用变换矩阵
    return transformed_vertices


def get_path_bounds(path_element):
    """计算 <path> 元素的变换后边界框 (x_min, y_min, x_max, y_max)"""
    d = path_element.get('d')
    transform = path_element.get('transform')  # 获取 transform 属性
    if not d:
        return None

    try:
        mpl_path = parse_path(d)  # 解析 SVG 路径
        transformed_vertices = apply_transform(
            mpl_path.vertices, transform
        )  # 应用 transform
        if len(transformed_vertices) < 3:  # 直线情况（少于 3 点）
            shape = LineString(transformed_vertices)
        else:
            shape = Polygon(transformed_vertices)  # 其他情况，假设是多边形

        return shape.bounds  # (min_x, min_y, max_x, max_y)
    except Exception as e:
        print(f'解析失败: {e}')
        return None


def is_path_out_of_bounds(path_element, svg_bounds):
    """判断 <path> 是否超出 SVG 画面边界"""
    path_bounds = get_path_bounds(path_element)
    if not path_bounds:
        return False  # 解析失败，不删除

    min_x, min_y, max_x, max_y = path_bounds
    svg_min_x, svg_min_y, svg_max_x, svg_max_y = svg_bounds

    is_out = (
        max_x < svg_min_x or min_x > svg_max_x or max_y < svg_min_y or min_y > svg_max_y
    )
    if is_out:
        print(f'Path bbox: {path_bounds}')
    return is_out


def remove_out_of_bounds_paths(svg_path, output_path):
    """删除超出边界的 <path> 并保存新的 SVG"""
    svg_root = parse_svg(svg_path)

    svg_bounds = get_svg_bounds(svg_root)
    namespace = '{' + svg_root.nsmap.get(None) + '}'

    removed_count = 0
    elements_to_remove = []

    # 遍历所有 <path> 并找到其父级
    for path in svg_root.findall('.//' + namespace + 'path'):
        parent = path.getparent()  # 获取父元素
        if parent is None:
            parent = svg_root  # 如果找不到父元素，默认 root
        if parent.tag.endswith('clipPath'):
            continue
        if is_path_out_of_bounds(path, svg_bounds):
            elements_to_remove.append((parent, path))
            removed_count += 1

    # 统一删除，避免遍历时修改 DOM
    for parent, path in elements_to_remove:
        parent.remove(path)

    # 保存新的 SVG 文件
    with open(output_path, 'w') as f:
        f.write(le.tostring(svg_root).decode())

    print(f'✅ 已删除 {removed_count} 个超出边界的 <path> 元素')
    print(f'💾 新的 SVG 文件已保存为 {output_path}')
