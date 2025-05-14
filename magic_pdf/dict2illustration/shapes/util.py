import re
import numpy as np
from svgpathtools import parse_path
from svgpathtools import Path, Line, CubicBezier, QuadraticBezier, Arc
from svgpathtools.path import transform
from lxml import etree


def sample_svg_path(svg_str, num_points=200, normalize=False):
    """
    利用 svgpathtools 对 SVG path 字符串进行解析，并在轮廓上等距离采样离散点。
    """
    path = parse_path(svg_str)
    ts = np.linspace(0, 1, num_points, endpoint=True)
    points = []
    for t in ts:
        pt = path.point(t)
        points.append([pt.real, pt.imag])
    sampled_points = np.array(points)
    if normalize:
        cx, cy, w, h = get_bounding_rectangle(sampled_points)
        sampled_points = (sampled_points - np.array([cx, cy])) / max(w, h)
    return sampled_points


def is_path_closed(svg_str):
    path = parse_path(svg_str)
    if len(path) == 0:
        return False
    start_point = path[0].start
    end_point = path[-1].end
    return start_point == end_point


def get_bounding_rectangle(target_points):
    min_x, min_y = target_points.min(axis=0)
    max_x, max_y = target_points.max(axis=0)
    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    w = max_x - min_x
    h = max_y - min_y
    return cx, cy, w, h


def get_bounding_line(target_points):
    x1, y1 = target_points[0]
    x2, y2 = target_points[-1]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return cx, cy, w, h


def get_overall_bounding_rectangle(paths):
    min_x = 1e9
    min_y = 1e9
    max_x = 0
    max_y = 0

    for svg_path in paths:
        d = get_path_d(svg_path)
        path = parse_path(d)
        try:
            bbox = path.bbox()
            if bbox:  # 确保bbox不为空
                x1, x2, y1, y2 = bbox
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)
        except (ValueError, TypeError):
            continue  # 跳过无效的路径

    return (min_x, min_y, max_x, max_y)


def sample_segment(p0, p1, num_points, include_endpoint=True):
    """
    对线段 p0 -> p1 进行均匀采样，返回 (num_points x 2) 数组
    """
    return np.linspace(p0, p1, num_points, endpoint=include_endpoint)


def sample_arc(a0, a1, num_points, center, rx, ry, include_endpoint=True):
    angles = np.linspace(a0, a1, num_points, endpoint=include_endpoint)
    return np.column_stack(
        (center[0] + rx * np.cos(angles), center[1] + ry * np.sin(angles))
    )


def allocate_segment_points(
    segments, num_points, min_num_edge_points=3, allocation='len_imp'
):
    seg_lengths = [s.length for s in segments]
    total_length = sum(seg_lengths)

    seg_weights = [s.weight for s in segments]
    total_weight = sum(seg_weights)

    if total_length == 0.0:
        allocation = 'imp'
    elif total_weight == 0.0:
        allocation = 'len'

    if allocation == 'len':
        weight = np.array(seg_lengths) / total_length
    elif allocation == 'imp':
        weight = np.array(seg_weights) / total_weight
    elif allocation == 'len_imp':
        len_w = np.array(seg_lengths) / total_length
        imp_w = np.array(seg_weights) / total_weight
        weight = (len_w + imp_w) / sum(len_w + imp_w)
    else:
        raise TypeError(f"""
            Unsupported allocation type {allocation}.
            Supported types {'len', 'imp', 'len_imp'}
                        """)

    segment_counts = []
    allocated = 0
    for i in range(len(seg_lengths)):
        n = max(min_num_edge_points, int(round(weight[i] * num_points)))
        segment_counts.append(n)
        allocated += n

    discrepancy = allocated - num_points
    if discrepancy > 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy
    elif discrepancy < 0:
        idx = np.argmin(segment_counts)
        segment_counts[idx] += discrepancy

    return segment_counts


def parse_transform(transform_str):
    """
    | a  c  e |
    | b  d  f |
    | 0  0  1 |

    | sx ry tx|
    | ry sy ty|
    |  0  0  1|
    """
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    transform_regex = r'(\w+)\(\s*([-\d.]+(?:\s*,\s*[-\d.]+)*)?\s*\)'

    matches = re.findall(transform_regex, transform_str)
    for transform_type, params in matches:
        if transform_type == 'matrix':
            a, b, c, d, e, f = map(float, params.split(','))
            matrix = np.dot(matrix, np.array([[a, c, e], [b, d, f], [0, 0, 1]]))
        elif transform_type == 'translate':
            tx, ty = map(float, params.split(',')) if params else (0, 0)
            translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            matrix = np.dot(matrix, translation_matrix)
        elif transform_type == 'scale':
            sx, sy = map(float, params.split(',')) if params else (1, 1)
            scaling_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            matrix = np.dot(matrix, scaling_matrix)
        elif transform_type == 'rotate':
            angle, cx, cy = (map(float, params.split(',')) + [None, None])[:3]
            angle_rad = np.radians(angle)
            rotation_matrix = np.array(
                [
                    [np.cos(angle_rad), -np.sin(angle_rad), 0],
                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                    [0, 0, 1],
                ]
            )
            if cx is not None and cy is not None:
                translate_to_center = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
                translate_back = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
                matrix = np.dot(matrix, translate_back)
                matrix = np.dot(matrix, rotation_matrix)
                matrix = np.dot(matrix, translate_to_center)
            else:
                matrix = np.dot(matrix, rotation_matrix)
        elif transform_type == 'skewX':
            angle = float(params)
            skewX_matrix = np.array(
                [[1, np.tan(np.radians(angle)), 0], [0, 1, 0], [0, 0, 1]]
            )
            matrix = np.dot(matrix, skewX_matrix)
        elif transform_type == 'skewY':
            angle = float(params)
            skewY_matrix = np.array(
                [[1, 0, 0], [np.tan(np.radians(angle)), 1, 0], [0, 0, 1]]
            )
            matrix = np.dot(matrix, skewY_matrix)

    return matrix


def apply_transform(target_points, matrix):
    target_points = np.concatenate(
        [target_points, np.ones([target_points.shape[0], 1])], axis=1
    )
    target_points = np.dot(matrix, target_points.T).T
    return target_points[:, :2]


def apply_transform_to_path(path, matrix):
    assert isinstance(path, Path)
    path = transform(path, matrix)
    return path


def get_svg_paths(svg_file):
    with open(svg_file, 'r') as file:
        svg_content = file.read()

    svg_tree = etree.fromstring(svg_content)
    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    paths = svg_tree.xpath('//svg:path', namespaces=namespaces)
    paths = [
        path
        for path in paths
        if path.getparent().tag != f"{{{namespaces['svg']}}}clipPath"
    ]

    return paths


def get_path_attributes(path):
    attributes = path.attrib
    # for attr in ["xmlns", "xmlns:xlink", "xmlns:inkscape", "d", "transform"]:
    for attr in ['xmlns', 'xmlns:xlink', 'xmlns:inkscape', 'd']:
        if attr in attributes:
            del attributes[attr]
    return attributes


def get_path_d(path):
    return path.attrib.get('d', None)


def get_path_transform(path):
    return path.attrib.get('transform', None)


def path_to_svg(path):
    return etree.tostring(path, pretty_print=True, encoding='unicode')


def points_to_path(points, close_path=False):
    """
    将坐标点元组转换为SVG路径字符串

    Args:
        points (tuple): 坐标点元组，例如 ((x1,y1), (x2,y2), ...)
        close_path (bool): 是否添加闭合路径命令

    Returns:
        str: SVG路径字符串
    """
    if not points:
        return ''

    # 构建移动命令和线段命令
    path = f'M {points[0][0]},{points[0][1]}'
    for x, y in points[1:]:
        path += f' L {x},{y}'

    # 添加闭合路径命令
    if close_path:
        path += ' Z'

    return path


def count_edges_and_curves(svg_path, return_detail=False):
    path = parse_path(svg_path)

    line_segments = 0
    quadratic_curves = 0
    cubic_curves = 0
    arcs = 0

    for segment in path:
        if isinstance(segment, Line):
            line_segments += 1
        elif isinstance(segment, QuadraticBezier):
            quadratic_curves += 1
        elif isinstance(segment, CubicBezier):
            cubic_curves += 1
        elif isinstance(segment, Arc):
            arcs += 1

    if return_detail:
        return {
            'line_segments': line_segments,
            'quadratic_curves': quadratic_curves,
            'cubic_curves': cubic_curves,
            'arcs': arcs,
        }
    else:
        return {
            'num_edges': line_segments,
            'num_curves': quadratic_curves + cubic_curves + arcs,
        }


def calculate_ioa(A, B):
    area_A = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_B = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])

    inter_x1 = np.maximum(A[:, np.newaxis, 0], B[:, 0])
    inter_y1 = np.maximum(A[:, np.newaxis, 1], B[:, 1])
    inter_x2 = np.minimum(A[:, np.newaxis, 2], B[:, 2])
    inter_y2 = np.minimum(A[:, np.newaxis, 3], B[:, 3])

    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)

    inter_area = inter_width * inter_height
    iou_matrix = inter_area / (area_A[:, np.newaxis] + 1e-8)
    return iou_matrix


def judge_arrow_base(svg_path):
    edges = parse_path(svg_path)

    # 获取所有闭合子路径
    closed_subpaths = find_all_closed_subpaths(edges)

    for subpath in closed_subpaths:
        # 检查是否为矩形
        if is_rectangle(subpath):
            # 获取剩余部分的中心点
            other_subpath = get_other_subpath(edges, subpath)
            if not other_subpath:
                continue
            other_center = get_subpath_center(other_subpath)
            # 计算带方向的中线
            midline = get_rectangle_midline(subpath, direction_point=other_center)
            if midline:
                return midline

    return None


def find_all_closed_subpaths(edges):
    """查找所有可能的闭合子路径"""
    closed_subpaths = []
    n = len(edges)
    for i in range(n):
        for j in range(i + 1, n + 1):
            subpath = edges[i:j]
            if is_subpath_closed(subpath):
                closed_subpaths.append(subpath)
    return closed_subpaths


def is_subpath_closed(subpath):
    """判断子路径是否闭合"""
    if not subpath:
        return False
    start = subpath[0].start
    end = subpath[-1].end
    return np.isclose(start.real, end.real) and np.isclose(start.imag, end.imag)


def get_other_subpath(edges, subpath):
    """获取子路径外的剩余部分"""
    start_idx = edges.index(subpath[0])
    end_idx = edges.index(subpath[-1])
    return edges[:start_idx] + edges[end_idx + 1 :]


def is_rectangle(subpath, tol=1):
    """判断闭合子路径是否为矩形"""
    if len(subpath) != 4:
        return False

    # 获取四个顶点（A, B, C, D）
    points = [seg.start for seg in subpath] + [subpath[-1].end]
    A, B, C, D = points[:4]

    # 计算向量
    AB = (B.real - A.real, B.imag - A.imag)
    BC = (C.real - B.real, C.imag - B.imag)
    CD = (D.real - C.real, D.imag - C.imag)
    DA = (A.real - D.real, A.imag - D.imag)

    # 判断对边是否平行且等长
    def are_parallel(v1, v2):
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        return np.isclose(cross, 0, atol=tol)

    def same_length(v1, v2):
        len1 = np.hypot(v1[0], v1[1])
        len2 = np.hypot(v2[0], v2[1])
        return np.isclose(len1, len2, atol=tol)

    if not (are_parallel(AB, CD) and same_length(AB, CD)):
        return False
    if not (are_parallel(BC, DA) and same_length(BC, DA)):
        return False

    # 判断相邻边是否垂直
    dot_product = AB[0] * BC[0] + AB[1] * BC[1]
    if not np.isclose(dot_product, 0, atol=tol):
        return False

    return True


def get_rectangle_midline(subpath, direction_point):
    """计算指向指定方向的矩形中线"""
    # 提取顶点（保持原有逻辑）
    points = []
    seen = set()
    for seg in subpath:
        start = seg.start
        key = (start.real, start.imag)
        if key not in seen:
            points.append(start)
            seen.add(key)
    end = subpath[-1].end
    key = (end.real, end.imag)
    if key not in seen:
        points.append(end)
    points = points[:4]

    # 计算短边中线（保持原有逻辑）
    from itertools import pairwise

    edges = list(pairwise(points)) + [(points[-1], points[0])]
    lengths = [abs(p1 - p2) for p1, p2 in edges]
    min_len = min(lengths)
    short_edges = [
        (p1, p2)
        for (p1, p2), l in zip(edges, lengths)
        if np.isclose(l, min_len, atol=min_len * 0.1)
    ]

    if len(short_edges) != 2:
        return None

    # 计算中线端点
    (a1, a2), (b1, b2) = short_edges
    mid1 = ((a1.real + a2.real) / 2, (a1.imag + a2.imag) / 2)
    mid2 = ((b1.real + b2.real) / 2, (b2.imag + b2.imag) / 2)

    # 确定方向（选择离目标点更近的端点作为终点）
    if direction_point is not None:
        dist1 = np.hypot(mid1[0] - direction_point[0], mid1[1] - direction_point[1])
        dist2 = np.hypot(mid2[0] - direction_point[0], mid2[1] - direction_point[1])
        if dist1 < dist2:
            start, end = mid2, mid1  # 方向从mid2指向mid1
        else:
            start, end = mid1, mid2  # 方向从mid1指向mid2
    else:
        start, end = mid1, mid2

    return points_to_path([start, end])


def get_subpath_center(subpath):
    """计算子路径的中心点"""
    points = []
    for seg in subpath:
        points.append((seg.start.real, seg.start.imag))
    points.append((subpath[-1].end.real, subpath[-1].end.imag))
    points = np.array(points)
    return np.mean(points, axis=0)
