import numpy as np

from svgpathtools import parse_path

from .util import (
    path_to_svg,
    get_path_d,
    get_path_transform,
    is_path_closed,
    count_edges_and_curves,
)
from .util import parse_transform, apply_transform_to_path


def is_point_in_box(points, boxes):
    n_points = points.shape[0]
    m_boxes = boxes.shape[0]

    points_expanded = points[:, np.newaxis, :]
    boxes_expanded = boxes[np.newaxis, :, :]

    in_box = (
        (points_expanded[:, :, 0] >= boxes_expanded[:, :, 0])
        & (points_expanded[:, :, 0] <= boxes_expanded[:, :, 2])
        & (points_expanded[:, :, 1] >= boxes_expanded[:, :, 1])
        & (points_expanded[:, :, 1] <= boxes_expanded[:, :, 3])
    )

    return in_box


def parse_arrows(paths, bbox, num_points=200):
    # 三种箭头：
    # 1. 一条path，里面包含了arrow和bar,这是闭合路径 num_edges >= 5
    # 2. 一条path，表示arrow shape 这个匹配比较困难
    # 2. 两条path，包含了三角形和line,非闭合路径
    # 3. 三条path，包含bar, 菱形，菱形

    lines = []
    arrows = []
    others = []

    # Determine whether it is an arrow or a line
    for svg_path in paths:
        is_arrow = False
        is_line = False
        # arrow, open_arrow, stealth_arrow, diamond_arrow, oval_arrow
        arrow_type = ''
        try:
            d = get_path_d(svg_path)
            x1, x2, y1, y2 = parse_path(d).bbox()
            w, h = (x2 - x1), (y2 - y1)
            area = (x2 - x1) * (y2 - y1)
        except Exception as e:
            print(f'Error processing path: {str(e)}')
            continue
        bx1, by1, bx2, by2 = bbox
        total_area = (bx2 - bx1) * (by2 - by1)
        ratio = area / total_area
        ar = min(w, h) / max(w, h)

        counts = count_edges_and_curves(d)
        num_edges = counts['num_edges']
        num_curves = counts['num_curves']

        # if not closed, num_edges=1 -> line, num_edges=2 -> w,h
        if not is_path_closed(d):
            if num_edges == 1 and num_curves == 0:
                is_arrow = False
                is_line = True
            elif num_edges == 2 and num_curves == 0:
                if ar >= 0.1:
                    is_arrow = True
                    is_line = False
                    arrow_type = ''
                else:
                    is_arrow = False
                    is_line = True

            else:
                print(path_to_svg(svg_path))
                continue
        else:
            if num_curves > 0 and ratio < 0.001:
                is_arrow = True
                is_line = False

        if is_arrow and (not is_line):
            arrows.append(svg_path)
        elif is_line and (not is_arrow):
            lines.append(svg_path)
        else:
            others.append(svg_path)

    print('Lines:', len(lines))
    print('Arrows:', len(arrows))
    print('Others:', len(others))

    if arrows == [] or lines == []:
        return arrows, lines, others, []

    # grouping: 1. bbox 2. sampling
    arrow_bboxes = []
    for svg_path in arrows:
        T_str = get_path_transform(svg_path)
        transform_matrix = parse_transform(T_str)

        d = get_path_d(svg_path)
        path = parse_path(d)
        transformed_path = apply_transform_to_path(path, transform_matrix)

        x1, x2, y1, y2 = transformed_path.bbox()
        bbox = np.array([x1, y1, x2, y2])
        arrow_bboxes.append(bbox)
    arrow_bboxes = np.stack(arrow_bboxes, axis=0)

    line_ids = []
    flattened_segments = []
    for idx, svg_path in enumerate(lines):
        T_str = get_path_transform(svg_path)
        transform_matrix = parse_transform(T_str)

        d = get_path_d(svg_path)
        path = parse_path(d)
        transformed_path = apply_transform_to_path(path, transform_matrix)

        for segment in transformed_path:
            flattened_segments.append(segment)
            line_ids.append(idx)

    s_points = np.array([(s.start.real, s.start.imag) for s in flattened_segments])
    e_points = np.array([(s.end.real, s.end.imag) for s in flattened_segments])
    s_inside = is_point_in_box(s_points, arrow_bboxes)
    e_inside = is_point_in_box(e_points, arrow_bboxes)
    pairs = get_arrow_line_pairs(
        s_inside, e_inside, line_ids, arrows, lines, strict=True
    )
    for arrow_idx, line_idx in pairs:
        print(f'Arrow {arrow_idx} is connected to Line {line_idx}')

    print(s_inside)
    print('***' * 30)
    print(e_inside)

    return arrows, lines, others, pairs


def get_arrow_line_pairs(s_inside, e_inside, line_ids, arrows, lines, strict=False):
    pairs = []
    for j in range(s_inside.shape[1]):  # arrows
        for i in range(s_inside.shape[0]):  # lines
            if strict:
                # only match if both start and end points are inside the arrow box
                if e_inside[i, j]:
                    line_id = line_ids[i]
                    pairs.append((j, line_id))
            else:
                # match if either start or end point is inside the arrow box
                if s_inside[i, j] or e_inside[i, j]:
                    line_id = line_ids[i]
                    pairs.append((j, line_id))
    return list(set(pairs))
