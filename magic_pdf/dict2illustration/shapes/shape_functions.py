import numpy as np


def sample_segment(p0, p1, num_points, include_endpoint=True):
    """
    对线段 p0 -> p1 进行均匀采样，返回 (num_points x 2) 的数组
    """
    return np.linspace(p0, p1, num_points, endpoint=include_endpoint)


def generate_cross_shape(params, num_edge_points=50):
    """
    Cross 模型函数（十字形）

    参数:
       params = [cx, cy, w, h, tx, ty, theta]
         - cx, cy : 全局平移，将局部十字形移动到目标位置（中心）
         - w, h   : 十字形的整体外扩尺寸（水平臂延展至 ±w/2，垂直臂延展至 ±h/2）
         - tx     : 垂直臂的宽度（对应十字形中心水平部分的宽度，要求 tx < w）
         - ty     : 水平臂的高度（对应十字形中心垂直部分的高度，要求 ty < h）
         - theta  : 整体旋转角（单位：弧度）
       num_edge_points: 每条边均匀采样的点数（默认50）

    返回:
       (N x 2) 的 numpy 数组，表示十字形边界的采样点
    """
    cx, cy, w, h, tx, ty, theta = params

    # 定义局部坐标下闭合十字形的关键点
    # 顺序按（逆时针或顺时针）排列，避免交叉
    # 这里采用以下顺序:
    A = np.array([-w / 2, -ty / 2])
    B = np.array([-tx / 2, -ty / 2])
    C = np.array([-tx / 2, -h / 2])
    D = np.array([tx / 2, -h / 2])
    E = np.array([tx / 2, -ty / 2])
    F = np.array([w / 2, -ty / 2])
    G = np.array([w / 2, ty / 2])
    H = np.array([tx / 2, ty / 2])
    I = np.array([tx / 2, h / 2])
    J = np.array([-tx / 2, h / 2])
    K = np.array([-tx / 2, ty / 2])
    L = np.array([-w / 2, ty / 2])
    # 闭合回 A
    key_points = [A, B, C, D, E, F, G, H, I, J, K, L, A]

    segments = []
    for i in range(len(key_points) - 1):
        seg = sample_segment(
            key_points[i], key_points[i + 1], num_edge_points, include_endpoint=True
        )
        # 为避免重复点，将除了最后一段外，每段去掉最后一个采样点
        if i < len(key_points) - 2:
            seg = seg[:-1]
        segments.append(seg)
    polygon_local = np.concatenate(segments, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    polygon_rotated = (R @ polygon_local.T).T
    polygon_global = polygon_rotated + np.array([cx, cy])

    return polygon_global


def generate_diagonal_stripe(params, num_edge_points=50):
    """
    Diagonal stripe 模型函数

    参数:
       params = [cx, cy, L, t, theta]
         - cx, cy: 全局平移，将局部条带平移到目标位置
         - L     : 条带沿主方向（局部 x 轴）的长度
         - t     : 条带厚度（局部 y 轴方向），决定条带宽度
         - theta : 整体旋转角（单位: 弧度），旋转后的条带即为对角条带
       num_edge_points: 每条边均匀采样的点数（默认50）

    返回:
       (N x 2) 的 numpy 数组，为闭合的 diagonal stripe 边界采样点，可用于绘制轮廓
    """
    cx, cy, L, t, theta = params

    # 定义局部坐标下条带的四个角点（基于水平延伸）
    A = np.array([0, -t / 2])  # 左下角
    B = np.array([L, -t / 2])  # 右下角
    C = np.array([L, t / 2])  # 右上角
    D = np.array([0, t / 2])  # 左上角

    # 构成闭合路径：A -> B -> C -> D -> A
    key_points = [A, B, C, D, A]

    segments = []
    for i in range(len(key_points) - 1):
        seg = sample_segment(
            key_points[i], key_points[i + 1], num_edge_points, include_endpoint=True
        )
        # 避免重复点：每条边除了最后一段外去掉最后一个采样点
        if i < len(key_points) - 2:
            seg = seg[:-1]
        segments.append(seg)
    stripe_local = np.concatenate(segments, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    stripe_rotated = (R @ stripe_local.T).T
    stripe_global = stripe_rotated + np.array([cx, cy])

    return stripe_global


def generate_l_shape(params, num_edge_points=50):
    """
    L-shape 模型函数

    参数:
       params = [cx, cy, w, h, t, theta]
         - cx, cy : 全局平移，将 L-shape 移动到指定位置
         - w, h   : L-shape 的整体宽度和高度
         - t      : 边框厚度，即水平与垂直臂的宽度（需满足 0 < t < min(w, h)）
         - theta  : 整体旋转角（单位：弧度）
       num_edge_points: 每条边均匀采样的点数（默认50）

    返回:
       (N x 2) 的 numpy 数组，表示闭合 L-shape 边界的采样点
    """
    cx, cy, w, h, t, theta = params

    # 1. 定义 L-shape 的局部关键点
    A = np.array([0, 0])  # 底部左角
    B = np.array([w, 0])  # 底部右角
    C = np.array([w, t])  # 底部右悬臂上沿
    D = np.array([t, t])  # 内拐角
    E = np.array([t, h])  # 垂直臂右侧顶点
    F = np.array([0, h])  # 顶部左角

    # 组成立闭合路径 A → B → C → D → E → F → A
    key_points = [A, B, C, D, E, F, A]

    segments = []
    for i in range(len(key_points) - 1):
        # 为每条边采样（各段尾部除最后一个外去除重复点）
        seg = sample_segment(
            key_points[i], key_points[i + 1], num_edge_points, include_endpoint=True
        )
        if i < len(key_points) - 2:
            seg = seg[:-1]
        segments.append(seg)
    polygon_local = np.concatenate(segments, axis=0)

    # 2. 全局变换：先旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    polygon_rotated = (R @ polygon_local.T).T
    polygon_global = polygon_rotated + np.array([cx, cy])

    return polygon_global


def generate_closed_half_frame(params, num_edge_points=50):
    """
    Closed Half frame 模型函数

    ini
    参数:
       params = [cx, cy, w, h, tx, ty, theta]
         - cx, cy : 全局平移，将局部构造的形状移动到指定位置
         - w, h   : 外框宽度和半框区域的高度（局部坐标中外框左下 (–w/2, 0)，右上 (w/2, h)）
         - tx, ty : 水平和垂直方向的边框内缩厚度（生成内框角点）
         - theta  : 整体旋转角（单位：弧度）
       num_edge_points: 每条边均匀采样的点数（默认50）

    返回:
       (N x 2) 的 numpy 数组，表示闭合的半框多边形的采样边界点
    """
    cx, cy, w, h, tx, ty, theta = params

    # 1. 定义外框（上半区域）的关键角点（局部坐标）
    D = np.array([-w / 2, 0])  # 外框左下
    A = np.array([-w / 2, h])  # 外框左上
    B = np.array([w / 2, h])  # 外框右上
    C = np.array([w / 2, 0])  # 外框右下

    # 2. 定义内框（孔）的关键角点，
    #    内框在水平方向内缩 tx，右侧和左侧均向中心内缩；
    #    在垂直方向，上侧内缩 ty，但下侧保持与外框底边对齐（形成连接桥）
    G = np.array([w / 2 - tx, 0])  # 内框右下
    F = np.array([w / 2 - tx, h - ty])  # 内框右上
    E = np.array([-w / 2 + tx, h - ty])  # 内框左上
    H = np.array([-w / 2 + tx, 0])  # 内框左下

    # 3. 构造闭合路径
    #    依次采样：外框从 D -> A -> B -> C，
    #    然后从 C 连到 G，再沿内框逆序采样：G -> F -> E -> H，
    #    最后闭合 H -> D.
    key_points = [D, A, B, C, G, F, E, H, D]

    segments = []
    for i in range(len(key_points) - 1):
        seg = sample_segment(
            key_points[i], key_points[i + 1], num_edge_points, include_endpoint=True
        )
        # 为避免重复点，除最后一段外每段去掉最后一个点
        if i < len(key_points) - 2:
            seg = seg[:-1]
        segments.append(seg)
    polygon_local = np.concatenate(segments, axis=0)

    # 4. 全局变换：旋转和平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    polygon_rotated = (R @ polygon_local.T).T
    polygon_global = polygon_rotated + np.array([cx, cy])

    return polygon_global


def generate_half_frame(params, num_edge_points=50):
    """
    half frame 模型函数

    参数:
       params = [cx, cy, w, h, t, theta]
         - cx, cy : 全局平移，将局部形状移动到指定位置
         - w, h   : 外框的宽度和高度（这里 h 定义为外框整体高度，半框取上半部分 y ∈ [0, h/2]）
         - t      : 边框厚度（保证 0 < t < min(w/2, h/2)）
         - theta  : 整体旋转角（单位：弧度）
       num_edge_points: 用于采样每条边的点数（默认50个）

    返回:
       返回一个字典对象，包含两个键：
         "outer": 外框上半部分采样点（U 形开放曲线，不闭合下边）
         "inner": 内框上半部分采样点（U 形开放曲线，不闭合左侧，下侧由采样点结束）
    """
    cx, cy, w, h, t, theta = params
    # 定义外框上半部四个关键点（局部坐标）
    # 外框原始矩形中心位于原点，y 轴正上为 h/2，最底部为 -h/2，
    # 但此处我们只取上半部分：y ∈ [0, h/2]
    A = np.array([-w / 2, h / 2])  # 左上角
    B = np.array([w / 2, h / 2])  # 右上角
    C = np.array([w / 2, 0])  # 右下端（开放边界的起始端）
    D = np.array([-w / 2, 0])  # 左下端

    # 构造外框上半部：依次采样上边、右边和左边（下侧不封闭）
    outer_top = sample_segment(A, B, num_edge_points)
    outer_right = sample_segment(B, C, num_edge_points, include_endpoint=False)
    outer_left = sample_segment(D, A, num_edge_points, include_endpoint=False)
    outer_local = np.concatenate([outer_top, outer_right, outer_left], axis=0)

    # 定义内框上半部关键点（局部坐标），内框由外框缩进 t 得到
    E = np.array([-w / 2 + t, h / 2 - t])  # 内框左上角
    F = np.array([w / 2 - t, h / 2 - t])  # 内框右上角
    G = np.array([w / 2 - t, t])  # 内框右下端
    H = np.array([-w / 2 + t, t])  # 内框左下端

    # 采样内框上半部：依次采样上边、右侧和左侧（下侧保持开放）
    inner_top = sample_segment(E, F, num_edge_points)
    inner_right = sample_segment(F, G, num_edge_points, include_endpoint=False)
    inner_left = sample_segment(H, E, num_edge_points, include_endpoint=False)
    inner_local = np.concatenate([inner_top, inner_right, inner_left], axis=0)

    # 全局变换：先旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    outer_global = (R @ outer_local.T).T + np.array([cx, cy])
    inner_global = (R @ inner_local.T).T + np.array([cx, cy])

    return {'outer': outer_global, 'inner': inner_global}


def generate_frame(params, num_edge_points=50):
    """
    frame 模型函数

    参数:
       params = [cx, cy, w, h, t, theta]
         - cx, cy: 全局平移，将局部形状移动到指定位置
         - w, h  : 外框的宽度和高度
         - t     : 边框厚度，应满足 0 < t < min(w/2, h/2)
         - theta : 整体旋转角（单位：弧度）
       num_edge_points: 每条边采样的点数（默认50个）

    返回:
       返回一个字典对象，包含两个键：
         "outer": 外框边界采样点（(N×2) numpy 数组）
         "inner": 内框边界采样点（(M×2) numpy 数组），内框点按顺时针排列形成孔洞
    """
    cx, cy, w, h, t, theta = params

    # 1. 外框角点（局部坐标），逆时针排列，形成闭合多边形
    outer_corners = np.array(
        [
            [-w / 2, -h / 2],
            [w / 2, -h / 2],
            [w / 2, h / 2],
            [-w / 2, h / 2],
            [-w / 2, -h / 2],
        ]
    )

    # 2. 内框角点（局部坐标），顺时针排列，构造内孔
    #    内框通过向外框内缩厚度 t 得到，保证 t < min(w/2, h/2)
    inner_corners = np.array(
        [
            [-w / 2 + t, -h / 2 + t],
            [-w / 2 + t, h / 2 - t],
            [w / 2 - t, h / 2 - t],
            [w / 2 - t, -h / 2 + t],
            [-w / 2 + t, -h / 2 + t],
        ]
    )

    # 3. 分别沿外框和内框每条边进行采样
    outer_local = sample_polygon(outer_corners, num_edge_points)
    inner_local = sample_polygon(inner_corners, num_edge_points)

    # 4. 全局变换：先旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    outer_global = (R @ outer_local.T).T + np.array([cx, cy])
    inner_global = (R @ inner_local.T).T + np.array([cx, cy])

    return {'outer': outer_global, 'inner': inner_global}


def generate_teardrop(params, num_head_points=50, num_tail_points=20):
    """
    teardrop 模型函数

    ini
    参数:
        params = [cx, cy, r, d, theta]
            - cx, cy  : 全局平移（将局部形状移动到指定位置）
            - r       : 头部圆弧的半径（决定上部曲线的形状）
            - d       : 尖端向下延伸的距离（泪滴尾部的位置，d>0）
            - theta   : 整体旋转角（弧度制）
        num_head_points : 采样头部圆弧的点数（默认50个）
        num_tail_points : 采样尾部直线的点数（默认20个，两侧均采样）

    返回:
        (N×2) 的 numpy 数组，每一行为全局坐标下的边界采样点 [x, y]
    """
    cx, cy, r, d, theta = params

    # 1. 生成头部圆弧（局部坐标）
    #    以圆心 (0, r) 、半径 r，角度采样范围从 -pi/3 到 -2pi/3，
    #    得到的 head_points[0] 为右端点，head_points[-1] 为左端点。
    arc_angles = np.linspace(-np.pi / 3, -2 * np.pi / 3, num_head_points)
    head_points = np.column_stack((r * np.cos(arc_angles), r + r * np.sin(arc_angles)))

    # 2. 定义尾部尖端坐标（局部坐标）
    tip = np.array([0, -d])

    # 3. 生成尾部直线采样点
    #    从左端点到尖端；从尖端到右端点
    left_endpoint = head_points[-1]
    right_endpoint = head_points[0]
    tail_left = np.linspace(left_endpoint, tip, num_tail_points, endpoint=True)
    tail_right = np.linspace(tip, right_endpoint, num_tail_points, endpoint=True)

    # 4. 拼接边界：按顺序连接头部圆弧（右端到左端）、左侧尾部（除去重复的左端点）、
    #    再接右侧尾部（除去重复的尖端），形成闭合曲线
    boundary_local = np.concatenate(
        [head_points, tail_left[1:], tail_right[1:]], axis=0
    )

    # 5. 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    boundary_rotated = (R @ boundary_local.T).T
    boundary_global = boundary_rotated + np.array([cx, cy])
    return boundary_global


def generate_chord(params, num_arc_points=100, num_line_points=2):
    """
    chord 模型函数

    参数:
        params = [cx, cy, r, start_angle, sweep]
            - cx, cy      : 圆的全局中心位置
            - r           : 圆的半径
            - start_angle : 圆弧起始角（弧度制）
            - sweep       : 扫过角（弧度制）
        num_arc_points  : 用于采样圆弧部分的点数（默认100个）
        num_line_points : 用于采样弦线上的点数（默认2个，足以表示端点）

    返回:
        (N×2) 的 numpy 数组，每一行为全局坐标下的边界采样点 [x, y]
    """
    cx, cy, r, start_angle, sweep = params
    end_angle = start_angle + sweep

    # 1. 生成弧形采样点（局部坐标）
    arc_angles = np.linspace(start_angle, end_angle, num_arc_points)
    arc_points = np.column_stack((r * np.cos(arc_angles), r * np.sin(arc_angles)))

    # 2. 生成弦线采样点：从弧形终点到弧形起点
    chord_line = np.linspace(
        arc_points[-1], arc_points[0], num_line_points, endpoint=True
    )

    # 为避免重复采样弧形起点，弦线采样时去除第一个点
    boundary_local = np.concatenate([arc_points, chord_line[1:]], axis=0)

    # 3. 全局平移：将局部采样点平移 (cx,cy)
    boundary_global = boundary_local + np.array([cx, cy])

    return boundary_global


def generate_pie(params, num_arc_points=100, num_line_points=2):
    """
    pie 模型函数

    参数:
        params = [cx, cy, r, start_angle, sweep]
            - cx, cy      : 全局中心位置
            - r           : 弧所在圆的半径
            - start_angle : 圆弧起始角（弧度制）
            - sweep       : 扫过角（弧度制），即圆弧的总跨度

        num_arc_points  : 圆弧的采样点数（默认100个）
        num_line_points : 直线边采样点数（默认2个，足够表示直线端点）

    返回:
        (N×2) 的 numpy 数组，每一行为全局坐标下的边界采样点 [x, y]

    构造步骤：
        1. 生成圆弧部分采样点：角度在 [start_angle, start_angle+sweep] 均匀分布，
           对应坐标 (r*cos(angle), r*sin(angle))；
        2. 生成从弧形终点到中心的直线路径采样点；
        3. 生成从中心到弧形起点的直线路径采样点；
        4. 按顺序合并以上各部分，并加上中心偏移 (cx,cy) 得到全局边界点。
    """
    cx, cy, r, start_angle, sweep = params
    end_angle = start_angle + sweep

    # 1. 生成弧形采样点：角度均匀分布
    arc_angles = np.linspace(start_angle, end_angle, num_arc_points)
    arc_points = np.column_stack((r * np.cos(arc_angles), r * np.sin(arc_angles)))

    # 2. 生成从弧形终点到中心的直线采样点（包括两端点）
    line_from_end_to_center = np.linspace(
        arc_points[-1], np.array([0, 0]), num_line_points, endpoint=True
    )

    # 3. 生成从中心到弧形起点的直线采样点（包括两端点）
    line_from_center_to_start = np.linspace(
        np.array([0, 0]), arc_points[0], num_line_points, endpoint=True
    )

    # 4. 合并部分：按顺序连接弧形、从弧形终点到中心，再从中心到弧形起点
    # 注意：直线部分的第一个点已和前一部分重复，因此需去重处理（取直线采样的第 2 个点开始）
    boundary_local = np.concatenate(
        [arc_points, line_from_end_to_center[1:], line_from_center_to_start[1:]], axis=0
    )

    # 全局坐标变换：将所有点平移 (cx,cy)
    boundary_global = boundary_local + np.array([cx, cy])

    return boundary_global


def generate_dodecagon(params, num_points=200):
    """
    dodecagon 模型函数

    参数:
        params = [cx, cy, r, theta]
           - cx, cy : 正十二边形全局质心
           - r      : 外接圆半径（顶点到质心的距离）
           - theta  : 整体逆时针旋转角（单位：弧度）
        num_points : 沿边界采样的总点数（默认200个）

    返回:
        (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 12  # 正十二边形有12条边

    # 在局部构造顶点，初始角设为 π/2（保证上方顶点）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算每条边的长度（理论上正十二边形各边相等，但此处按公式计算以适应一般情况）
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 按边长比例分配采样点数（确保每边至少采样2个点）
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample

    # 调整采样点数确保总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 沿每条边采样：前 n_sides-1 条边不取终点，最后一条边包含终点以闭合曲线
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：先旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_decagon(params, num_points=200):
    """
    decagon 模型函数

    参数：
        params = [cx, cy, r, theta]
          - cx, cy : 正十边形全局质心
          - r      : 外接圆半径（顶点到质心的距离）
          - theta  : 整体逆时针旋转角（单位：弧度）
        num_points : 沿边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 10  # 正十边形有10条边

    # 在局部坐标系下构造顶点，初始角设为 π/2（保证上方顶点）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算各边长度（正十边形各边理论上应相等）
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 按各边长度比例分配采样点数，每边至少采样2个点
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample
    # 调整采样点数确保总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 对各边采样：前 n_sides-1 条边不包含终点，最后一条边包含终点以闭合轮廓
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：首先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_octagon(params, num_points=200):
    """
    octagon 模型函数

    参数：
        params = [cx, cy, r, theta]
          - cx, cy : 正八边形全局质心
          - r      : 外接圆半径（顶点到质心的距离）
          - theta  : 整体旋转角（单位：弧度，逆时针方向）
        num_points : 沿边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 8  # 正八边形有8条边

    # 在局部坐标系下构造顶点，初始角设置为 π/2（保证上方顶点）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算每条边的长度（正八边形理论上各边相等）
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 根据边长按比例分配采样点数，每边至少采样2个点
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample
    # 调整采样点总数确保严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 对各边进行均匀采样：前 n_sides-1 条边不包含终点，最后一条边包含终点以闭合轮廓
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 旋转和平移：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_heptagon(params, num_points=200):
    """
    heptagon 模型函数

    参数：
        params = [cx, cy, r, theta]
          - cx, cy : 正七边形全局质心
          - r      : 外接圆半径（顶点到质心的距离）
          - theta  : 整体旋转角（单位：弧度，逆时针）
        num_points : 沿边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 7  # 正七边形有7条边

    # 在局部坐标下构造顶点，初始角设为 π/2（上方顶点位于正上方）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算各边长度（正七边形各边理论上相等，但仍依实际计算）
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 按各边长度比例分配采样点数，每边至少采样2个点
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample
    # 调整采样点数确保总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 沿各边采样：前n_sides-1条边不包含终点，最后一条边包含终点以闭合边界
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：旋转后平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_hexagon(params, num_points=200):
    """
    hexagon 模型函数

    参数：
        params = [cx, cy, r, theta]
          - cx, cy : 正六边形全局质心
          - r      : 外接圆半径（顶点到质心的距离）
          - theta  : 整体旋转角（单位：弧度，逆时针旋转）
        num_points : 沿边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 6  # 正六边形有6条边

    # 在局部坐标下构造顶点，初始角设为 π/2（确保上方顶点）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算各边长度（正六边形各边应当相等）
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 按边长比例分配采样点，每边至少采样2个点
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 对每条边采样：前 n_sides-1 边用 endpoint=False，最后一边包含终点以闭合边界
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：旋转后平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_regular_pentagon(params, num_points=200):
    """
    regular pentagon 模型函数

    参数：
        params = [cx, cy, r, theta]
          - cx, cy : 正五边形全局质心
          - r      : 外接圆半径
          - theta  : 整体旋转角（单位：弧度，逆时针旋转）
        num_points : 沿边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, r, theta = params
    n_sides = 5  # 正五边形有5条边

    # 在局部坐标下构造正五边形顶点，初始角设为 π/2（顶点位于正上方）
    initial_angle = np.pi / 2
    vertices = []
    for i in range(n_sides):
        angle = initial_angle - i * (2 * np.pi / n_sides)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        vertices.append(np.array([x, y]))
    vertices = np.array(vertices)

    # 计算每条边的长度：顶点依次为 A, B, C, D, E
    lengths = []
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        lengths.append(np.linalg.norm(p2 - p1))
    total_length = sum(lengths)

    # 按各边长度占比分配采样点数，确保每边至少采样2个点
    segment_counts = []
    allocated = 0
    for L in lengths:
        n_sample = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n_sample)
        allocated += n_sample
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 沿各边采样，除最后一边外不包含终点以避免重复的顶点
    for i in range(n_sides):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n_sides]
        if i < n_sides - 1:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=False)
        else:
            seg = np.linspace(p1, p2, segment_counts[i], endpoint=True)
        segments.append(seg)
    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_diamond(params, num_points=200):
    """
    diamond 模型函数

    参数：
        params = [cx, cy, w, h, theta]
          - cx, cy : 菱形全局质心
          - w      : 水平方向顶点间距
          - h      : 垂直方向顶点间距
          - theta  : 整体旋转角（单位：弧度，逆时针）
        num_points : 沿菱形边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, w, h, theta = params

    # 在局部坐标下定义四个顶点（菱形各顶点, 质心为原点）
    A = np.array([0, h / 2])  # 顶部点
    B = np.array([w / 2, 0])  # 右侧点
    C = np.array([0, -h / 2])  # 底部点
    D = np.array([-w / 2, 0])  # 左侧点

    # 计算各边长度
    L1 = np.linalg.norm(B - A)  # A→B
    L2 = np.linalg.norm(C - B)  # B→C
    L3 = np.linalg.norm(D - C)  # C→D
    L4 = np.linalg.norm(A - D)  # D→A
    total_length = L1 + L2 + L3 + L4

    # 按各边长度分配采样点，每边至少采样 2 个点
    seg_lengths = [L1, L2, L3, L4]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整采样点数量确保总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 边 A → B（不包含终点）
    seg1 = np.linspace(A, B, segment_counts[0], endpoint=False)
    segments.append(seg1)
    # 边 B → C（不包含终点）
    seg2 = np.linspace(B, C, segment_counts[1], endpoint=False)
    segments.append(seg2)
    # 边 C → D（不包含终点）
    seg3 = np.linspace(C, D, segment_counts[2], endpoint=False)
    segments.append(seg3)
    # 边 D → A（包含终点以闭合边界）
    seg4 = np.linspace(D, A, segment_counts[3], endpoint=True)
    segments.append(seg4)

    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_trapezoid(params, num_points=200):
    """
    trapezoid 模型函数

    参数：
        params = [cx, cy, w_bottom, w_top, h, theta]
          - cx, cy   : 梯形全局质心
          - w_bottom : 下底边宽度
          - w_top    : 上底边宽度
          - h        : 梯形高度（上下底边之间的垂直距离）
          - theta    : 整体旋转角（单位：弧度，逆时针旋转）
        num_points : 沿梯形边界采样的总点数（默认200个）

    返回：
         (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, w_bottom, w_top, h, theta = params

    # 局部坐标下构造顶点
    A = np.array([-w_bottom / 2, -h / 2])  # 下底左端
    B = np.array([w_bottom / 2, -h / 2])  # 下底右端
    C = np.array([w_top / 2, h / 2])  # 上底右端
    D = np.array([-w_top / 2, h / 2])  # 上底左端

    # 计算各边长度
    L1 = np.linalg.norm(B - A)  # 下底边长度，等于 w_bottom
    L2 = np.linalg.norm(C - B)  # 右侧斜边
    L3 = np.linalg.norm(D - C)  # 上底边长度，等于 w_top
    L4 = np.linalg.norm(A - D)  # 左侧斜边，和 L2 相等
    total_length = L1 + L2 + L3 + L4

    # 按各边长度比例分配采样点数，保证每边至少有 2 个采样点
    seg_lengths = [L1, L2, L3, L4]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整采样点数确保总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # 边 A → B（不包含终点）
    seg1 = np.linspace(A, B, segment_counts[0], endpoint=False)
    segments.append(seg1)
    # 边 B → C（不包含终点）
    seg2 = np.linspace(B, C, segment_counts[1], endpoint=False)
    segments.append(seg2)
    # 边 C → D（不包含终点）
    seg3 = np.linspace(C, D, segment_counts[2], endpoint=False)
    segments.append(seg3)
    # 边 D → A（闭合边界，包含终点）
    seg4 = np.linspace(D, A, segment_counts[3], endpoint=True)
    segments.append(seg4)

    pts_local = np.concatenate(segments, axis=0)

    # 旋转和平移至全局坐标：先旋转后平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_parallelogram(params, num_points=200):
    """
    parallelogram 模型函数
    参数：
    params = [cx, cy, w, h, shear, theta]
    - cx, cy : 平行四边形全局质心
    - w : 底边宽度
    - h : 两组平行边之间的距离（高度）
    - shear : 上边相对于下边的水平偏移
    - theta : 整体旋转角（单位：弧度，逆时针）
    num_points : 沿边界采样的总点数（默认为200个）
    返回：
    (N×2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, w, h, shear, theta = params

    # 局部坐标下定义顶点，使质心位于原点
    A = np.array([-w / 2 - shear / 2, -h / 2])  # 底部左
    B = np.array([w / 2 - shear / 2, -h / 2])  # 底部右
    C = np.array([w / 2 + shear / 2, h / 2])  # 顶部右
    D = np.array([-w / 2 + shear / 2, h / 2])  # 顶部左

    # 计算各边长度
    L1 = np.linalg.norm(B - A)  # 底边，长度为 w
    L2 = np.linalg.norm(C - B)  # 右侧边：sqrt(shear^2 + h^2)
    L3 = np.linalg.norm(D - C)  # 顶边，长度为 w
    L4 = np.linalg.norm(A - D)  # 左侧边：sqrt(shear^2 + h^2)
    total_length = L1 + L2 + L3 + L4

    # 按比例分配采样点数（确保每条边至少采样2个点）
    seg_lengths = [L1, L2, L3, L4]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # Segment 1: 从 A 到 B，不包含终点
    seg1 = np.linspace(A, B, segment_counts[0], endpoint=False)
    segments.append(seg1)

    # Segment 2: 从 B 到 C，不包含终点
    seg2 = np.linspace(B, C, segment_counts[1], endpoint=False)
    segments.append(seg2)

    # Segment 3: 从 C 到 D，不包含终点
    seg3 = np.linspace(C, D, segment_counts[2], endpoint=False)
    segments.append(seg3)

    # Segment 4: 从 D 到 A，包含终点以闭合边界
    seg4 = np.linspace(D, A, segment_counts[3], endpoint=True)
    segments.append(seg4)

    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_right_triangle(params, num_points=200):
    """
    right triangle 模型函数
    参数：
    params = [cx, cy, w, h, theta]
    - cx, cy: 三角形质心在全局坐标中的位置
    - w : 右三角形中直角对应的水平边长度
    - h : 右三角形中直角对应的垂直边长度
    - theta : 整体旋转角（单位：弧度，逆时针）
    num_points : 沿三角形边界采样的总点数（默认为200个）
    返回：
    (N x 2) 的 numpy 数组，每一行为全局坐标下的采样点 [x, y]
    """
    cx, cy, w, h, theta = params

    # 局部顶点（将右三角形平移，使质心位于原点）
    # 初始右三角形顶点为： (0,0), (w,0) 和 (0,h)，其质心在 (w/3, h/3)
    V1 = np.array([-w / 3, -h / 3])  # 对应 (0,0) 平移后
    V2 = np.array([2 * w / 3, -h / 3])  # 对应 (w,0) 平移后
    V3 = np.array([-w / 3, 2 * h / 3])  # 对应 (0,h) 平移后

    # 计算各边长度
    L1 = np.linalg.norm(V2 - V1)  # 边 V1→V2，长度应等于 w
    L2 = np.linalg.norm(V3 - V2)  # 边 V2→V3（斜边）
    L3 = np.linalg.norm(V1 - V3)  # 边 V3→V1（另一斜边）
    total_length = L1 + L2 + L3

    # 按各边长度比例分配采样点数，确保每段至少包含2个点
    seg_lengths = [L1, L2, L3]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整采样点数使得总数正好为 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # Segment 1: 从 V1 到 V2
    seg1 = np.linspace(V1, V2, segment_counts[0], endpoint=False)
    segments.append(seg1)

    # Segment 2: 从 V2 到 V3
    seg2 = np.linspace(V2, V3, segment_counts[1], endpoint=False)
    segments.append(seg2)

    # Segment 3: 从 V3 回到 V1（闭合边界，包含终点）
    seg3 = np.linspace(V3, V1, segment_counts[2], endpoint=True)
    segments.append(seg3)

    pts_local = np.concatenate(segments, axis=0)

    # 全局变换：先旋转再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_isosceles_triangle(params, num_points=200):
    """
    isosceles triangle 模型函数
    参数：
    params = [cx, cy, w, h, theta]
    - cx, cy : 三角形质心的全局坐标
    - w : 底边宽度
    - h : 三角形高度（底边到顶点的距离）
    - theta : 整体旋转角（单位：弧度，逆时针）
    num_points : 沿边界采样的总点数（默认为200个）
    返回：
    (N x 2) 的 numpy 数组，每一行为在全局坐标下的一个采样点 [x, y]
    """
    cx, cy, w, h, theta = params

    # 在局部坐标下构造三角形顶点，使质心在原点
    V1 = np.array([-w / 2, -h / 3])  # 左底顶点
    V2 = np.array([w / 2, -h / 3])  # 右底顶点
    V3 = np.array([0, 2 * h / 3])  # 顶点

    # 计算各边长度
    L1 = np.linalg.norm(V2 - V1)  # 底边，长度 = w
    L2 = np.linalg.norm(V3 - V2)  # 右侧边
    L3 = np.linalg.norm(V1 - V3)  # 左侧边
    total_length = L1 + L2 + L3

    # 根据比例分配各边的采样点数（确保每段至少 2 个采样点）
    seg_lengths = [L1, L2, L3]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整分配使总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments = []
    # Segment 1: 底边，从 V1 到 V2
    seg1 = np.linspace(V1, V2, segment_counts[0], endpoint=False)
    segments.append(seg1)

    # Segment 2: 右侧边，从 V2 到 V3
    seg2 = np.linspace(V2, V3, segment_counts[1], endpoint=False)
    segments.append(seg2)

    # Segment 3: 左侧边，从 V3 到 V1，包含终点以闭合边界
    seg3 = np.linspace(V3, V1, segment_counts[2], endpoint=True)
    segments.append(seg3)

    # 合并所有段
    pts_local = np.concatenate(segments, axis=0)

    # 旋转和平移至全局坐标
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_round_diagonal_corner_rectangle(params, num_points=250):
    """
    round diagonal corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, r]
    cx, cy : 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角（单位弧度，逆时针）
    r : 圆角半径（要求 r <= min(w/2, h/2)），用于右上和左下两个对角的圆角化
    num_points : 沿边界采样的总点数（默认为250个）
    返回：
    (N x 2) 的 numpy 数组，每一行为全局坐标下的采样点 (x, y)
    """
    cx, cy, w, h, theta, r = params
    r = min(r, w / 2, h / 2)

    # 定义局部坐标下的关键点
    P_TL = np.array([-w / 2, h / 2])  # 顶左角（保持尖角）
    T_R1 = np.array([w / 2 - r, h / 2])  # 右上角——顶边截断点
    T_R2 = np.array([w / 2, h / 2 - r])  # 右上角——右边截断点
    P_BR = np.array([w / 2, -h / 2])  # 右下角（保持尖角）
    T_BL = np.array([-w / 2 + r, -h / 2])  # 底左角——底边截断点
    T_BL2 = np.array([-w / 2, -h / 2 + r])  # 底左角——左边截断点

    # 定义圆弧中心
    O_TR = np.array([w / 2 - r, h / 2 - r])  # 右上圆弧中心
    O_BL = np.array([-w / 2 + r, -h / 2 + r])  # 底左圆弧中心

    # 计算各段长度
    L1 = np.linalg.norm(T_R1 - P_TL)  # 顶边直线长度 = (w - r)
    L2 = (np.pi / 2) * r  # 右上圆弧长度（90°圆弧）
    L3 = np.linalg.norm(P_BR - T_R2)  # 右边直线长度 = h - r
    L4 = np.linalg.norm(P_BR - T_BL)  # 底边直线长度 = (w - r)
    L5 = (np.pi / 2) * r  # 底左圆弧长度（90°圆弧）
    L6 = np.linalg.norm(P_TL - T_BL2)  # 左边直线长度 = h - r
    total_length = L1 + L2 + L3 + L4 + L5 + L6

    # 按各段比例分配采样点（确保每段至少两个采样点以保持连续性）
    seg_lengths = [L1, L2, L3, L4, L5, L6]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments_pts = []

    # Segment 1: 顶边直线，从 P_TL 到 T_R1
    seg1 = np.linspace(P_TL, T_R1, segment_counts[0], endpoint=False)
    segments_pts.append(seg1)

    # Segment 2: 右上圆弧，从 T_R1 到 T_R2
    # 对于 T_R1 相对于 O_TR 的向量为 (0, r)（角度 π/2），对于 T_R2 为 (r, 0)（角度 0）
    angles_TR = np.linspace(np.pi / 2, 0, segment_counts[1], endpoint=False)
    seg2 = np.array([O_TR + r * np.array([np.cos(a), np.sin(a)]) for a in angles_TR])
    segments_pts.append(seg2)

    # Segment 3: 右边直线，从 T_R2 到 P_BR
    seg3 = np.linspace(T_R2, P_BR, segment_counts[2], endpoint=False)
    segments_pts.append(seg3)

    # Segment 4: 底边直线，从 P_BR 到 T_BL
    seg4 = np.linspace(P_BR, T_BL, segment_counts[3], endpoint=False)
    segments_pts.append(seg4)

    # Segment 5: 底左圆弧，从 T_BL 到 T_BL2
    # 对于 T_BL 相对于 O_BL 的向量为 (0, -r)（角度 3π/2），对于 T_BL2 为 (-r, 0)（角度 π）
    angles_BL = np.linspace(3 * np.pi / 2, np.pi, segment_counts[4], endpoint=False)
    seg5 = np.array([O_BL + r * np.array([np.cos(a), np.sin(a)]) for a in angles_BL])
    segments_pts.append(seg5)

    # Segment 6: 左边直线，从 T_BL2 到 P_TL，包含终点以闭合边界
    seg6 = np.linspace(T_BL2, P_TL, segment_counts[5], endpoint=True)
    segments_pts.append(seg6)

    pts_local = np.concatenate(segments_pts, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_round_same_side_corner_rectangle(params, num_points=300):
    """
    round same side corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, r]
    - cx, cy : 矩形中心坐标
    - w, h : 矩形宽和高
    - theta : 整体旋转角（单位：弧度，逆时针）
    - r : 上侧（左上、右上）圆角半径（要求 r <= min(w/2, h/2)）
    num_points : 总采样点数（默认为 300个）

    返回：
      (N x 2) 的 numpy 数组，每行为在全局坐标下的一个采样点 (x, y)
    """
    cx, cy, w, h, theta, r = params
    r = min(r, w / 2, h / 2)  # 限制 r 不超过矩形尺寸的一半

    # 定义局部坐标下的关键点
    # 各点按顺时针顺序
    P_BL = np.array([-w / 2, -h / 2])  # 左下角（起点）
    P_left_top = np.array([-w / 2, h / 2 - r])  # 左上边与圆角起点
    P_TL_flat = np.array([-w / 2 + r, h / 2])  # 左上圆角终点，同时为顶边起点
    P_TR_flat = np.array([w / 2 - r, h / 2])  # 右上圆角起点，同时为顶边终点
    P_right_top = np.array([w / 2, h / 2 - r])  # 右上圆角终点
    P_BR = np.array([w / 2, -h / 2])  # 右下角
    # 底边：从 P_BR 返回至 P_BL

    # 定义圆弧的圆心
    O_left = np.array([-w / 2 + r, h / 2 - r])  # 左上圆弧中心
    O_right = np.array([w / 2 - r, h / 2 - r])  # 右上圆弧中心

    # 计算各段长度
    L1 = np.linalg.norm(P_left_top - P_BL)  # 左侧直线：长度 = h - r
    L2 = (np.pi / 2) * r  # 左上圆弧：90°圆弧长度
    L3 = np.linalg.norm(P_TR_flat - P_TL_flat)  # 顶边直线：长度 = w - 2r
    L4 = (np.pi / 2) * r  # 右上圆弧：90°圆弧长度
    L5 = np.linalg.norm(P_BR - P_right_top)  # 右侧直线：长度 = h - r
    L6 = np.linalg.norm(P_BR - P_BL)  # 底边直线：长度 = w

    total_length = L1 + L2 + L3 + L4 + L5 + L6

    seg_lengths = [L1, L2, L3, L4, L5, L6]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    segments_pts = []

    # Segment 1: 左侧直线，从 P_BL 到 P_left_top
    seg1 = np.linspace(P_BL, P_left_top, segment_counts[0], endpoint=False)
    segments_pts.append(seg1)

    # Segment 2: 左上圆弧
    # 对于 P_left_top 相对于 O_left 的向量：(-w/2, h/2 - r) - (-w/2 + r, h/2 - r) = (-r, 0)，角度为 π
    # 对于 P_TL_flat 相对于 O_left 的向量：(-w/2 + r, h/2) - (-w/2 + r, h/2 - r) = (0, r)，角度为 π/2
    angles_left = np.linspace(np.pi, np.pi / 2, segment_counts[1], endpoint=False)
    seg2 = np.array(
        [O_left + r * np.array([np.cos(a), np.sin(a)]) for a in angles_left]
    )
    segments_pts.append(seg2)

    # Segment 3: 顶边直线，从 P_TL_flat 到 P_TR_flat
    seg3 = np.linspace(P_TL_flat, P_TR_flat, segment_counts[2], endpoint=False)
    segments_pts.append(seg3)

    # Segment 4: 右上圆弧
    # 对于 P_TR_flat 相对于 O_right： (w/2 - r, h/2) - (w/2 - r, h/2 - r) = (0, r)，角度为 π/2
    # 对于 P_right_top 相对于 O_right： (w/2, h/2 - r) - (w/2 - r, h/2 - r) = (r, 0)，角度为 0
    angles_right = np.linspace(np.pi / 2, 0, segment_counts[3], endpoint=False)
    seg4 = np.array(
        [O_right + r * np.array([np.cos(a), np.sin(a)]) for a in angles_right]
    )
    segments_pts.append(seg4)

    # Segment 5: 右侧直线，从 P_right_top 到 P_BR
    seg5 = np.linspace(P_right_top, P_BR, segment_counts[4], endpoint=False)
    segments_pts.append(seg5)

    # Segment 6: 底边直线，从 P_BR 回到 P_BL（闭合曲线，包含终点）
    seg6 = np.linspace(P_BR, P_BL, segment_counts[5], endpoint=True)
    segments_pts.append(seg6)

    pts_local = np.concatenate(segments_pts, axis=0)

    # 对局部采样点进行旋转和平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_round_single_corner_rectangle(params, num_points=200):
    """
    round single corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, r]
    cx, cy : 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角（单位：弧度，逆时针）
    r : 圆角半径（要求 r <= min(w/2, h/2)）
    num_points : 沿边界采样的总点数（默认为 200 个）
    返回：
    (N x 2) 数组，每一行为模型边界采样点 (x, y)（全局坐标）
    """
    cx, cy, w, h, theta, r = params
    # 确保圆角半径不超过矩形的一半尺寸
    r = min(r, w / 2, h / 2)

    # 定义局部坐标下的关键点（以矩形中心为原点，未旋转）
    P1 = np.array([-w / 2, h / 2])  # 左上角
    T1 = np.array([w / 2 - r, h / 2])  # 顶边与圆弧起点
    T2 = np.array([w / 2, h / 2 - r])  # 右边与圆弧终点
    P3 = np.array([w / 2, -h / 2])  # 右下角
    P4 = np.array([-w / 2, -h / 2])  # 左下角

    # 定义各边段：
    # Segment 1: 直线从 P1 到 T1（上边的左侧部分）
    # Segment 2: 圆弧从 T1 到 T2（右上角圆弧）
    # Segment 3: 直线从 T2 到 P3（右边下侧部分）
    # Segment 4: 直线从 P3 到 P4（底边）
    # Segment 5: 直线从 P4 到 P1（左侧边）

    # 计算每段长度
    L1 = np.linalg.norm(T1 - P1)  # 上边直线长度 = (w - r)
    L2 = (np.pi / 2) * r  # 圆弧长度（90°圆弧）
    L3 = np.linalg.norm(P3 - T2)  # 右边直线长度 = h - r
    L4 = np.linalg.norm(P3 - P4)  # 底边长度 = w
    L5 = np.linalg.norm(P1 - P4)  # 左侧边长度 = h
    total_length = L1 + L2 + L3 + L4 + L5

    # 根据各段比例分配采样点数，确保每段至少有 2 个采样点
    seg_lengths = [L1, L2, L3, L4, L5]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整采样点总数严格等于 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    pts_segments = []
    # Segment 1: 直线从 P1 到 T1
    pts_seg1 = np.linspace(P1, T1, segment_counts[0], endpoint=False)
    pts_segments.append(pts_seg1)

    # Segment 2: 圆弧从 T1 到 T2
    # 圆弧中心 O = (w/2 - r, h/2 - r)
    O = np.array([w / 2 - r, h / 2 - r])
    # 对应角度从 π/2 到 0（不包含终点以免重复）
    arc_angles = np.linspace(np.pi / 2, 0, segment_counts[1], endpoint=False)
    pts_seg2 = np.array(
        [O + r * np.array([np.cos(ang), np.sin(ang)]) for ang in arc_angles]
    )
    pts_segments.append(pts_seg2)

    # Segment 3: 直线从 T2 到 P3（右边）
    pts_seg3 = np.linspace(T2, P3, segment_counts[2], endpoint=False)
    pts_segments.append(pts_seg3)

    # Segment 4: 直线从 P3 到 P4（底边）
    pts_seg4 = np.linspace(P3, P4, segment_counts[3], endpoint=False)
    pts_segments.append(pts_seg4)

    # Segment 5: 直线从 P4 到 P1（左边），包含终点确保闭合
    pts_seg5 = np.linspace(P4, P1, segment_counts[4], endpoint=True)
    pts_segments.append(pts_seg5)

    # 合并所有段的采样点
    pts_local = np.concatenate(pts_segments, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_snip_and_round_single_corner_rectangle(params, num_points=200):
    """
    snip and round single corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, d]
    cx, cy : 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角（单位：弧度，逆时针）
    d : 截取长度，同时定义圆弧半径，用于右上角（要求 d <= min(w/2, h/2)）
    num_points : 沿边界采样的总点数（默认为 200 个）
    返回：
    (N x 2) 数组，每一行为模型边界采样点 (x, y)（全局坐标）
    """
    cx, cy, w, h, theta, d = params
    # 限制 d 的大小
    d = min(d, w / 2, h / 2)

    # 定义局部坐标下的关键点（以矩形中心为原点，不旋转）
    P1 = np.array([-w / 2, h / 2])  # 左上角（保持不变）
    A1 = np.array([w / 2 - d, h / 2])  # 顶边截取点（替换右上角的一部分）
    A2 = np.array([w / 2, h / 2 - d])  # 右边截取点（替换右上角另一部分）
    P4 = np.array([w / 2, -h / 2])  # 右下角（保持不变）
    P5 = np.array([-w / 2, -h / 2])  # 左下角（保持不变])

    # 定义各边段：
    # Segment 1: 从 P1 到 A1（未修改顶边左部分）
    # Segment 2: 圆弧从 A1 到 A2
    # Segment 3: 从 A2 到 P4（右边）
    # Segment 4: 从 P4 到 P5（底边）
    # Segment 5: 从 P5 回到 P1（左边）

    # 计算各段长度
    L1 = np.linalg.norm(A1 - P1)  # 顶边部分长度 = (w/2 - d) - (-w/2) = w - d
    L2 = (np.pi / 2) * d  # 圆弧长度（90°圆弧，半径 d）
    L3 = np.linalg.norm(P4 - A2)  # 右边长度 = (h/2 - d) - (-h/2) = h - d
    L4 = np.linalg.norm(P5 - P4)  # 底边长度 = w
    L5 = np.linalg.norm(P1 - P5)  # 左边长度 = h
    total_length = L1 + L2 + L3 + L4 + L5

    # 根据各段长度，按比例分配采样点数（每段最少 2 个点以确保连续性）
    seg_lengths = [L1, L2, L3, L4, L5]
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    pts_segments = []
    # Segment 1: 直线从 P1 到 A1
    pts_seg1 = np.linspace(P1, A1, segment_counts[0], endpoint=False)
    pts_segments.append(pts_seg1)

    # Segment 2: 圆弧从 A1 到 A2
    # 圆弧中心 O = (w/2-d, h/2-d)
    O = np.array([w / 2 - d, h / 2 - d])
    # 对应角度从 90° (pi/2) 到 0°，注意这里按降序采样
    arc_angles = np.linspace(np.pi / 2, 0, segment_counts[1], endpoint=False)
    pts_seg2 = np.array(
        [O + d * np.array([np.cos(ang), np.sin(ang)]) for ang in arc_angles]
    )
    pts_segments.append(pts_seg2)

    # Segment 3: 直线从 A2 到 P4
    pts_seg3 = np.linspace(A2, P4, segment_counts[2], endpoint=False)
    pts_segments.append(pts_seg3)

    # Segment 4: 直线从 P4 到 P5
    pts_seg4 = np.linspace(P4, P5, segment_counts[3], endpoint=False)
    pts_segments.append(pts_seg4)

    # Segment 5: 直线从 P5 到 P1，包含终点以闭合多边形
    pts_seg5 = np.linspace(P5, P1, segment_counts[4], endpoint=True)
    pts_segments.append(pts_seg5)

    pts_local = np.concatenate(pts_segments, axis=0)

    # 对局部坐标应用旋转和平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_snip_diagonal_corner_rectangle(params, num_points=200):
    """
    snip diagonal corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, d]
    cx, cy: 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角（弧度，逆时针）
    d : 截取长度，用于剪切右上角和左下角，要求 d <= min(w/2, h/2)
    num_points: 沿边界采样的总点数（默认 200 个）
    返回:
    (N x 2) 数组，每一行为采样点 (x, y) 坐标（全局坐标）
    """
    cx, cy, w, h, theta, d = params
    # 限制截断长度
    d = min(d, w / 2, h / 2)

    # 定义局部坐标下的关键点（以矩形中心为原点，不旋转）
    # 右上角原始点 (w/2, h/2) 截断后变为两个点：B, C
    # 左下角原始点 (-w/2, -h/2) 截断后变为两个点：E, F
    A = np.array([-w / 2, h / 2])  # 左上角（不变）
    B = np.array([w / 2 - d, h / 2])  # 顶边，右上角截断前
    C = np.array([w / 2, h / 2 - d])  # 右边，右上角截断后
    D = np.array([w / 2, -h / 2])  # 右下角（不变）
    E = np.array([-w / 2 + d, -h / 2])  # 底边，左下角截断前
    F = np.array([-w / 2, -h / 2 + d])  # 左边，左下角截断后

    # 按顺时针依次构造闭合边界：A -> B -> C -> D -> E -> F -> A
    endpoints = [A, B, C, D, E, F, A]

    # 计算各线段长度
    seg_lengths = []
    for i in range(len(endpoints) - 1):
        seg_lengths.append(np.linalg.norm(endpoints[i + 1] - endpoints[i]))
    total_length = sum(seg_lengths)

    # 按各段长度分配采样点数，确保每段至少 2 个点
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整使总采样点数恰好为 num_points
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    # 对各边段线性采样（除最后一段外 endpoint=False 以避免重复采样公共点）
    pts_segments = []
    for i in range(len(endpoints) - 1):
        if i < len(endpoints) - 2:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=False
            )
        else:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=True
            )
        pts_segments.append(pts_seg)

    pts_local = np.concatenate(pts_segments, axis=0)

    # 全局变换：先旋转，再平移
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])

    return pts_global


def generate_snip_same_side_corner_rectangle(params, num_points=200):
    """
    snip same side corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, d]
    cx, cy: 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角（单位：弧度，逆时针）
    d : 截取长度，用于剪切上边两个角，要求 d <= min(w/2, h/2)
    num_points: 沿边界采样的总点数（默认为 200 个）
    返回:
    (N x 2) 数组，每一行为采样点 (x, y) 坐标（全局坐标）
    """
    cx, cy, w, h, theta, d = params
    d = min(d, w / 2, h / 2)  # 确保截取长度在合理范围内

    # 定义局部坐标下的关键点（以矩形中心为原点，不旋转）
    # 设矩形左右边界为 -w/2 和 w/2，上下边界为 h/2 和 -h/2
    # 对上边的两个顶角进行截断，构造如下顺时针排列的点：
    P1 = np.array([-w / 2, -h / 2])  # 左下角
    P2 = np.array([w / 2, -h / 2])  # 右下角
    P3 = np.array([w / 2, h / 2 - d])  # 右边上部（截断前的点）
    P4 = np.array([w / 2 - d, h / 2])  # 右上角截断点
    P5 = np.array([-w / 2 + d, h / 2])  # 左上角截断点
    P6 = np.array([-w / 2, h / 2 - d])  # 左边上部（截断前的点)
    # 闭合边界：顺序为 P1→P2→P3→P4→P5→P6→回到P1
    endpoints = [P1, P2, P3, P4, P5, P6, P1]

    # 计算各边段长度
    seg_lengths = []
    for i in range(len(endpoints) - 1):
        seg_lengths.append(np.linalg.norm(endpoints[i + 1] - endpoints[i]))
    total_length = sum(seg_lengths)

    # 按各边段比例分配采样点数（每段至少分配 2 个点以确保连续性）
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整确保总采样点数恰好为 num_points，将差异加减到采样最多的边段上
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    # 对各边段进行线性采样，除最后一段外不重复终点
    pts_segments = []
    for i in range(len(endpoints) - 1):
        if i < len(endpoints) - 2:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=False
            )
        else:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=True
            )
        pts_segments.append(pts_seg)

    pts_local = np.concatenate(pts_segments, axis=0)

    # 对局部采样点应用旋转和平移，即全局变换
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_snip_single_corner_rectangle(params, num_points=200):
    """
    snip single corner rectangle 模型函数
    参数：
    params = [cx, cy, w, h, theta, d]
    cx, cy : 矩形中心坐标
    w, h : 矩形宽和高
    theta : 整体旋转角 (弧度，逆时针)
    d : 截取长度，用于剪切右上角，d 应不超过 min(w/2, h/2)
    num_points: 沿边界采样的总点数（默认为 200）
    返回：
    (N x 2) 数组，表示模型边界的采样点（全局坐标）。
    """
    cx, cy, w, h, theta, d = params
    # 限制 d 不超过允许的最大值
    d = min(d, w / 2, h / 2)

    # 定义局部坐标下的关键点（未旋转、以矩形中心为原点）
    P1 = np.array([-w / 2, h / 2])  # 左上角
    P2 = np.array([w / 2 - d, h / 2])  # 顶边结束位置（右上角截断前）
    P3 = np.array([w / 2, h / 2 - d])  # 截断后右边起始位置
    P4 = np.array([w / 2, -h / 2])  # 右下角
    P5 = np.array([-w / 2, -h / 2])  # 左下角

    # 按顺时针依次构造闭合边界（从 P1 开始回到 P1）
    endpoints = [P1, P2, P3, P4, P5, P1]

    # 计算各边段长度
    seg_lengths = []
    for i in range(len(endpoints) - 1):
        seg_lengths.append(np.linalg.norm(endpoints[i + 1] - endpoints[i]))
    total_length = sum(seg_lengths)

    # 根据各段长度按比例分配采样点数（每段至少分配 2 个点以保证连续性）
    segment_counts = []
    allocated = 0
    for L in seg_lengths:
        n = max(2, int(round((L / total_length) * num_points)))
        segment_counts.append(n)
        allocated += n
    # 调整使得总点数精确为 num_points（将调整分配给最长的边段）
    discrepancy = allocated - num_points
    if discrepancy != 0:
        idx = np.argmax(segment_counts)
        segment_counts[idx] -= discrepancy

    # 对每个边段采用线性插值采样（除最后一段外不包含结束点以避免重复）
    pts_segments = []
    for i in range(len(endpoints) - 1):
        # 最后一段确保包含终点（使得闭合多边形最后回到起点）
        if i < len(endpoints) - 2:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=False
            )
        else:
            pts_seg = np.linspace(
                endpoints[i], endpoints[i + 1], segment_counts[i], endpoint=True
            )
        pts_segments.append(pts_seg)

    pts_local = np.concatenate(pts_segments, axis=0)

    # 对局部采样点应用整体旋转及平移变换
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts_rotated = (R @ pts_local.T).T
    pts_global = pts_rotated + np.array([cx, cy])
    return pts_global


def generate_rounded_rectangle(params, num_points=200):
    """
    圆角矩形模型函数
    参数：
    params = [cx, cy, w, h, theta, r]
    cx, cy：矩形中心坐标
    w, h：矩形的宽和高
    theta：全局旋转角（弧度，逆时针）
    r：圆角半径（不应超过 min(w/2, h/2)）
    num_points：采样点总数，默认 200 个。
    返回：
    一个 (N x 2) 的数组，表示圆角矩形边界上的采样点。
    """
    cx, cy, w, h, theta, r = params
    # 防止圆角半径超过允许范围
    r = min(r, w / 2, h / 2)
    half_w = w / 2
    half_h = h / 2

    # 将圆角矩形外围分为 8 个连续的段：
    # 4 条直线段 和 4 个圆角弧段
    # 为简化后续采样，这里每个段使用相同的采样点数（若 num_points 不是8的倍数则近似）
    s = max(3, num_points // 8)  # 每个段的采样点数

    # 定义局部坐标系下的各关键点
    # Segment A: 顶边直线段，从左侧 + 圆角终点 到右侧 - 圆角起点
    A = np.array([-half_w + r, half_h])
    B = np.array([half_w - r, half_h])
    seg_A = np.linspace(A, B, s, endpoint=False)

    # Segment B: 顶右圆角弧，中心在 (half_w - r, half_h - r)
    center_TR = np.array([half_w - r, half_h - r])
    # 弧线从顶边（角度 = π/2，即指向上）到右边（角度 = 0，即指向右）
    angles_B = np.linspace(np.pi / 2, 0, s, endpoint=False)
    seg_B = np.column_stack(
        (center_TR[0] + r * np.cos(angles_B), center_TR[1] + r * np.sin(angles_B))
    )

    # Segment C: 右边直线段，从上侧圆角终点到下侧圆角起点
    C = np.array([half_w, half_h - r])
    D = np.array([half_w, -half_h + r])
    seg_C = np.linspace(C, D, s, endpoint=False)

    # Segment D: 右下圆角弧，中心在 (half_w - r, -half_h + r)
    center_BR = np.array([half_w - r, -half_h + r])
    # 弧线从右边（角度 = 0）到下边（角度 = -π/2）
    angles_D = np.linspace(0, -np.pi / 2, s, endpoint=False)
    seg_D = np.column_stack(
        (center_BR[0] + r * np.cos(angles_D), center_BR[1] + r * np.sin(angles_D))
    )

    # Segment E: 底边直线段，从右侧圆角终点到左侧圆角起点
    E = np.array([half_w - r, -half_h])
    F = np.array([-half_w + r, -half_h])
    seg_E = np.linspace(E, F, s, endpoint=False)

    # Segment F: 左下圆角弧，中心在 (-half_w + r, -half_h + r)
    center_BL = np.array([-half_w + r, -half_h + r])
    # 弧线从下边（角度 = -π/2）到左边（角度 = -π）
    angles_F = np.linspace(-np.pi / 2, -np.pi, s, endpoint=False)
    seg_F = np.column_stack(
        (center_BL[0] + r * np.cos(angles_F), center_BL[1] + r * np.sin(angles_F))
    )

    # Segment G: 左边直线段，从下侧圆角终点到上侧圆角起点
    G = np.array([-half_w, -half_h + r])
    H = np.array([-half_w, half_h - r])
    seg_G = np.linspace(G, H, s, endpoint=False)

    # Segment H: 左上圆角弧，中心在 (-half_w + r, half_h - r)
    center_TL = np.array([-half_w + r, half_h - r])
    # 弧线从左边（角度 = π）到顶边（角度 = π/2）
    angles_H = np.linspace(np.pi, np.pi / 2, s, endpoint=False)
    seg_H = np.column_stack(
        (center_TL[0] + r * np.cos(angles_H), center_TL[1] + r * np.sin(angles_H))
    )

    # 拼接所有段
    points_local = np.concatenate(
        [seg_A, seg_B, seg_C, seg_D, seg_E, seg_F, seg_G, seg_H], axis=0
    )

    # 旋转变换：构造旋转矩阵
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # 对所有局部点应用旋转，并加上平移 (cx, cy)
    points_rotated = (R @ points_local.T).T
    points_global = points_rotated + np.array([cx, cy])
    return points_global


def generate_ellipse(params, num_points=200):
    cx, cy, a, b, theta = params
    ts = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # 椭圆在局部坐标下的点
    x_local = a * np.cos(ts)
    y_local = b * np.sin(ts)

    # 旋转变换
    x_rotated = x_local * np.cos(theta) - y_local * np.sin(theta)
    y_rotated = x_local * np.sin(theta) + y_local * np.cos(theta)

    # 平移至全局坐标系
    x_global = x_rotated + cx
    y_global = y_rotated + cy

    return np.column_stack((x_global, y_global))


def generate_triangle(params, num_points=200):
    """
    等边三角形模型函数
    参数：
    params = [cx, cy, s, theta]
    cx, cy：三角形中心（重心）坐标
    s：三角形边长
    theta：整体旋转角（弧度，逆时针）
    num_points：沿三角形边界采样的总点数，默认 200
    返回：
    (N x 2) 数组，每一行为采样点的 (x, y) 坐标
    """
    cx, cy, s, theta = params
    # 计算外接圆的半径，对于等边三角形，R = s/√3
    R = s / np.sqrt(3)

    # 计算三个顶点的局部坐标（均在全局坐标中构造）
    vertices = []
    for i in range(3):
        angle = theta + 2 * np.pi * i / 3
        vertex = np.array([cx + R * np.cos(angle), cy + R * np.sin(angle)])
        vertices.append(vertex)
    vertices = np.array(vertices)

    # 沿每条边均匀采样，三条边总点数近似为 num_points
    n_edge = num_points // 3
    pts = []
    for i in range(3):
        start = vertices[i]
        end = vertices[(i + 1) % 3]
        # 每条边采样时使用 endpoint=False，防止重复采样顶点，
        # 最后可根据需要调整点数，在保证边界连续的情况下拼接
        edge_points = np.linspace(start, end, n_edge, endpoint=False)
        pts.append(edge_points)
    pts = np.concatenate(pts, axis=0)
    return pts


def generate_arbitrary_triangle(params, num_points=200):
    """
    任意三角形模型函数
    参数：
    params = [x1, y1, x2, y2, x3, y3] — 三个顶点按照边界顺序排列
    num_points: 沿三角形边界采样的总点数，默认 200 个
    返回：
    (N x 2) 数组，每一行为采样点的 (x, y) 坐标
    """
    if len(params) != 6:
        raise ValueError('参数长度错误，期望格式为 [x1, y1, x2, y2, x3, y3]')

    # 提取顶点坐标，确保顺序为边界顺序（顺时针或逆时针均可）
    vertices = np.array(
        [[params[0], params[1]], [params[2], params[3]], [params[4], params[5]]]
    )

    # 将采样点数在三条边上平均分配
    n_edge = num_points // 3
    pts_list = []

    # 对三个边分别采样：边 AB, 边 BC, 边 CA
    for i in range(3):
        start = vertices[i]
        end = vertices[(i + 1) % 3]
        # endpoint 设置为 False 以防止重复顶点被多次采样
        edge_points = np.linspace(start, end, n_edge, endpoint=False)
        pts_list.append(edge_points)

    pts = np.concatenate(pts_list, axis=0)
    return pts


def generate_rectangle(params, num_points=200):
    """
    矩形模型函数
    参数：params = [cx, cy, w, h, theta]
    cx,cy 为矩形中心坐标，
    w, h 为矩形的宽和高，
    theta 为旋转角度（弧度）。
    返回：矩形边界上均匀采样的点（沿4条边采样）
    """
    cx, cy, w, h, theta = params
    half_w = w / 2
    half_h = h / 2
    # 在局部坐标系中定义矩形四个顶点
    corners = np.array(
        [[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]]
    )

    # 旋转
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners_rot = (corners @ R.T) + np.array([cx, cy])

    # 沿每条边采样点
    pts = []
    points_per_edge = num_points // 4
    for i in range(4):
        start = corners_rot[i]
        end = corners_rot[(i + 1) % 4]
        edge_points = np.linspace(start, end, points_per_edge, endpoint=False)
        pts.append(edge_points)
    pts = np.concatenate(pts, axis=0)
    return pts


def generate_circle(params, num_points=200):
    """
    圆形模型函数
    参数：params = [cx, cy, r]
    返回：圆形边界上均匀采样的点 (num_points × 2 array)
    """
    cx, cy, r = params
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))
