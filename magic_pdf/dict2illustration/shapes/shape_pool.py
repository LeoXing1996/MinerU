from . import *
from .util import count_edges_and_curves, judge_arrow_base


class ShapePool:
    def __init__(self):
        self.pool = [
            OvalShape,
            IsoscelesTriangleShape,
            RightTriangleShape,
            RectangleShape,
            RoundedRectangleShape,
            SnipSingleCornerRectangleShape,
            SnipSameSideCornerRectangleShape,
            SnipDiagonalCornerRectangleShape,
            SnipAndRoundSingleCornerRectangleShape,
            ParallelogramShape,
            TrapezoidShape,
        ]

    def get_candidates(self, svg_path):
        # if a rectangle and a non-rectangle, seem to be an arrow
        if judge_arrow_base(svg_path):
            return []

        cnt = count_edges_and_curves(svg_path)
        num_edges = cnt['num_edges']
        num_curves = cnt['num_curves']
        print(f'Target SVG path has {num_edges} edges and {num_curves} curves.')

        candidates = []
        for shape in self.pool:
            if num_edges != shape.NUM_EDGES:
                continue
            if num_curves < shape.NUM_CURVES:
                continue
            candidates.append(shape)

        print(f'{len(candidates)} shapes are matched.')

        # if num_edges == 8 and num_curves == 0:
        #     import ipdb; ipdb.set_trace()
        #     edges = parse_path(svg_path)
        #     start_point = edges[0].start
        #     end_point = edges[-1].end

        #     for idx, edg in enumerate(edges):
        #         if edg.end == start_point:
        #             sub_shape1 = edges[:idx+1]
        #             sub_shape2 = edges[idx+1:]

        #             is_arrow_base = judge_arrow_base(sub_shape1, sub_shape2)
        #             if is_arrow_base:
        #                 return []

        return candidates
