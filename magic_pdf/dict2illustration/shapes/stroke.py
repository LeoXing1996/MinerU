import re
from pptx.util import Pt, Mm, Cm, Inches
from pptx.enum.dml import MSO_LINE_DASH_STYLE


def parse_stroke_width(value):
    pattern = r'([\d.]+)(px|pt|cm|mm|in|%)?'
    match = re.match(pattern, value)

    if match:
        number = int(round(float(match.group(1))))  # int(match.group(1))
        unit = match.group(2)
    else:
        raise ValueError(f'Invalid stroke-width value: {value}')

    if unit is None or unit.lower() == 'pt':
        return Pt(number) if number <= 1584 else Pt(1584)
    elif unit.lower() == 'px':
        return number
    elif unit.lower() == 'cm':
        return Cm(number)
    elif unit.lower() == 'mm':
        return Mm(number)
    elif unit.lower() == 'in':
        return Inches(number)


def parse_stroke_linecap(s):
    if s.lower() == 'butt':
        return MSO_LINE_CAP_STYLE.FLAT
    elif s.lower() == 'round':
        return MSO_LINE_CAP_STYLE.ROUND
    elif s.lower() == 'square':
        return MSO_LINE_CAP_STYLE.SQUARE
    raise ValueError(f'Unsupported line cap {s}.')


def parse_stroke_linejoin(s):
    if s.lower() == 'miter':
        return MSO_LINE_JOIN_STYLE.MIRER
    elif s.lower() == 'round':
        return MSO_LINE_JOIN_STYLE.ROUND
    elif s.lower() == 'bevel':
        return MSO_LINE_JOIN_STYLE.BEVEL
    raise ValueError(f'Unsupported line join {s}.')


def parse_stroke_miterlimit(s):
    return int(s)


def parse_stroke_dasharray(s):
    if s.lower() == 'none':
        return None

    parts = [int(part.strip()) for part in s.replace(',', ' ').split()]
    return tuple(parts)


def parse_stroke_dashoffset(s):
    return float(s)


def parse_stroke_dashstyle(s):
    # if s.lower() == "none":
    if s is None:
        return None

    # parts = [int(part.strip()) for part in s.replace(",", " ").split()]
    parts = [float(part.strip()) for part in s.replace(',', ' ').split()]

    if len(parts) == 2 and parts[0] == parts[1]:
        return MSO_LINE_DASH_STYLE.DASH

    if len(parts) == 2 and parts[0] > parts[1] and parts[0] // parts[1] <= 2:
        return MSO_LINE_DASH_STYLE.DASH
    if len(parts) == 2 and parts[0] > parts[1] and parts[0] // parts[1] >= 3:  # 4:
        return MSO_LINE_DASH_STYLE.LONG_DASH
    if len(parts) == 2 and parts[0] < parts[1]:
        return MSO_LINE_DASH_STYLE.DOT
    if len(parts) == 4 and parts[0] > parts[1] and parts[0] // parts[1] <= 2:
        return MSO_LINE_DASH_STYLE.DASH_DOT
    if len(parts) == 4 and parts[0] > parts[1] and parts[0] // parts[1] >= 4:
        return MSO_LINE_DASH_STYLE.LONG_DASH_DOT
    if len(parts) == 5 and parts[0] > parts[1] and parts[0] // parts[1] <= 2:
        return MSO_LINE_DASH_STYLE.DASH_DOT_DOT
    if len(parts) == 5 and parts[0] > parts[1] and parts[0] // parts[1] >= 4:
        return MSO_LINE_DASH_STYLE.LONG_DASH_DOT_DOT

    raise ValueError('Unsupported dash style {s}.')
