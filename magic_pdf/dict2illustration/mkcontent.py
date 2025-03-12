import re
from typing import Optional, Tuple, Union

import fitz
import pymupdf
from loguru import logger
from pymupdf import Document

from magic_pdf.config.ocr_content_type import BlockType, ContentType
from magic_pdf.data.dataset import Dataset
from magic_pdf.dict2md.ocr_mkcontent import merge_para_with_text
from magic_pdf.libs.config_reader import get_llm_aided_config
from magic_pdf.post_proc.llm_aided import llm_aided_description_search

from .pdf_to_code import image_to_code, shape_to_code, text_to_code


def filter_with_status(target: list, status: list[bool]) -> list:
    return [item for item, stat in zip(target, status) if stat]


def shape_filter(paths: list[dict]):
    """Function to filter out the invlaid shapes from `page.get_drawings()`
    function."""
    valid_paths = []
    filter_keys = [
        'fill',
        'color',
        'dashes',
        'lineJoin',
        'lineCap',
        'closePath',
        'width',
        'stroke_opacity',
    ]
    for path in paths:
        is_valid = True
        for key in filter_keys:
            if path.get(key, None) is None:
                is_valid = False
                break
        if is_valid:
            valid_paths.append(path)

    return valid_paths


def crop_pdf(
    dataset: Dataset,
    page_idx: int,
    bbox: list[int],
    image_ratio_threshold: Optional[float] = 0.1,
) -> Tuple[Optional[Document], bool, str]:
    # create a new pdf instance from raw data
    pdf = fitz.open('pdf', dataset._data_bits)
    page = pdf[page_idx]
    # bbox_pts = [coor / 2 for coor in bbox]
    bbox_pts = [coor for coor in bbox]
    page.set_cropbox(pymupdf.Rect(*bbox_pts))

    new_pdf = fitz.open()
    new_pdf.insert_pdf(pdf, from_page=page_idx, to_page=page_idx)

    # apply image / (text + shape) ratio threshold
    n_text = len([blk for blk in page.get_text('dict')['blocks'] if blk['type'] == 0])
    n_shape = len(shape_filter(page.get_drawings()))
    n_image = len(page.get_images())
    # print("n_image: ", n_image)

    if (n_text + n_shape) == 0:  # no text and shape
        image_ratio = 'ALL_IMAGE'
    else:
        image_ratio = n_image / (n_text + n_shape)

    if image_ratio_threshold is not None:
        if n_shape + n_text == 0:  # no text and shape
            return new_pdf, False, 'Image only illustration.'
        if image_ratio > image_ratio_threshold:  # too much image
            error_msg = (
                f'Image / (text + shape) ratio is too high ({n_image / (n_text + n_shape)}'
                f'> {image_ratio_threshold}).'
            )
            return new_pdf, False, error_msg
    print(f'Success, Image / (text + shape) ratio is "{image_ratio}".')
    return new_pdf, True, f'Success, Image / (text + shape) ratio is "{image_ratio}".'


def handle_caption_and_idx(para_block: dict) -> Tuple[Tuple[str, str], bool, str]:
    caption_start = False
    illus_caption = ''
    raw_caption = ''

    # 1. Get caption -> then we can have better output log
    for block in para_block['blocks']:  # 2nd.拼image_caption
        if block['type'] == BlockType.ImageCaption:
            processed_cap = merge_para_with_text(block) + '  \n'
            # NOTE: here we need to process a very special case:
            #   MinerU take captions from the previous table/figure to this block.
            #   Therefore, we need to check if the caption is started with "Figure"
            #   If found, we will start to save the caption from this block.
            if not caption_start:
                illus_idx, status, msg = get_idx_from_caption(processed_cap)
                if not status:
                    continue
                else:
                    caption_start = True
            if caption_start:
                illus_caption += processed_cap
            raw_caption += processed_cap

    if not caption_start:
        return (
            ('', None),
            False,
            f'Cannot find figure index in caption. Raw caption is started with: "{raw_caption[:15]}"',
        )

    for block in para_block['blocks']:  # 3rd.拼image_footnote
        if block['type'] == BlockType.ImageFootnote:
            illus_caption += merge_para_with_text(block) + '  \n'

    return (illus_caption.strip(), illus_idx), True, 'Success'


def union_make(
    pdf_info_list: list[dict],
    md_content: str,
    dataset: Dataset,
    parse_text: bool = True,
    parse_shape: bool = True,
    parse_image: bool = True,
    drop_failed: bool = False,
):
    """Build illustration information from extracted pdf info.

    Modified from https://github.com/opendatalab/MinerU/blob/1e4d4b596a29b78336a77d1db4ddcee8e4ba1b9a/magic_pdf/dict2md/ocr_mkcontent.py#L244

    Args:
        drop_failed: If False, error message will be added to the illustration info.
            Otherwise, the illustration will be dropped.
    """

    illus_info_list = []

    md_content_list = md_content.split('\n\n')

    # index to save image if we cannot match the figure index
    illus_match_error_counter = 0

    for page_info in pdf_info_list:
        paras_of_layout = page_info.get('para_blocks')
        page_idx = page_info.get('page_idx')

        for para_block in paras_of_layout:
            if para_block['type'] == BlockType.Image:
                # 1. Get caption -> then we can have better output log
                (illus_caption, illus_idx), cap_status, cap_msg = (
                    handle_caption_and_idx(para_block)
                )
                if drop_failed and cap_status is False:
                    continue

                # 2. loop the inner blocks and find the image body
                illus_list = []
                illus_status = []
                shape_filter_msg_list = []
                for block in para_block['blocks']:
                    if block['type'] == BlockType.ImageBody:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.Image:
                                    bbox = span['bbox']
                                    illus, status, msg = crop_pdf(
                                        dataset, page_idx, bbox
                                    )
                                    # TODO: insert a MLLM-based filter here!
                                    illus_list.append(illus)
                                    illus_status.append(status)
                                    shape_filter_msg_list.append(msg)

                if drop_failed:
                    illus_list = filter_with_status(illus_list, illus_status)
                    shape_filter_msg_list = filter_with_status(
                        shape_filter_msg_list, illus_status
                    )
                    illus_status = filter_with_status(illus_status, illus_status)

                if illus_list == []:
                    continue

                # 3. get text, shape, and image for training
                if parse_text:
                    text_code_list = [text_to_code(pdf) for pdf in illus_list]
                else:
                    text_code_list = None

                if parse_shape:
                    shape_code_list = [shape_to_code(pdf) for pdf in illus_list]
                else:
                    shape_code_list = None

                if parse_image:
                    image_code_list = [image_to_code(pdf) for pdf in illus_list]
                else:
                    image_code_list = None

                # 4. Get figure index from caption, and build name for illustration
                if illus_idx is None:
                    # use a counter to save the image if we cannot match the figure index
                    illus_idx = illus_match_error_counter
                    illus_match_error_counter += 1

                    save_name = [
                        f'page{page_idx}_Error{illus_idx}_{idx}.pdf'
                        for idx in range(len(illus_list))
                    ]
                else:
                    save_name = [
                        f'page{page_idx}_Fig{illus_idx}_{idx}.pdf'
                        for idx in range(len(illus_list))
                    ]

                # 5. Get corresponding content in paper from md_content
                illus_name = make_illus_name(illus_idx)
                related_content = []

                desp_list = None
                desp_modified_by_llm = []
                for content in md_content_list:
                    # skip the caption block, only search in the content block
                    if illus_caption.strip() in content:
                        continue
                    for name in illus_name:
                        # write a regex to match the template
                        desp_list = get_illus_description_from_content(name, content)
                        modified_by_llm = False
                        if desp_list is not None:
                            llm_aided_config = get_llm_aided_config()
                            if llm_aided_config is not None:
                                description_aided_config = llm_aided_config.get(
                                    'description_aided', None
                                )
                                if description_aided_config is not None:
                                    desp_for_illus = llm_aided_description_search(
                                        name,
                                        content,
                                        description_aided_config,
                                    )
                                    if desp_for_illus is not None:
                                        desp_list = desp_for_illus
                                        modified_by_llm = True

                            related_content += desp_list
                            desp_modified_by_llm.append(modified_by_llm)

                if not related_content:
                    content_msg = f'Cannot find match content for illustration "Figure {illus_idx}" @ (Page {page_idx}).'
                    logger.warning(content_msg)
                else:
                    content_msg = f'Found {len(related_content)} related content for illustration "Figure {illus_idx}" @ (Page {page_idx}).'
                logger.info(
                    f'"Figure {illus_idx}" (Page {page_idx}), finished processing.'
                )

                # 6. make info dict for illustration
                illus_info = {
                    'page_idx': page_idx,
                    'illus': illus_list,
                    'illus_idx': illus_idx,
                    'illus_name': save_name,
                    'text_code': text_code_list,
                    'shape_code': shape_code_list,
                    'image_code': image_code_list,
                    'caption': illus_caption.strip(),
                    'content': [content.strip() for content in related_content],
                    'content_modified_by_llm': desp_modified_by_llm,
                    # filter results
                    'caption_handle_status': cap_status,
                    'caption_handle_msg': cap_msg,
                    'shape_filter_status': illus_status,
                    'shape_filter_msg': shape_filter_msg_list,
                    'content_msg': content_msg,
                }
                illus_info_list.append(illus_info)

    return illus_info_list


def get_idx_from_caption(caption: str) -> Tuple[str, bool, str]:
    figure_pattern = (
        r'\b([Ff]igure|[Ff]ig|FIG|FIGS|Figs)\.?\s*(\d+(?:\([a-z]\))?(?:–\d+)?)'
    )

    match = re.search(figure_pattern, caption)
    fig_id_in_paper = None
    if match:
        fig_id_in_paper = match.group(2)  # match index
    else:
        return (
            None,
            False,
            f'Cannot find figure index in caption. Caption is started with "{caption[:15]}".',
        )

    return fig_id_in_paper, True, 'Success'


def make_illus_name(illus_idx: str):
    illus_idx_in_content = [
        f'Figure {illus_idx}',  # CVPR, ICCV, NIPS, ICML, SIGGRAPH
        f'Fig. {illus_idx}',  # ICLR
        f'Fig {illus_idx}',
        f'fig. {illus_idx}',
        f'fig {illus_idx}',
        f'FIG. {illus_idx}',
    ]
    return illus_idx_in_content


def get_illus_description_from_content(
    illus_name: str,
    content: str,
) -> Union[list[str], None]:
    desp_pattern = re.compile(r'[^.!?]*\b' + re.escape(illus_name) + r'\b[^.!?]*[.!?]')
    desp = desp_pattern.findall(content)
    return desp if desp else None
