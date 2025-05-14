import re
import os
from typing import Optional, Tuple, Union

import fitz
import pymupdf
import base64
from loguru import logger
from pymupdf import Document
from openai import OpenAI

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
    image_ratio_threshold: Optional[float] = 0.2,
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
                                    # filtered by mllm

                                    ###################################
                                    if status:
                                        img_path = get_illus_images(illus)
                                        # mllm_status, mllm_msg = mllm_filter(img_path)
                                        mllm_status, mllm_msg = multi_turn_filter(
                                            img_path
                                        )
                                        print(status, mllm_status, mllm_msg)
                                        status = status and mllm_status
                                        print(status)
                                        msg = (
                                            'Rule-based: '
                                            + msg
                                            + ' VLM based: '
                                            + mllm_msg
                                        )
                                        # 删除图片
                                        os.remove(img_path)
                                    ###################################

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


def get_illus_images(doc, save_path='temp_images'):
    os.makedirs(save_path, exist_ok=True)
    # img_paths = []

    assert len(doc) == 1, 'Only one page is allowed.'
    pix = doc[0].get_pixmap(
        matrix=fitz.Identity,
        dpi=720,
        colorspace=fitz.csRGB,
        clip=None,
        alpha=False,
        annots=True,
    )

    import hashlib
    import time

    # 计算图像数据的哈希值以确保唯一性
    img_hash = hashlib.md5(pix.samples).hexdigest()[:6]
    timestamp = int(time.time())
    img_name = f'figure_page{doc[0].number}_{timestamp}_{img_hash}.jpg'

    img_path = os.path.join(save_path, img_name)
    pix.save(img_path)

    return img_path


def mllm_filter(image_path):
    prompt = """
    Please analyze the provided image to determine if it is a methodology, overview, or pipeline figure from an academic paper.
    Consider the following criteria:
    1. Does the image illustrate a step-by-step process or workflow?
    2. Is there a clear depiction of stages, components, or modules in a system or method?
    3. Does the image contain text labels, annotations, or descriptions explaining a procedure, technique, or system?
    4. Does the image have a structured design (e.g., block diagrams, flowcharts) that outlines the working principle or steps of a system or method?
    5. Does the image align with the typical context in academic papers where methodology or pipeline figures are presented?
    6. The image should not include any experimental tables, charts, or graphs.
    8. The image should not be a figure that includes texts only.

    Based on these criteria, please respond 'Yes' if the image meets the description of a methodology, overview, or pipeline figure, otherwise 'No'.
    Please respond with 'Yes' or 'No' only.
    """

    def encode_image(image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key='sk-02ec4b4ba31a406894cbe3bdb92e96a1',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    completion = client.chat.completions.create(
        model='qwen-vl-max-latest',
        messages=[
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'},
                    },
                    {'type': 'text', 'text': f'{prompt}'},
                ],
            },
        ],
    )
    # print(completion.choices[0].message.content)
    judge_responds = completion.choices[0].message.content
    if judge_responds.lower() in ['yes']:
        return True, 'Success, Image is a methodology, overview, or pipeline figure.'
    return False, 'Failed, Image is not a methodology, overview, or pipeline figure.'


def multi_turn_filter(image_path):
    def encode_image(image_path):
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Round 1: Check Exclusion Criteria
    # {"decision": "yes"/"no", "reason": "YOUR REASON"}
    prompt_round1 = """
    Please analyze if the image meets the following exclusion criteria:

    1. The image should not contain experimental tables, charts, or graphs in major part.
    2. The image should not be purely text-based figures such as an algorithms figure.

    Note: If the image violates any of the above criteria, please respond 'no'.
    Please respond in the following format:
    {"decision": "yes"/"no"}
    """

    # Round 2: Check Content Features
    prompt_round2 = """
    If the first round check passed, please analyze the following content features:
    1. Does the image demonstrate a step-by-step process or workflow?
    2. Does it contain text labels, annotations, or descriptions explaining procedures/techniques/systems?
    3. Is there a clear logical connection and sequence between components?

    Please respond based only on the above criteria in the following format:
    {"decision": "yes"/"no"}
    """

    # Round 3: Check the basic structure of the image
    prompt_round3 = """
    If the second round check passed, please analyze the basic structure of the image:
    1. Can you infer the methodology or pipeline from the image?
    2. Does the overall layout match the typical characteristics of methodology figures in academic papers?

    Please respond based only on the above criteria in the following format:
    {"decision": "yes"/"no"}
    """

    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key='sk-02ec4b4ba31a406894cbe3bdb92e96a1',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    )

    messages = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}],
        }
    ]

    # Execute three rounds of dialogue
    prompts = [prompt_round1, prompt_round2, prompt_round3]
    final_decision = True

    for round_num, prompt in enumerate(prompts, 1):
        if round_num == 1:
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{base64_image}'
                            },
                        },
                        {'type': 'text', 'text': prompt},
                    ],
                }
            )
        else:
            messages.append(
                {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}
            )

        completion = client.chat.completions.create(
            model='qwen-vl-max-2025-04-02',
            messages=messages,
            temperature=0.7,
            top_p=0.8,
        )

        response = completion.choices[0].message
        print(f'Round {round_num} response:', response.content)
        messages.append(response.model_dump())

        # regular expression to parse the response
        try:
            decision_match = re.search(r'"decision":\s*"(\w+)"', response.content)
            if decision_match and decision_match.group(1).lower() == 'no':
                final_decision = False
                break
        except Exception as e:
            print(f'Warning: Error parsing response: {e}')
            # 如果解析失败，回退到简单的字符串匹配
            if '"decision": "no"' in response.content.lower():
                final_decision = False
                break

    if final_decision:
        return True, 'Success, Image is a methodology, overview, or pipeline figure.'
    return False, 'Failed, Image is not a methodology, overview, or pipeline figure.'
