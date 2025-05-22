import json
import os
from magic_pdf.dict2illustration.code_to_pptx import code_to_pptx


test_illus_info = '/cpfs01/shared/llm_dev/maqianli/dataset/output_parallel_processing_onlysvg/0325newpapers_split_1/cs.CV_2503.11205v1/cs.CV_2503.11205v1/auto/cs.CV_2503.11205v1_illustration_info.json'
test_svg_path = '/cpfs01/shared/llm_dev/maqianli/dataset/output_parallel_processing_onlysvg/0325newpapers_split_1/cs.CV_2503.11205v1/cs.CV_2503.11205v1/auto/images/page3_Fig3_0.svg'

paper_name = test_illus_info.split('/')[-1].split('_')[0]
paper_name = test_illus_info.split('/')[-3]


def test_code_to_pptx():
    with open(test_illus_info, 'r') as f:
        illus_info_list = json.load(f)['illustration_info']

    for idx, illus_info in enumerate(illus_info_list):
        for sub_idx in range(len(illus_info['illus_name'])):
            # if os.path.basename(test_svg_path) != os.path.basename(illus_info["svg_save_path"][0]):
            #     continue

            code_to_pptx(
                text_info_list=illus_info['text_code'][sub_idx],
                shape_info_list=test_svg_path,
                image_info_list=illus_info['image_code'][sub_idx],
                save_path=f'test/test_pdf_pptx/{paper_name}_{idx}_{sub_idx}.pptx',
            )
            print(f'Processed test/test_pdf_pptx/{paper_name}_{idx}_{sub_idx}.pptx')


def batch_code_to_pptx(test_illus_info_dir):
    os.makedirs(os.path.join(test_illus_info_dir, 'test_pdf_pptx'), exist_ok=True)

    for root, dirs, files in os.walk(test_illus_info_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    illus_info_list = json.load(f)['illustration_info']

                for idx, illus_info in enumerate(illus_info_list):
                    code_to_pptx(
                        text_info_list=illus_info['text_code'],
                        shape_info_list=illus_info['shape_code'],
                        image_info_list=illus_info['image_code'],
                        svg_path_file=os.path.join(
                            root, os.path.basename(illus_info['path_code'])
                        ),
                        save_path=f'{test_illus_info_dir}/test_pdf_pptx/{os.path.basename(root)}_{idx}.pptx',
                    )
                    print(
                        f'Processed {os.path.dirname(root)}/test_pdf_pptx/{os.path.basename(root)}_{idx}.pptx'
                    )


def cscv_code_to_pptx(test_illus_info_json):
    test_illus_info_dir = os.path.dirname(test_illus_info_json)

    with open(test_illus_info_json, 'r') as f:
        paper_info_list = json.load(f)

    for paper_name, paper_item in paper_info_list.items():
        with open(paper_item['illustration_dir'], 'r') as f:
            illus_info_list = json.load(f)['illustration_info']

        for idx, illus_info in enumerate(illus_info_list):
            if os.path.exists(
                f'{test_illus_info_dir}/test_pdf_pptx/{paper_name}_{idx}.pptx'
            ):
                continue
            code_to_pptx(
                text_info_list=illus_info['text_code'][0]
                if isinstance(illus_info['text_code'], list)
                else illus_info['text_code'],
                shape_info_list=illus_info['svg_save_path'][0]
                if isinstance(illus_info['svg_save_path'], list)
                else illus_info['svg_save_path'],
                image_info_list=illus_info['image_code'][0]
                if isinstance(illus_info['image_code'], list)
                else illus_info['image_code'],
                svg_path_file=None,
                save_path=f'{test_illus_info_dir}/test_pdf_pptx/{paper_name}_{idx}.pptx',
            )
            print(
                f'Processed {os.path.dirname(test_illus_info_dir)}/test_pdf_pptx/{paper_name}_{idx}.pptx'
            )


if __name__ == '__main__':
    cscv_code_to_pptx(
        '/cpfs01/shared/llm_dev/maqianli/dataset/output_parallel_processing_onlysvg/cscv4refine/dataset_info_with_svg_count_filtered_threshold_threshold_1000.json'
    )
    # test_code_to_pptx()
    # batch_code_to_pptx("/cpfs01/user/maqianli/LLM4Pipeline/processed_illus_drop1")
    # batch_code_to_pptx("/cpfs01/user/maqianli/LLM4Pipeline/test_json_files_0425")
