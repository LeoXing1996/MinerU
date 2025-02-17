import copy
import json
import os
from typing import Callable

from magic_pdf.config.make_content_config import DropMode, MakeMode
from magic_pdf.data.data_reader_writer import DataWriter
from magic_pdf.data.dataset import Dataset
from magic_pdf.dict2illustration import union_make as union_make_illus
from magic_pdf.dict2md.ocr_mkcontent import union_make
from magic_pdf.libs.draw_bbox import (
    draw_layout_bbox,
    draw_line_sort_bbox,
    draw_span_bbox,
)
from magic_pdf.libs.json_compressor import JsonCompressor


class PipeResult:
    def __init__(self, pipe_res, dataset: Dataset):
        """Initialized.

        Args:
            pipe_res (list[dict]): the pipeline processed result of model inference result
            dataset (Dataset): the dataset associated with pipe_res
        """
        self._pipe_res = pipe_res
        self._dataset = dataset

    def get_illustration_info(self, md_content: str) -> dict:
        pdf_info_list = self._pipe_res['pdf_info']
        illustration_info_list = union_make_illus(
            pdf_info_list,
            md_content,
            self._dataset,
            drop_failed=True,
        )
        illustration_info = {
            'illustration_info': illustration_info_list,
            '_version_name': self._pipe_res['_version_name'],
            '_parse_type': self._pipe_res['_parse_type'],
        }
        return illustration_info

    def dump_illustration_info(
        self,
        writer: DataWriter,
        file_path: str,
        md_content: str,
        save_image: bool = True,
        save_pdf: bool = False,
        save_svg: bool = False,
        pdf_file_path: str = 'illustration',
    ):
        """Dump the illustration content used for training."""
        illus_info = self.get_illustration_info(md_content)
        illus_info_list = illus_info['illustration_info']

        for one_illus_info in illus_info_list:
            if save_pdf:
                one_illus_info['illus_save_path'] = []
                for illus, name in zip(
                    one_illus_info['illus'], one_illus_info['illus_name']
                ):
                    save_path = os.path.join(writer._parent_dir, pdf_file_path, name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    illus.save(save_path)
                    one_illus_info['illus_save_path'].append(save_path)
            # remove illus instance
            one_illus_info.pop('illus')

            if save_svg:
                one_illus_info['svg_save_path'] = []
                for illus, name in zip(
                    one_illus_info['shape_code'], one_illus_info['illus_name']
                ):
                    svg = illus.pop('svg')
                    save_path = os.path.join(
                        writer._parent_dir, 'images', name.replace('.pdf', '.svg')
                    )
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, 'w') as f:
                        f.write(svg)
                    one_illus_info['svg_save_path'].append(save_path)

            image_info_list = one_illus_info['image_code']
            for image_info, name in zip(image_info_list, one_illus_info['illus_name']):
                for idx, sub_image_info in enumerate(image_info):
                    if save_image:
                        image_save_path = os.path.join(
                            writer._parent_dir,
                            'images',
                            name.replace('.pdf', f'_{idx}.png'),
                        )
                        os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                        sub_image_info.pop('image').save(image_save_path)
                        sub_image_info['image'] = image_save_path
                    else:
                        # do not save, directly drop the image
                        sub_image_info.pop('image')

        writer.write_string(
            file_path, json.dumps(illus_info, ensure_ascii=False, indent=4)
        )

    def get_markdown(
        self,
        img_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ) -> str:
        """Get markdown content.

        Args:
            img_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
            md_make_mode (str, optional): The content Type of Markdown be made. Defaults to MakeMode.MM_MD.

        Returns:
            str: return markdown content
        """
        pdf_info_list = self._pipe_res['pdf_info']
        md_content = union_make(
            pdf_info_list, md_make_mode, drop_mode, img_dir_or_bucket_prefix
        )
        return md_content

    def dump_md(
        self,
        writer: DataWriter,
        file_path: str,
        img_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
        md_make_mode=MakeMode.MM_MD,
    ) -> str:
        """Dump The Markdown.

        Args:
            writer (DataWriter): File writer handle
            file_path (str): The file location of markdown
            img_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
            md_make_mode (str, optional): The content Type of Markdown be made. Defaults to MakeMode.MM_MD.
        """

        md_content = self.get_markdown(
            img_dir_or_bucket_prefix, drop_mode=drop_mode, md_make_mode=md_make_mode
        )
        writer.write_string(file_path, md_content)
        return md_content

    def get_content_list(
        self,
        image_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
    ) -> str:
        """Get Content List.

        Args:
            image_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.

        Returns:
            str: content list content
        """
        pdf_info_list = self._pipe_res['pdf_info']
        content_list = union_make(
            pdf_info_list,
            MakeMode.STANDARD_FORMAT,
            drop_mode,
            image_dir_or_bucket_prefix,
        )
        return content_list

    def dump_content_list(
        self,
        writer: DataWriter,
        file_path: str,
        image_dir_or_bucket_prefix: str,
        drop_mode=DropMode.NONE,
    ):
        """Dump Content List.

        Args:
            writer (DataWriter): File writer handle
            file_path (str): The file location of content list
            image_dir_or_bucket_prefix (str): The s3 bucket prefix or local file directory which used to store the figure
            drop_mode (str, optional): Drop strategy when some page which is corrupted or inappropriate. Defaults to DropMode.NONE.
        """
        content_list = self.get_content_list(
            image_dir_or_bucket_prefix,
            drop_mode=drop_mode,
        )
        writer.write_string(
            file_path, json.dumps(content_list, ensure_ascii=False, indent=4)
        )

    def get_middle_json(self) -> str:
        """Get middle json.

        Returns:
            str: The content of middle json
        """
        return json.dumps(self._pipe_res, ensure_ascii=False, indent=4)

    def dump_middle_json(self, writer: DataWriter, file_path: str):
        """Dump the result of pipeline.

        Args:
            writer (DataWriter): File writer handler
            file_path (str): The file location of middle json
        """
        middle_json = self.get_middle_json()
        writer.write_string(file_path, middle_json)

    def draw_layout(self, file_path: str) -> None:
        """Draw the layout.

        Args:
            file_path (str): The file location of layout result file
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_layout_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def draw_span(self, file_path: str):
        """Draw the Span.

        Args:
            file_path (str): The file location of span result file
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_span_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def draw_line_sort(self, file_path: str):
        """Draw line sort.

        Args:
            file_path (str): The file location of line sort result file
        """
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        pdf_info = self._pipe_res['pdf_info']
        draw_line_sort_bbox(pdf_info, self._dataset.data_bits(), dir_name, base_name)

    def get_compress_pdf_mid_data(self):
        """Compress the pipeline result.

        Returns:
            str: compress the pipeline result and return
        """
        return JsonCompressor.compress_json(self._pipe_res)

    def apply(self, proc: Callable, *args, **kwargs):
        """Apply callable method which.

        Args:
            proc (Callable): invoke proc as follows:
                proc(pipeline_result, *args, **kwargs)

        Returns:
            Any: return the result generated by proc
        """
        return proc(copy.deepcopy(self._pipe_res), *args, **kwargs)
