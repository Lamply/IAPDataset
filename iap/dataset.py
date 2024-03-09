# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import cv2
import os
import shutil
import copy
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from .logger import setup_logger
from .utils import (
    get_base_name,
    make_dataset,
    read_json_file,
    write_json_file,
    anno_exists,
    create_base_annotation,
)
from .version import parse_version
from yacs.config import CfgNode as CN
import traceback

__all__ = ["IAPDataset"]

logger = setup_logger()


class IAPDataset:
    """Dataset for Image and Annotation Pairs."""

    __version__ = "lamply-1.5"

    def __init__(self, meta_file):
        """Open dataset via meta_file.

        Args:
            meta_file (str):    meta file path
        """
        self.dataset_root = os.path.dirname(os.path.abspath(meta_file))
        self.meta_file = os.path.abspath(meta_file)
        self.current_idx = 0
        self.current_name = ""
        self.current_data = None
        self.img_path = None
        self.img_files = None
        self.name_list = None
        self.anno_file = None
        self.anno_dict = None
        self.metadata = CN()
        with open(meta_file) as f:
            self.metadata = self.metadata.load_cfg(f)
        self._set_dataset()

    @staticmethod
    def create_dataset(
        dataset_root,
        parser=None,
        anno_file="auto_annotations.json",
        img_dir="images",
        meta_file="metadata.yml",
        continue_flag=False,
        from_empty=False,
        **kwargs,
    ):
        """Create dataset in dataset_root.

        Notice:
            - Related file name is required.

        Args:
            dataset_root (str):            root directory of dataset
            anno_file (str):               data annotation filename
        """
        if "/" in img_dir or "/" in meta_file or "/" in anno_file:
            raise ValueError(
                f"{img_dir}, {meta_file}, {anno_file}: leaf name is required"
            )
        img_path = os.path.join(dataset_root, img_dir)
        metafile_path = os.path.join(dataset_root, meta_file)
        anno_path = os.path.join(dataset_root, anno_file)
        assert not os.path.exists(metafile_path)
        if from_empty:
            os.makedirs(img_path)
        if parser:
            logger.info(
                f"create dataset with auto-parsing annotation into {anno_path}."
            )
            parser.parse(img_path, out_json_file=anno_path, continue_flag=continue_flag)
            anno_dict = parser.out_json
        else:
            if os.path.exists(anno_path):
                logger.warning(
                    f"create dataset with annotation {anno_path} while parser version is unknown, you should set it manually."
                )
                anno_dict = read_json_file(anno_path)
            else:
                logger.info(
                    f"create dataset with base name annotation into {anno_path}."
                )
                anno_dict = create_base_annotation(img_path)
                write_json_file(anno_path, anno_dict)

        metadata = OrderedDict()
        metadata["name"] = os.path.abspath(dataset_root).split("/")[-1]
        metadata["img_path"] = img_dir
        metadata["anno_file"] = anno_file
        metadata["size"] = len(make_dataset(img_path))
        metadata["from_dataset"] = kwargs.get("from_dataset", [])
        metadata["from_processors"] = kwargs.get("from_processors", [])
        metadata["parser_version"] = kwargs.get(
            "parser_version", parser.__version__ if parser else None
        )
        metadata["dataset_version"] = kwargs.get(
            "dataset_version", IAPDataset.__version__
        )
        metadata = CN(metadata)
        with open(metafile_path, mode="w+") as f:
            metadata.dump(stream=f, sort_keys=False)

    @classmethod
    def merge_dataset(
        cls,
        dataset_root,
        sub_datasets,
        anno_file="auto_annotations.json",
        img_dir="images",
        meta_file="metadata.yml",
    ):
        """Merge some of sub-datasets to dataset_root.

        Notice:
            All sub-datasets should go through the same parsers and processers.

        Args:
            dataset_root (str):            root directory of dataset
            sub_datasets (list):           list of IAPDataset to be merged
        """
        # Check if all sub_datasets have the same source
        assert all(
            [
                iap.metadata["from_processors"]
                == sub_datasets[0].metadata["from_processors"]
                for iap in sub_datasets
            ]
        )
        assert all(
            [
                iap.metadata["parser_version"]
                == sub_datasets[0].metadata["parser_version"]
                for iap in sub_datasets
            ]
        )

        # 1- create new empty dataset
        img_path = os.path.join(dataset_root, img_dir)
        anno_path = os.path.join(dataset_root, anno_file)
        os.makedirs(img_path)

        # 2- move data to new dataset
        _ = [
            shutil.move(p, img_path)
            for dataset in sub_datasets
            for p in dataset.img_files
        ]
        anno_dict = {}
        [anno_dict.update(dataset.anno_dict) for dataset in sub_datasets]
        anno_dict_ordered = OrderedDict(sorted(anno_dict.items(), key=lambda kv: kv[0]))
        write_json_file(anno_path, anno_dict_ordered)

        # 3- set meta message
        meta_message = {}
        meta_message["from_dataset"] = [
            dataset.metadata["from_dataset"] + [dataset.metadata["name"]]
            for dataset in sub_datasets
        ]
        meta_message["from_processors"] = sub_datasets[0].metadata["from_processors"]
        meta_message["parser_version"] = sub_datasets[0].metadata["parser_version"]
        meta_message["dataset_version"] = cls.__version__
        cls.create_dataset(
            dataset_root,
            anno_file=anno_file,
            img_dir=img_dir,
            meta_file=meta_file,
            **meta_message,
        )

    def _check_datasize(self):
        if self.metadata["size"] != len(self.anno_dict):
            logger.info(
                "Annotation files size (%d) is not equal to dataset size (%d)"
                % (len(self.anno_dict), self.metadata["size"])
            )
        if self.metadata["size"] != len(self.img_files):
            logger.info(
                "Image files size (%d) is not equal to dataset size (%d)"
                % (len(self.img_files), self.metadata["size"])
            )

    def _check_name_idx_align(self):
        name_list = list(self.anno_dict.keys())
        if name_list != self.name_list:
            logger.info("Annotation key is not align with images name.")

    def _set_dataset(self):
        parser_v = parse_version(self.metadata["parser_version"])
        dataset_v = parse_version(self.metadata["dataset_version"])
        logger.info("Loading dataset version {dataset_v}, parser version {parser_v}")
        assert (
            parser_v.domain == "lamply"
        )  # parser and processor heavily rely on anno's structure

        if dataset_v == parse_version("lamply-1.0"):
            self.metadata["img_path"] = self.metadata["data_path"]
        if dataset_v < parse_version("lamply-1.2"):
            logger.info("Detected deprecated dataset, update from messages...")
            self.metadata["from_dataset"] = (
                []
                if self.metadata["from_dataset"] is None
                else [self.metadata["from_dataset"]]
            )
            self.metadata["from_processors"] = (
                []
                if self.metadata["from_processors"] is None
                else self.metadata["from_processors"]
            )
            self.metadata["dataset_version"] = self.__version__

        self.img_path = os.path.join(self.dataset_root, self.metadata["img_path"])
        self.anno_file = os.path.join(self.dataset_root, self.metadata["anno_file"])
        if os.path.isdir(self.img_path):
            self.img_files = sorted(make_dataset(self.img_path))
        else:
            raise RuntimeError("Data path is not exists, %s" % self.img_path)
        self.anno_dict = read_json_file(self.anno_file)
        self.name_list = [get_base_name(p) for p in self.img_files]

        if parser_v < parse_version("lamply-1.3"):
            logger.info("Detected deprecated annotations, update name...")
            for key in self.anno_dict:
                del self.anno_dict[key]["image_path"]
                self.anno_dict[key] = {"name": key, **self.anno_dict[key]}
            parser_v.version_num = float(1.3)
            self.metadata["parser_version"] = f"{parser_v}"

        self._check_datasize()
        self._check_name_idx_align()

    def get_idx(self, name):
        return self.name_list.index(name)

    def get_idx_list(self, name_list):
        return [self.get_idx(name) for name in name_list]

    def get_name(self, idx):
        return self.name_list[idx]

    def load_data(self, idx=0, name=None, rgb_out=False, anno_only=False):
        idx = idx if name is None else self.get_idx(name)
        if name is None:
            self.current_idx = idx
            self.current_name = self.name_list[idx]
        else:
            self.current_idx = idx
            self.current_name = name
        if anno_only:
            self.current_data = None, self.anno_dict[self.current_name]
        else:
            if rgb_out:
                self.current_data = (
                    np.array(Image.open(self.img_files[idx])),
                    self.anno_dict[self.current_name],
                )
            else:
                # IMPROVE: cv2.IMREAD_COLOR is not a good idea, because the alpha channel will be missing
                self.current_data = (
                    cv2.imread(self.img_files[idx], cv2.IMREAD_COLOR),
                    self.anno_dict[self.current_name],
                )
        return self.current_data

    def load_random_data(self):
        idx = np.random.randint(0, self.metadata["size"], 1)[0]
        return self.load_data(idx)

    def update_state(self, dump=True):
        self.img_files = sorted(make_dataset(self.img_path))
        self.metadata["size"] = len(self.img_files)
        self.name_list = [get_base_name(p) for p in self.img_files]
        anno_dict_ordered = OrderedDict(
            sorted(self.anno_dict.items(), key=lambda kv: kv[0])
        )
        self._check_datasize()
        if dump:
            write_json_file(self.anno_file, anno_dict_ordered, override=True)
            with open(self.meta_file, mode="w+") as f:
                self.metadata.dump(stream=f, sort_keys=False)

    # Runing slowly if saving lots of data
    def add_data(self, data, state_update=True):
        base_name = data[1]["name"]
        cv2.imwrite(os.path.join(self.img_path, base_name + ".png"), data[0])
        self.anno_dict[base_name] = data[1]
        if state_update:
            self.update_state()

    def delete_data(self, idx=0, name=None):
        if name is None:
            name = self.get_name(idx)
        else:
            idx = self.get_idx(name)
        os.remove(self.img_files[idx])
        if anno_exists(self.anno_dict[name], "masks"):
            [
                os.remove(mask_path)
                for mask_path in self.anno_dict[name]["masks"].values()
            ]
        del self.img_files[idx]
        del self.anno_dict[name]
        self.update_state()

    def select_data(self, processors, idx_list=[], debug=False):
        """Select data via processors, only annotations are used.

        Args:
            processors (iterable):        list of processor
            idx_list (iterable):          list of data for processing, all data will be processing by default
            debug (bool):                 log option for debugging
        """
        clean_i = []
        clean_key = []
        process_id = idx_list if len(idx_list) > 0 else range(self.metadata["size"])
        for i in process_id:
            key = self.get_name(i)
            processed_data = copy.deepcopy(self.load_data(i, anno_only=True))
            try:
                for processor in processors:
                    if isinstance(processed_data, list):
                        processed_data = [
                            processor(p_data) for p_data in processed_data
                        ]
                    elif isinstance(processed_data, tuple):
                        processed_data = processor(processed_data)
            except Exception:
                if debug:
                    logger.warning(i, traceback.print_exc())
                continue
            if isinstance(processed_data, list):
                for p_data in processed_data:
                    if isinstance(p_data, tuple):
                        clean_i.append(i)
                        clean_key.append(key)
                        break
            elif isinstance(processed_data, tuple):
                clean_i.append(i)
                clean_key.append(key)
        return clean_i, clean_key

    def select_outlier_data(self, idx_list_before, idx_list_after):
        outlier_i = []
        for idx in idx_list_before:
            if idx not in idx_list_after:
                outlier_i.append(idx)
        return outlier_i

    def pipeline(
        self,
        dataset_root,
        processors,
        idx_list=[],
        anno_file="auto_annotations.json",
        img_dir="images",
        meta_file="metadata.yml",
    ):
        """Processing dataset to next generation.

        Args:
            name (str):                   name of next generation dataset
            processors (iterable):        list of processor
            idx_list (iterable):          list of data for processing, all data will be processing by default
        """
        # 1- Create next generation dataset meta-message and files root
        new_config = {}
        new_config["from_dataset"] = self.metadata["from_dataset"] + [
            self.metadata["name"]
        ]
        new_config["from_processors"] = [
            processor.dump_config() for processor in processors
        ]
        new_config["parser_version"] = self.metadata["parser_version"]
        new_config["dataset_version"] = self.__version__
        self.create_dataset(
            dataset_root,
            anno_file=anno_file,
            img_dir=img_dir,
            meta_file=meta_file,
            from_empty=True,
            **new_config,
        )
        meta_file = os.path.join(dataset_root, meta_file)
        nextgen = IAPDataset(meta_file)

        # 2- Process all data
        process_id = idx_list if len(idx_list) > 0 else range(self.metadata["size"])
        for i in tqdm(process_id):
            processed_data = copy.deepcopy(self.load_data(i))
            try:
                for processor in processors:
                    if isinstance(processed_data, list):
                        processed_data = [
                            processor(p_data)
                            for p_data in processed_data
                            if isinstance(p_data, tuple)
                        ]
                    elif isinstance(processed_data, tuple):
                        processed_data = processor(processed_data)
                    else:
                        break
            except Exception as e:
                logger.warning(self.current_idx, repr(e))
                logger.warning(traceback.print_exc())
                processed_data = None
            if isinstance(processed_data, list):
                [
                    nextgen.add_data(p_data, state_update=False)
                    for p_data in processed_data
                    if isinstance(p_data, tuple)
                ]
            elif isinstance(processed_data, tuple):
                nextgen.add_data(processed_data, state_update=False)
        nextgen.update_state()

    def display_data(self, display_func, idx=None, name=None, disable_domain=[]):
        """Display data using display_func function.

        Args:
            display_func (callable):          see more in cvglue.render_lamply
            idx (int, optional):              setup data by index. Defaults to None.
            name (str, optional):             setup data by name. Defaults to None.
            disable_domain (list, optional):  ignore some domian. Defaults to [].

        Returns:
            array: data image with annotations drawing.
        """
        if not isinstance(disable_domain, list):
            raise RuntimeError(
                "disable_domain should be a list with domain: ['landmarks', 'genderage', 'headpose']"
            )
        if idx:
            self.load_data(idx)
        elif name:
            self.load_data(self.get_idx(name))

        img_disp = display_func(self.current_data, disable_domain=disable_domain)
        return img_disp

    def __len__(self):
        return self.metadata["size"]
