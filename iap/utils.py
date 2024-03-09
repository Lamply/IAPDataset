import os
import glob
import json
import numpy as np
from typing import Union
try:
    # pip install u-msgpack-python
    import umsgpack
except:  # noqa: E722
    pass

SUPPORTED_IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP'
]

SUPPORTED_VIDEO_EXTENSIONS = [
    '.mp4', '.MP4', '.flv', '.FLV'
]

def get_file_name(file_path):
    file_name = os.path.basename(file_path)
    return file_name

def get_base_name(file_path):
    file_name = get_file_name(file_path)
    split_name = file_name.split('.')
    base_name = '.'.join(split_name[:-1]) if len(split_name) > 1 else split_name[0]
    return base_name

def get_ext_name(file_path):
    try:
        ext_name = os.path.splitext(file_path)[-1]
    except:
        ext_name = None
    return ext_name


def make_dataset(dir):
    paths = []
    for ext in SUPPORTED_IMG_EXTENSIONS:
        paths += glob.glob(os.path.join(dir, '*'+ext))
    return paths

def select_subdataset(dataset, subset_list, path2name=False):
    name_list = [get_base_name(p) for p in subset_list] if path2name else subset_list
    return [p for p in dataset if get_base_name(p) in name_list]

def select_subdataset_idxs(dataset, subset_list, path2name=False):
    name_list = [get_base_name(p) for p in subset_list] if path2name else subset_list
    return [i for i, p in enumerate(dataset) if get_base_name(p) in name_list]

def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    sequences_dir = sorted(os.listdir(dir))
    for seq in sequences_dir:
        seq_dir = os.path.join(dir, seq)
        if os.path.isdir(seq_dir) and seq[0] != '.':
            paths = sorted(make_dataset(seq_dir))
            if len(paths) > 0:
                images.append(paths)
    return images

def make_structural_dataset(root_dir, leaf_depth):
    '''Build structural dataset which images place in the leaf of directory tree.

    Args:
        root_dir(str):            root directory
        leaf_depth(int):          depth of leaf

    Outs:
        dataset(dict)
    '''
    def recursive_list(cur_dir, cur_depth=0):
        cur_depth += 1
        if cur_depth == leaf_depth:
            return sorted(make_dataset(cur_dir))
        dataset_tree = {}
        for file_name in os.listdir(cur_dir):
            file_path = os.path.join(cur_dir, file_name)
            if os.path.isdir(file_path) and file_name[0] != '.':
                dataset_tree[file_name] = recursive_list(os.path.join(cur_dir, file_name), cur_depth)
        return dataset_tree

    dataset = recursive_list(root_dir)
    return dataset


def check_if_overlap_files(paths):
    from collections import Counter
    base_names = [get_base_name(path) for path in paths]
    count_list = dict(Counter(base_names))
    return {key:value for key,value in count_list.items()if value > 1}


def check_image_format(img, allow_float=False, fix_channel=True):
    if img is None:
        raise RuntimeError("ERROR: image is empty!")
    if fix_channel:
        if len(img.shape) == 2 or len(img.shape) == 3:
            img = img.reshape([img.shape[0], img.shape[1], -1])
        else:
            raise ValueError(f"ERROR: image is not regular image! img.shape: {img.shape}")
    if img.dtype != np.uint8 and not allow_float:
        print(f"Warning: image type is not uint8 and {img.dtype} is not allow, auto-sacling into np.uint8...")
        img = img * 1.0
        img = np.uint8(255. * (img - np.min(img)) / (np.max(img) - np.min(img)))
    return img


def check_aligned_img_dataset(A_paths, B_paths):
    if len(A_paths) == 0:
        raise Exception("Dataset A is empty, please check the `dataroot` option.")
    if len(A_paths) != len(B_paths):
        raise ValueError("Different size of A=%d and B=%d " % (len(A_paths), len(B_paths)))
    for i in range(len(A_paths)):
        A_name = get_base_name(A_paths[i])
        B_name = get_base_name(B_paths[i])
        if A_name != B_name:
            raise ValueError("A and B names is not aligned: {}, {}".format(A_paths[i], B_paths[i]))

def check_grouped_img_dataset(A_paths, B_paths):
    if len(A_paths) == 0:
        raise Exception("Dataset is empty, please check the `dataroot` option.")
    if len(A_paths) != len(B_paths):
        raise ValueError("Different size of A=%d and B=%d " % (len(A_paths), len(B_paths)))
    for i in range(len(A_paths)):
        check_aligned_img_dataset(A_paths[i], B_paths[i])


def read_json_file(file, use_msgpack=False, **open_kwargs):
    with open(file, 'rb' if use_msgpack else 'r', **open_kwargs) as f:
        out = umsgpack.unpack(f) if use_msgpack else json.load(f)
    return out

def write_json_file(file, obj, override=False, use_msgpack=False, min_size=10, **dump_kwargs):
    if os.path.exists(file) and not override:
        raise RuntimeError(f"Try to override file with override={override}.")
    dump_cont = umsgpack.packb(obj) if use_msgpack else json.dumps(obj, **dump_kwargs)
    if len(dump_cont) == min_size:
        raise RuntimeError(f"Dump object size is smaller than {min_size}, which is not expected.")
    with open(file, 'wb+' if use_msgpack else 'w+') as f:
        f.write(dump_cont)

def check_dict_contents(dict1, dict2, strict=False):
    if len(dict1) != len(dict2):
        return False
    for key, value in dict1.items():
        if key not in dict2:
            return False
        if type(value) != type(dict2[key]):
            return False
        if isinstance(value, list):
            if not all(isinstance(element, type(value[0])) for element in dict2[key]):
                return False
            if strict and len(value) != len(dict2[key]):
                return False
    return True


def set_image_anno(base_name, **kwargs):
    return {"name": base_name, **kwargs}

def anno_exists(anno: dict, domain: Union[str, list], **kwargs):
    if isinstance(domain, list):
        logit = [anno_exists(anno, spec) for spec in domain]
        return np.sum(logit) == len(domain)
    if not anno.__contains__(domain):
        return False
    if isinstance(anno[domain], (dict, list)) and len(anno[domain]) == 0:
        return False
    return True

def create_base_annotation(img_path):
    anno_dict = {}
    img_files = sorted(make_dataset(img_path))
    [anno_dict.update({get_base_name(img_f): set_image_anno(get_base_name(img_f))}) for img_f in img_files]
    return anno_dict

class SubscriDict():
    def __init__(self, d):
        self.d = d
        self.d_keys = list(d.keys())

    def __getitem__(self, item):
        if isinstance(item, slice):
            return {key:self.d[key] for key in self.d_keys[item]}
        elif isinstance(item, list):
            if isinstance(item[0], str):
                return {key:self.d[key] for key in item}
            return {self.d_keys[idx]:self.d[self.d_keys[idx]] for idx in item}
        elif isinstance(item, str):
            return self.d[item]
        else:
            return self.d[self.d_keys[item]]

    def __len__(self):
        return len(self.d)

    def size(self):
        full_size = 0
        for k in self.d_keys:
            if isinstance(k, list):
                full_size += len(k)
            else:
                full_size += 1
        return full_size

