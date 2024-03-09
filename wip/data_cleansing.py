import copy
import pandas as pd


def check_same_type(elements):
    return len(set(map(type, elements))) == 1


def sort_float_key(data):
    key = data
    if isinstance(data, tuple):
        try:
            key = float(data[0])
        except:  # noqa: E722
            key = float("inf")
    elif isinstance(data, str):
        try:
            key = float(data)
        except:  # noqa: E722
            key = float("inf")
    return key


def replace_empty_leaves(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                replace_empty_leaves(v)
            elif v in ["", {}, [], ()]:
                obj[k] = None
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                replace_empty_leaves(v)
            elif v in ["", {}, [], ()]:
                obj[i] = None


def reformat_dict_to_pandas(raw_anno_dict):
    anno_dict = copy.deepcopy(raw_anno_dict)
    replace_empty_leaves(anno_dict)

    # NAN will appear if some columns is mix with number and None
    df = pd.DataFrame.from_dict(anno_dict, orient="index").reset_index(drop=True)
    return df
