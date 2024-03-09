from typing import Union, Callable, Optional
from collections import defaultdict
from .utils import SubscriDict

__all__ = ['sorted_embedded_list_dict', 'make_typetrees_dataset', 'parse_full_type', 'preview_typetree_tabularize']

def sorted_embedded_list_dict(obj: Union[dict, list], key: Optional[Callable] = None):
    '''Sorts a nested dictionary or list of dictionaries.

    Args:
        obj (dict or list):  Input dictionary or list of dictionaries
        key (callable):      Sorting key function (optional)

    Returns:
        sorted_obj(dict or list):   Sorted input object
    '''
    if isinstance(obj, dict):
        for k in obj:
            if isinstance(obj[k], (dict, list)):
                obj[k] = sorted_embedded_list_dict(obj[k], key=key)
        return dict(sorted(obj.items(), key=key))
    elif isinstance(obj, list):
        sortable = True
        for idx, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                obj[idx] = sorted_embedded_list_dict(v, key=key)
                sortable = False
        if sortable:
            return sorted(obj, key=key)
    return obj

def parse_anno_typetree(anno: Union[dict, list], exclude_domain: list = None, value_mode : Union[dict, list] = False):
    '''Parse a single annotation structure recursively and create typetree from it.

    Args:
        anno (dict/list):        Input annotation data, it could be a dictionary or a list.
        exclude_domain (list):   Exclude list of domain string

    Returns:
        anno_struct (dict/list): Parsed typetree structure.
    '''
    if isinstance(anno, dict):
        anno_struct = {}
        for k, v in anno.items():
            if exclude_domain and k in exclude_domain:
                continue
            if isinstance(v, (dict, list)):
                anno_struct[k] = parse_anno_typetree(v, value_mode=value_mode)
            else:
                anno_struct[k] = v if value_mode else type(v).__name__
    elif isinstance(anno, list):
        anno_struct = []
        for i, v in enumerate(anno):
            if isinstance(v, (dict, list)):
                anno_struct.append(parse_anno_typetree(v, value_mode=value_mode))
            else:
                anno_struct.append(v if value_mode else type(v).__name__)
    return anno_struct

def make_typetrees_dataset(annotataions : Union[dict, list], exclude_names : Optional[list] = [], typetree_mode: Optional[bool] = False, sort_tree: Optional[bool] = False):
    '''Create a typetrees dataset where each unique typetree is associated with a list of indices.

    Args:
        annotataions (dict/list): Input annotation dictionary or list
        exclude_names (list):     Exclude all keys in name list
        typetree_mode (bool):     If the input annotations are already typetree format
        sort_tree (bool):         If typetree need to be sorted

    Returns:
        typetrees_dataset (defaultdict): Dictionary with typetrees as keys and list 
                                         of indices as values.

    Examples:
        from deepdiff import DeepDiff
        typetrees_dataset = make_typetrees_dataset(anno_dict)
        diff = DeepDiff(typetrees_dataset['typetree'][0], typetrees_dataset['typetree'][1])
        [len(typetrees_dataset[i]) for i in range(len(typetrees_dataset['typetree']))]
    '''
    typetrees_dataset = defaultdict(list)
    anno_list = SubscriDict(annotataions) if isinstance(annotataions, dict) else annotataions
    for i in range(len(anno_list)):
        tmp_typetree = parse_anno_typetree(anno_list[i], exclude_domain=exclude_names, value_mode=typetree_mode)
        tmp_typetree = sorted_embedded_list_dict(tmp_typetree) if sort_tree else tmp_typetree
        is_exist = False
        for ti, tyt in enumerate(typetrees_dataset['typetree']):
            if tyt == tmp_typetree:
                typetrees_dataset[ti].append(i)
                is_exist = True
        if not is_exist:
            typetrees_dataset[len(typetrees_dataset['typetree'])].append(i)
            typetrees_dataset['typetree'].append(tmp_typetree)
    return typetrees_dataset

def parse_full_type(type_list_dict: dict):
    '''Parse full type contain in list of dictionary in typetree

    Example:
        a = [{'id': 'str', 'key': 'str', 'value': 'float', 'des': 'str'},
             {'id': 'str', 'key': 'str', 'value': 'str', 'des': 'str'}]
        parse_full_type(a)

        > defaultdict(dict,
                    {'id': {'str': 2},
                     'key': {'str': 2},
                     'value': {'float': 1, 'str': 1},
                     'des': {str: 2}})
    '''
    full_types = defaultdict(dict)
    for i in range(len(type_list_dict)):
        for key in type_list_dict[i]:
            if not full_types.__contains__(key):
                full_types[key] = {}
            if not full_types[key].__contains__(type_list_dict[i][key]):
                full_types[key][type_list_dict[i][key]] = 0
            full_types[key][type_list_dict[i][key]] += 1
    return full_types

def preview_typetree_tabularize(typetree: Optional[dict, list]):
    '''Return a preview typetree to check whether it can be tabularize
    
    Notice:
        - Only list object with same type (params:[str, str, str])
            or list object with dictionary element (params:[dict, dict, dict]) supported
    '''
    if isinstance(typetree, dict):
        preview_typetree = {}
        for k, v in typetree.items():
            if isinstance(v, (dict, list)):
                preview_typetree[k] = preview_typetree_tabularize(v)
            else:
                preview_typetree[k] = v
    elif isinstance(typetree, list):
        if isinstance(typetree[0], dict):
            preview_typetree = parse_full_type(typetree)
            for key in preview_typetree:
                if len(preview_typetree[key]) == 1:
                    preview_typetree[key] = preview_typetree[key].pop()
            preview_typetree = dict(preview_typetree)
        elif isinstance(typetree[0], list):
            preview_typetree = set(map(type, typetree))
            preview_typetree = preview_typetree.pop() if len(preview_typetree) == 1 else preview_typetree

    return preview_typetree

