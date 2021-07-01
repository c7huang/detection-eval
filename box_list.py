import warnings
from itertools import compress
import numpy as np

__all__ = ['BoxInfo', 'BoxList', 'combine_box_lists']

class BoxInfo():
    def __init__(self, box_list, idx):
        self.box_list = box_list
        self.idx = idx
    
    @staticmethod
    def keys():
        return [
            'ignored', 'bg',
            'localized', 'loc_score',
            'classified', 'gt_label',
            'pred_label', 'pred_score',
            'matched_idx', 'data'
        ]

    def __getattr__(self, name):
        if name == 'box_list' or name == 'idx':
            raise AttributeError
        attr = getattr(self.box_list, name, None)
        if isinstance(attr, (list, np.ndarray)):
            return attr[self.idx]
        attr = getattr(self.box_list, name+'s', None)
        if isinstance(attr, (list, np.ndarray)):
            return attr[self.idx]
        raise AttributeError
        
    def __setattr__(self, name, value):
        if name == 'box_list' or name == 'idx':
            super().__setattr__(name, value)
            return
        attr = getattr(self.box_list, name, None)
        if isinstance(attr, (list, np.ndarray)):
            attr[self.idx] = value
            return
        attr = getattr(self.box_list, name+'s', None)
        if isinstance(attr, (list, np.ndarray)):
            attr[self.idx] = value
            return
    
    def __repr__(self):
        return repr(
            {attr_name: getattr(self, attr_name) for attr_name in self.keys() if attr_name != 'data'}
        )


class BoxList():
    def __init__(self, n_boxes=0):
        self.n_boxes = n_boxes

        self.ignored = np.zeros(n_boxes, dtype=bool)

        # This is an indicator list that determines if the box
        # does not match any GT/pred
        self.bg = np.zeros(n_boxes, dtype=bool)

        # This is an indicator list that determines if the box is
        #   * Correctly localized
        #   * Mislocalized
        self.localized = np.zeros(n_boxes, dtype=bool)
        self.loc_scores = np.full(n_boxes, np.nan, dtype=float)

        # This is an indicator list that determines if the box is
        #   * Correctly classified
        #   * Misclassified
        self.classified = np.zeros(n_boxes, dtype=bool)
        self.gt_labels = np.full(n_boxes, -1, dtype=int)
        self.pred_labels = np.full(n_boxes, -1, dtype=int)
        self.pred_scores = np.full(n_boxes, np.nan, dtype=float)

        # If a box is TP, then matched_idx will provide the index of the matching boxes
        self.matched_idx = np.full(n_boxes, -1, dtype=int)

        # A gt_list will be paired with a corresponding pred_list, and vice versa,
        # this is used to keep track of the indices of matching TP boxes
        self.paired_list = None

        # self.data = np.full(n_boxes, None, dtype=object)
        self.data = [None] * n_boxes
    
    @staticmethod
    def keys():
        return [
            'ignored', 'bg',
            'localized', 'loc_scores',
            'classified', 'gt_labels',
            'pred_labels', 'pred_scores',
            'matched_idx', 'data'
        ]
    
    def pair_with(self, paired_list):
        self.paired_list = paired_list
        paired_list.paired_list = self
    
    @property
    def valid(self):
        return ~self.ignored
    
    @property
    def localised(self):
        return self.localized
    
    @property
    def mislocalized(self):
        return ~self.localized

    @property
    def mislocalised(self):
        return ~self.localized
    
    @property
    def misclassified(self):
        return ~self.classified

    def __add__(self, other):
        ret = BoxList(self.n_boxes + other.n_boxes)
        for attr_name in self.keys():
            ret_attr = getattr(ret, attr_name)
            self_attr = getattr(self, attr_name)
            other_attr = getattr(other, attr_name)
            if not (type(ret_attr) == type(self_attr) == type(other_attr)):
                raise TypeError(f'Type mismatch for attribute `{attr_name}`. Expected `{type(ret_attr)}`, got `{type(self_attr)}` and `{type(other_attr)}`')

            if attr_name == 'matched_idx':
                if self.paired_list is not None:
                    other_attr += len(self.paired_list)
                else:
                    warnings.warn('No paired BoxList defined, thus `matched_idx` cannot be properly combined.')

            if isinstance(ret_attr, list):
                combined_attr = self_attr + other_attr
                setattr(ret, attr_name, combined_attr)
            elif isinstance(ret_attr, np.ndarray):
                ret_attr[:self.n_boxes] = self_attr
                ret_attr[self.n_boxes:] = other_attr
            else:
                raise TypeError(f'The attribute `{attr_name}` is corrupted.')
        return ret
    
    def __len__(self):
        return self.n_boxes

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self):
                raise IndexError
            return BoxInfo(self, idx)
        elif isinstance(idx, (list, slice, np.ndarray)):
            if isinstance(idx, np.ndarray) and (len(idx.shape) != 1):
                raise IndexError
            is_binary = isinstance(idx, np.ndarray) and idx.dtype == bool
            if is_binary and idx.shape[0] != self.n_boxes:
                raise IndexError
            n_boxes = len(self.ignored[idx])
            ret = BoxList(n_boxes)
            for attr_name in self.keys():
                attr = getattr(self, attr_name)
                if isinstance(attr, np.ndarray) or isinstance(idx, slice):
                    new_attr = attr[idx]
                elif hasattr(attr, '__getitem__'):
                    if is_binary:
                        new_attr = list(compress(attr, idx))
                    else:
                        new_attr = [attr[i] for i in idx]
                else:
                    raise TypeError(f'The attribute `{attr_name}` is corrupted.')
                setattr(ret, attr_name, new_attr)
            return ret
        raise IndexError
    
    def __repr__(self):
        from pprint import pformat
        return pformat([ info for info in self ])

def combine_box_lists(box_lists):
    n_boxes = 0
    for box_list in box_lists:
        n_boxes += len(box_list)

    pointer = 0
    matched_idx_pointer = 0
    ret = BoxList(n_boxes)
    for box_list in box_lists:
        if len(box_list) != 0:
            for attr_name in BoxList.keys():
                ret_attr = getattr(ret, attr_name)
                box_attr = getattr(box_list, attr_name)
                if type(ret_attr) != type(box_attr):
                    raise TypeError(f'Type mismatch for attribute `{attr_name}`. Expected `{type(ret_attr)}`, got `{type(box_attr)}`')

                if attr_name == 'matched_idx':
                    if box_list.paired_list is not None:
                        box_attr = box_attr + matched_idx_pointer
                    else:
                        warnings.warn('No paired BoxList defined, thus `matched_idx` cannot be properly combined.')

                if isinstance(box_attr, (list, np.ndarray)):
                    ret_attr[pointer:pointer+len(box_list)] = box_attr
                else:
                    raise TypeError(f'The attribute `{attr_name}` is corrupted.')

        pointer += len(box_list)
        matched_idx_pointer += len(box_list.paired_list)
    
    return ret