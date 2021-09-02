import pickle
import numpy as np

__all__ = [
    'DetectionFilter', 'ClassFilter', 'RangeFilter', 'ValueFilter',
    'CombinedFilter', 'build_class_filters',
    'KITTIFilter', 'build_kitti_filters'
]

class DetectionFilter():
    def __init__(
        self,
        name,
        gt_processor=None,
        pred_processor=None
    ):
        self.name = name
        self.gt_processor = gt_processor
        self.pred_processor = pred_processor
        if gt_processor is not None and not callable(gt_processor):
            raise ValueError('gt_processor must be callable')
        if pred_processor is not None and not callable(pred_processor):
            raise ValueError('pred_processor must be callable')

    def __setattr__(self, name, value):
        if name in ['gt_processor', 'pred_processor'] and \
                value is not None and not callable(value):
            raise ValueError(f'{name} must be callable')
        return super().__setattr__(name, value)

    def __str__(self):
        return self.name
    
    def __call__(self, gt, pred):
        if callable(self.gt_processor):
            gt = self.gt_processor(gt)
        if callable(self.pred_processor):
            pred = self.pred_processor(pred)
        return (
            self.get_ignored_gt(gt),
            self.get_discarded_gt(gt),
            self.get_discarded_pred(pred)
        )

    def get_ignored_gt(self, gt):
        return None

    def get_discarded_gt(self, gt):
        return None

    def get_discarded_pred(self, pred):
        return None


class ClassFilter(DetectionFilter):
    def __init__(self, name, label, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.label = label

    def get_discarded_gt(self, gt):
        labels, boxes = gt
        if isinstance(labels, (list, tuple, np.ndarray)):
            labels = np.array(labels)
        else:
            raise TypeError(f'Unknown type `{type(labels)}` for labels, expected list, tuple, or np.ndarray.')
        return labels != self.label

    def get_discarded_pred(self, pred):
        labels, scores, boxes = pred
        if isinstance(labels, (list, tuple, np.ndarray)):
            labels = np.array(labels)
        else:
            raise TypeError(f'Unknown type `{type(labels)}` for labels, expected list, tuple, or np.ndarray.')
        return labels != self.label

def build_class_filters(names, labels, gt_processor, pred_processor):
    assert(len(names) == len(labels))
    return [ClassFilter(
        name, label,
        gt_processor,
        pred_processor) for name, label in zip(names, labels)]


class RangeFilter(DetectionFilter):
    def __init__(self, name, value_range, *args, **kwargs):
        super().__init__(
            name=f'{name}:{value_range[0]}-{value_range[1]}', *args, **kwargs)
        self.range = value_range
    
    def get_ignored_gt(self, gt):
        ignore_mask = np.isnan(gt)
        ret = (gt < self.range[0]) | (gt >= self.range[1])
        ret[ignore_mask] = False
        return ret


class ValueFilter(DetectionFilter):
    def __init__(self, name, value, keep=True, *args, **kwargs):
        super().__init__(
            name=f'{name}:{value}', *args, **kwargs
        )
        self.keep = keep
        self.value = value
    
    def get_ignored_gt(self, gt):
        if self.keep:
            return gt != self.value
        else:
            return gt == self.value


class CombinedFilter(DetectionFilter):
    def __init__(self, filters, mode='and', *args, **kwargs):
        super().__init__(name='_'.join([f.name for f in filters]), *args, **kwargs)
        self.filters = filters
        if not isinstance(mode, str) or mode.lower() not in ['and', 'or']:
            raise ValueError(f'Invalid mode {mode}')
        self.mode = mode.lower()

    def get_ignored_gt(self, gt):
        ret = None
        for f in self.filters:
            gt_proc = f.gt_processor(gt) if callable(f.gt_processor) else gt
            mask = f.get_ignored_gt(gt_proc)
            if mask is None:
                continue
            if self.mode == 'and':
                ret = mask if ret is None else (ret | mask)
            elif self.mode == 'or':
                ret = mask if ret is None else (ret & mask)
            else:
                raise ValueError(f'Invalid mode {mode}')
        return ret
    
    def get_discarded_gt(self, gt):
        ret = None
        for f in self.filters:
            gt_proc = f.gt_processor(gt) if callable(f.gt_processor) else gt
            mask = f.get_discarded_gt(gt_proc)
            if mask is None:
                continue
            if self.mode == 'and':
                ret = mask if ret is None else (ret | mask)
            elif self.mode == 'or':
                ret = mask if ret is None else (ret & mask)
            else:
                raise ValueError(f'Invalid mode {mode}')
        return ret
    
    def get_discarded_pred(self, pred):
        ret = None
        for f in self.filters:
            pred_proc = f.pred_processor(pred) if callable(f.pred_processor) else pred
            mask = f.get_discarded_pred(pred_proc)
            if mask is None:
                continue
            if self.mode == 'and':
                ret = mask if ret is None else (ret | mask)
            elif self.mode == 'or':
                ret = mask if ret is None else (ret & mask)
            else:
                raise ValueError(f'Invalid mode {mode}')
        return ret


class KITTIFilter(DetectionFilter):
    DIFFICULTY = ['easy', 'moderate', 'hard']

    def __init__(self, class_name='Car', difficulty='moderate', *args, **kwargs):
        """KITTI evaluation filter

        Args:
            class_name (str, optional): class name. Defaults to 'Car'.
            difficulty (str, optional): difficulty level. Defaults to 'moderate'.
            gt_processor (callable, optional): takes in an object and returns a tuple of (class names, bounding boxes, occlusion, truncation)

        Raises:
            ValueError: when invalid difficulty is provided
        """

        if type(difficulty) is int and 0 <= difficulty < 3:
            pass
        elif type(difficulty) is str:
            difficulty = self.DIFFICULTY.index(difficulty)
        else:
            raise ValueError(f'invalid KITTI difficulty level: {difficulty}')
        self.difficulty = difficulty

        self.class_name = class_name
        class_name = class_name.lower()
        name = f'kitti_{class_name}_{self.DIFFICULTY[difficulty]}'

        super().__init__(name, *args, **kwargs)

    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]
    @classmethod
    def check_difficulty(cls, difficulty, box_height, occlusion, truncation):
        """Check if a set of (box height, occlusion, truncation) is of a particular difficulty level
        * Easy:     Min. bounding box height: 40 Px,
                    Max. occlusion level: Fully visible,
                    Max. truncation: 15 %
        * Moderate: Min. bounding box height: 25 Px,
                    Max. occlusion level: Partly occluded,
                    Max. truncation: 30 %
        * Hard:     Min. bounding box height: 25 Px,
                    Max. occlusion level: Difficult to see,
                    Max. truncation: 50 % 

        Args:
            difficulty (int): 0: easy, 1: moderate, 2: hard
            box_height (float, np.ndarray): bounding box height
            occlusion (int, np.ndarray): occlusion level
            truncation (float, np.ndarray): truncation

        Returns:
            bool:
        """
        if isinstance(box_height, np.ndarray) and isinstance(occlusion, np.ndarray) and isinstance(truncation, np.ndarray):
            return  (box_height > cls.MIN_HEIGHT[difficulty]) & \
                    (occlusion <= cls.MAX_OCCLUSION[difficulty]) & \
                    (truncation <= cls.MAX_TRUNCATION[difficulty])
        return (box_height > cls.MIN_HEIGHT[difficulty] and
                occlusion <= cls.MAX_OCCLUSION[difficulty] and
                truncation <= cls.MAX_TRUNCATION[difficulty])

    def get_ignored_gt(self, gt):
        # Ignored boxes will not be counted towards FP if detected or FN if not detected
        names, bboxes, occlusions, truncations = gt
        current_difficulty = self.check_difficulty( self.difficulty, bboxes[:,3]-bboxes[:,1], occlusions, truncations )

        ignored = np.zeros(len(names), dtype=bool)
        for idx, name in enumerate(names):
            # Three cases:
            # * same class but different difficulty level
            # * detecting person_sitting as pedestrian
            # * detection van as car
            if ( self.class_name == name and not current_difficulty[idx] ) or \
               ( self.class_name == 'Pedestrian' and name == 'Person_sitting' ) or \
               ( self.class_name == 'Car' and name == 'Van' ):
                ignored[idx] = True
        return ignored

    def get_discarded_gt(self, gt):
        # Discarded boxes will not be counted towards FN if not detected,
        # but will be counted as FP if detected
        labels = gt[0]
        return labels != self.class_name

    def get_discarded_pred(self, pred):
        # Discarded prediction will not be regarded as positive
        # i.e. will not be counted as either TP or FP
        labels = pred[0]
        return labels != self.class_name


def build_kitti_filters(class_names=['Car', 'Pedestrian', 'Cyclist'], 
                        difficulty_levels=['easy', 'moderate', 'hard'], *args, **kwargs):
    filters = []
    for name in class_names:
        for difficulty in difficulty_levels:
            filters.append( KITTIFilter(name, difficulty, *args, **kwargs) )
    return filters
