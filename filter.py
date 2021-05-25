import pickle
import numpy as np

__all__ = [
    'DetectionFilter', 'ClassFilter', 'RangeFilter',
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
        ret = (gt < self.range[0]) | (gt > self.range[1])
        ret[ignore_mask] = False
        return ret


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
    MIN_HEIGHT = [40, 25, 25]
    MAX_OCCLUSION = [0, 1, 2]
    MAX_TRUNCATION = [0.15, 0.3, 0.5]

    def __init__(self, class_name='car', class_label=1, difficulty='moderate', infos=None, info_path=None, *args, **kwargs):
        """ Detection filter for KITTI dataset.
        Results will be filtered by class and difficulty levels
        """
        if type(difficulty) is int and 0 <= difficulty < 3:
            pass
        elif type(difficulty) is str:
            difficulty = self.DIFFICULTY.index(difficulty)
        else:
            raise ValueError(f'invalid KITTI difficulty level: {difficulty}')

        class_name = class_name.lower()
        name = f'kitti_{class_name}_{self.DIFFICULTY[difficulty]}'
        super().__init__(name, *args, **kwargs)
        
        self.class_name = class_name
        self.class_label = class_label
        self.difficulty = difficulty

        if (infos is None and info_path is None) or (infos is not None and info_path is not None):
            raise ValueError('must provide either infos or info_path')
        elif infos is not None:        
            self.infos = infos
        else:
            self.infos = self.build_kitti_infos(info_path)
    
    @staticmethod
    def build_kitti_infos(info_path):
        with open(info_path, 'rb') as f:
            raw_infos = pickle.load(f)
        
        # Build metadata database for annotations
        # This will be used to calculate difficulty levels later
        kitti_infos = {}
        for info in raw_infos:
            frame_id = info['point_cloud']['lidar_idx']
            kitti_infos[frame_id] = []
            annos = info['annos']
            names = annos['name']
            boxes = annos['gt_boxes_lidar']
            occluded = annos['occluded']
            truncated = annos['truncated']
            bboxes = annos['bbox']
            heights = bboxes[:,3] - bboxes[:,1]
            num_points = annos['num_points_in_gt']
            bbox_occlusion = annos['occlusion_level'] if 'occlusion_level' in annos else np.full(len(names), np.nan)
            for name, box, o, t, h, n, bo in zip(names, boxes, occluded, truncated, heights, num_points, bbox_occlusion):
                kitti_infos[frame_id].append({
                    'name': name,
                    'loc': box[:2],
                    'occluded': o,
                    'truncated': t,
                    'height': h,
                    'num_points_in_gt': n,
                    'bbox_occlusion': bo
                })
        return kitti_infos

    @classmethod
    def check_difficulty(cls, difficulty, occluded, truncated, height):
        return (occluded <= cls.MAX_OCCLUSION[difficulty] and
                truncated <= cls.MAX_TRUNCATION[difficulty] and
                height > cls.MIN_HEIGHT[difficulty])

    def get_ignored_gt(self, gt):
        # Ignored boxes will not be counted towards FP if detected or FN if not detected
        # frame_id = gt['frame_id']
        # names = gt['gt_names']
        frame_id = gt['image']['image_idx']
        names = gt['annos']['name']

        ignored = np.zeros(len(names), dtype=bool)
        for idx, name in enumerate(names):
            name = name.lower()
            if name == 'dontcare':
                ignored[idx] = True
                continue
            annos = self.infos[frame_id][idx]
            # Ignore similar classes
            if self.class_name == 'pedestrian' and name == 'person_sitting':
                ignored[idx] = True
            elif self.class_name == 'car' and name == 'van':
                ignored[idx] = True
            # If label is a different class, don't ignore
            elif self.class_name != name:
                ignored[idx] = False
            # Ignore same class but different difficulty
            elif not self.check_difficulty(self.difficulty, annos['occluded'], annos['truncated'], annos['height']):
                ignored[idx] = True
            else:
                ignored[idx] = False
        return ignored
    
    def get_discarded_gt(self, gt):
        # Discarded boxes will not be counted towards FN if not detected,
        # but will be counted as FP if detected
        # labels = gt['gt_labels']
        classes = dict(
            Car=1, Pedestrian=2, Cyclist=3,
            Truck=-1, Misc=-1, Van=-1, Tram=-1, Person_sitting=-1,
            DontCare=-1
        )
        labels = []
        for name in gt['annos']['name']:
            if name == 'DontCare':
                continue
            labels.append(classes[name])
        # print('gt', gt)
        # print("gt['annos']['name']", gt['annos']['name'])
        # print('self.class_label', self.class_label)
        # print('labels', labels)
        # exit()
        ret = []
        for label in labels:
            ret.append(label != self.class_label)
        return ret
        # return (labels != self.class_label)

    def get_discarded_pred(self, pred):
        # Discarded prediction will not be regarded as positive
        # i.e. will not be counted as either TP or FP
        class_names = ['Car', 'Pedestrian', 'Cyclist']
        labels = np.array([class_names.index(n)+1 for n in pred['name']])
        return labels != self.class_label


def build_kitti_filters(info_path, class_names=['Car', 'Pedestrian', 'Cyclist'], class_labels=[1, 2, 3], difficulty_levels=['easy', 'moderate', 'hard'], *args, **kwargs):
    kitti_infos = KITTIFilter.build_kitti_infos(info_path)
    filters = []
    for name, label in zip(class_names, class_labels):
        for difficulty in difficulty_levels:
            filters.append( KITTIFilter(name, label, difficulty, kitti_infos, *args, **kwargs) )
    return filters
