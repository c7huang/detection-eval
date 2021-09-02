from functools import partial
import pickle
import multiprocessing as mp

import numpy as np

try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
        print('[detection_eval] Jupyter environment detected. Using tqdm.notebook.')
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm

try:
    # For integration with the active learning framework
    from ..metric import Metric
except:
    class Metric():
        def __init__(self, name=None, **eval_args):
            self.name = self.__name__ = self.__class__.__name__ if name is None else name
            self.eval_args = eval_args

        def __call__(self, gts, preds):
            return self.evaluate(gts, preds, **self.eval_args)

        def template(self):
            return {key: None for key in self.keys()}

        def keys(self):
            return [self.name]

        @staticmethod
        def evaluate(gts, preds, **eval_args):
            raise NotImplementedError

from .box_list import BoxList, combine_box_lists

__all__ = ['DetectionEval']

class DetectionEval(Metric):
    default_metrics = ['ap']
    def __init__(self, thresholds, criterion='iou', epsilon=0.1,
        filters=None, n_positions=100, metrics=None,
        gt_processor=None, pred_processor=None, verbose=False):
        super().__init__(
            thresholds=thresholds,
            criterion=criterion,
            epsilon=epsilon,
            filters=filters,
            n_positions=n_positions,
            metrics=metrics,
            gt_processor=gt_processor,
            pred_processor=pred_processor,
            verbose=verbose
        )

    def keys(self):
        keys = []
        metrics = self.eval_args['metrics']
        if metrics is None:
            metrics = self.default_metrics
        for metric in metrics:
            if self.eval_args['filters'] is None:
                keys.append(f'{metric}')
            else:
                for filta in self.eval_args['filters']:
                    if hasattr(filta, 'name'):
                        keys.append(f'{metric}_{filta.name}')
                    else:
                        keys.append(f'{metric}_{filta.__name__}')
        return keys

    @staticmethod
    def compute_ctable(boxes1, boxes2, criterion, device=0):
        """ Compute matching values based on criterion between each pair of boxes from boxes1 and boxes2

        Args:
            boxes1 (np.ndarray): 2D boxes: [N_a, x, y, w, h, ...], 3D boxes: [N_a, x, y, z, w, l, h, ...]
            boxes2 (np.ndarray): 2D boxes: [N_b, x, y, w, h, ...], 3D boxes: [N_b, x, y, z, w, l, h, ...]
            criterion (str): one of ['iou', 'iou_2d', 'iou_3d', 'iou_bev'] or scipy.spatial.distance.cdist metrics

        Returns:
            np.ndarray: [N_a, N_b] matrix where each entry corresponds to a maching value
        """
        if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]))

        if boxes1.shape[1] != boxes2.shape[1]:
            raise ValueError('Boxe size mismatch')

        if criterion in ['iou', 'iou_2d', 'iou_bev', 'iof', 'iof_2d', 'iof_bev']:
            from .box_iou_rotated import box_iou_rotated
            if boxes1.shape[1] == 7:
                boxes1 = boxes1[:,[0,1,3,4,6]]
                boxes2 = boxes2[:,[0,1,3,4,6]]
            iou = box_iou_rotated(boxes1, boxes2, mode=criterion[:3], device=device)
            if hasattr(iou, 'cpu'):
                iou = iou.cpu().numpy()
            return iou

        if criterion in ['iou_3d', 'iof_3d']:
            from .box_iou_rotated import box_iou_rotated_3d
            iou_3d = box_iou_rotated_3d(boxes1, boxes2, mode=criterion[:3], device=device)
            if hasattr(iou_3d, 'cpu'):
                iou_3d = iou_3d.cpu().numpy()
            return iou_3d

        cdist_metrics = [
            'euclidean', 'cityblock', 'cosine',
            'euclidean_3d', 'cityblock_3d', 'cosine_3d'
        ]
        if criterion in cdist_metrics:
            from scipy.spatial.distance import cdist
            if criterion[-2:] == '3d':
                criterion = criterion[:-3]
                coords1 = boxes1[:,:3]
                coords2 = boxes2[:,:3]
            else:
                coords1 = boxes1[:,:2]
                coords2 = boxes2[:,:2]
            return cdist(coords1, coords2, metric=criterion)
        
        raise ValueError(f'Unknown criterion {criterion}')

    @classmethod
    def evaluate_one_sample(cls, gt, pred, thresholds, criterion='iou', epsilon=0.1,
        filta=None, ctable=None, label_dtype=int, gt_processor=None, pred_processor=None, device=0):
        """ Evaluate one sample

        Args:
            gt (any): ground truths for the current sample. If not in (labels, boxes) format, then `gt_processor` is required.
            pred (any): predictions for the current sample. If not in (labels, scores, boxes) format, then `pred_processor` is required.
            thresholds (list or dict): matching thresholds for different classes.
            criterion (str, optional): box matching criterion. Defaults to 'iou'.
            epsilon (float, optional): minimum matching threshold. Defaults to 0.1.
            filta (DetectionFilter, optional): filter object for detection evaluation. Defaults to None.
            ctable (ndarray, optional): pairwise distance table between the gt and pred boxes. Defaults to None.
            gt_processor (callable, optional): a function that transforms `gt`into (labels, boxes) format. Defaults to None.
            pred_processor (callable, optional): a function that transforms `pred` into (labels, scores, boxes) format. Defaults to None.

        Returns:
            tuple: (gt_list, pred_list)
        """
        if filta is None:
            filta = lambda gt, pred: (None, None, None)
        elif not callable(filta):
            raise ValueError('Filter needs to be callable')

        if 'iou' in criterion or 'iof' in criterion:
            better = np.greater
        else:
            better = np.less

        # Extract ground truth labels and boxes
        gt_labels, gt_boxes = gt_processor(gt) if callable(gt_processor) else gt
        assert(len(gt_labels) == len(gt_boxes))

        # Extract predicted labels and boxes
        pred_labels, pred_scores, pred_boxes = pred_processor(pred) if callable(pred_processor) else pred
        assert(len(pred_scores) == len(pred_labels) == len(pred_boxes))

        filtered_masks = filta(gt, pred)
        if len(filtered_masks) == 2:
            ignored_gt = None
            discarded_gt, discarded_pred = filtered_masks
        elif len(filtered_masks) == 3:
            ignored_gt, discarded_gt, discarded_pred = filtered_masks
        
        # Compute matching scores (iou, distance, etc.) between each pair
        # of the predicted and ground truth boxes
        if ctable is None:
            ctable = cls.compute_ctable(gt_boxes, pred_boxes, criterion, device=device)
        
        gt_list = BoxList(len(gt_labels), label_dtype=label_dtype)
        pred_list = BoxList(len(pred_labels), label_dtype=label_dtype)
        gt_list.pair_with(pred_list)
        
        ################################################################################ 
        # False negative loop
        # For each ground truth label/box, find the best matching prediction and
        # evaluate matching score.
        ################################################################################ 
        for gt_idx, gt_label in enumerate(gt_labels):
            gt_info = gt_list[gt_idx]
            gt_info.gt_label = gt_label

            if discarded_gt is not None and discarded_gt[gt_idx]:
                gt_info.ignored = True
                continue
            if ignored_gt is not None and ignored_gt[gt_idx]:
                gt_info.ignored = True
                continue

            # Best matching score and pred idx for same class
            best_match_sc = -np.inf if better is np.greater else np.inf
            best_pred_idx_sc = np.nan
            # Best matching score and pred idx considering all classes
            best_match_ac = -np.inf if better is np.greater else np.inf
            best_pred_idx_ac = np.nan

            for pred_idx, pred_label in enumerate(pred_labels):
                # NOTE: do not skip discarded predictions when calculating FN
                match = ctable[gt_idx, pred_idx]
                # Record best match if better than existing one
                if better(match, best_match_sc) and gt_label == pred_label:
                    best_match_sc = match
                    best_pred_idx_sc = pred_idx
                if better(match, best_match_ac):
                    best_match_ac = match
                    best_pred_idx_ac = pred_idx

            if better(best_match_sc, thresholds[gt_label]):
                # TP case, correctly classified, localized
                gt_info.localized = True
                gt_info.loc_score = best_match_sc
                gt_info.classified = True
                gt_info.pred_label = gt_label
                gt_info.pred_score = pred_scores[best_pred_idx_sc]
                gt_info.matched_idx = best_pred_idx_sc
                pred_list.matched_idx[best_pred_idx_sc] = gt_idx
                if discarded_pred is not None:
                    discarded_pred[best_pred_idx_sc] = False
            else:
                # Not TP, check bounding boxes for all classes
                gt_info.bg = not better(best_match_ac, epsilon)
                gt_info.loc_score = best_match_ac
                if not gt_info.bg:
                    gt_info.localized = better(best_match_ac, thresholds[gt_label])
                    gt_info.classified = gt_label == pred_labels[best_pred_idx_ac]
                    gt_info.pred_label = pred_labels[best_pred_idx_ac]
                    gt_info.pred_score = pred_scores[best_pred_idx_ac]


        ################################################################################ 
        # False positive loop
        # For each predicted label/box, find the best matching GT and
        # evaluate matching score.
        ################################################################################ 
        for pred_idx, (pred_label, pred_score) in enumerate(zip(pred_labels, pred_scores)):
            pred_info = pred_list[pred_idx]
            pred_info.pred_label = pred_label
            pred_info.pred_score = pred_score

            if discarded_pred is not None and discarded_pred[pred_idx]:
                pred_info.ignored = True
                continue

            # Best matching score and pred idx for same class
            best_match_sc = -np.inf if better is np.greater else np.inf
            best_gt_idx_sc = np.nan
            # Best matching score and pred idx considering all classes
            best_match_ac = -np.inf if better is np.greater else np.inf
            best_gt_idx_ac = np.nan

            for gt_idx, gt_label in enumerate(gt_labels):
                # NOTE: do not skip discarded GTs when calculating FN
                match = ctable[gt_idx, pred_idx]
                # Record best match if better than existing one
                if better(match, best_match_sc) and gt_label == pred_label:
                    best_match_sc = match
                    best_gt_idx_sc = gt_idx
                if better(match, best_match_ac):
                    best_match_ac = match
                    best_gt_idx_ac = gt_idx

            if better(best_match_sc, thresholds[pred_label]):
                # TP case, correctly classified, localized
                pred_info.localized = True
                pred_info.loc_score = best_match_sc
                pred_info.classified = True
                pred_info.gt_label = pred_label
                if discarded_gt is not None and discarded_gt[best_gt_idx_sc]:
                    pred_info.ignored = True
                if ignored_gt is not None and ignored_gt[best_gt_idx_sc]:
                    pred_info.ignored = True
                if not pred_info.ignored and pred_info.matched_idx != best_gt_idx_sc:
                    import warnings
                    warnings.warn('Something is wrong with evaluation')
                pred_info.matched_idx = best_gt_idx_sc
            else:
                # Not TP, check bounding boxes for all classes
                pred_info.bg = not better(best_match_ac, epsilon)
                pred_info.loc_score = best_match_ac
                if not pred_info.bg:
                    gt_label = gt_labels[best_gt_idx_ac]
                    pred_info.localized = better(best_match_ac, thresholds[gt_label])
                    pred_info.classified = gt_label == pred_label 
                    pred_info.gt_label = gt_label
                    if pred_info.localized and ignored_gt is not None and ignored_gt[best_gt_idx_ac]:
                        pred_info.ignored = True

        return (gt_list, pred_list)
    
    @classmethod
    def evaluate_all_samples(cls, gts, preds, thresholds, criterion='iou', epsilon=0.1,
        filta=None, ctables=None, label_dtype=int, 
        gt_processor=None, pred_processor=None, pbar=None, callback=None, device=0):
        if not (callback is None or callable(callback)):
            raise ValueError('Callback must be callable, callback(idx, gt, pred, gt_list, pred_list)')

        if ctables is None:
            ctables = {}
        
        if pbar is True:
            print()
            pbar = tqdm(total=len(gts), desc=f'Evaluating detection ({filta})')

        gt_lists = [None] * len(gts)
        pred_lists = [None] * len(preds)
        for sample_idx, (pred, gt) in enumerate(zip(preds, gts)):
            _, gt_boxes = gt_processor(gt) if callable(gt_processor) else gt
            _, _, pred_boxes = pred_processor(pred) if callable(pred_processor) else pred

            if sample_idx not in ctables:
                ctables[sample_idx] = cls.compute_ctable(gt_boxes, pred_boxes, criterion, device=device)

            gt_list_i, pred_list_i = cls.evaluate_one_sample(
                gt=gt, pred=pred,
                thresholds=thresholds,
                criterion=criterion,
                epsilon=epsilon,
                filta=filta,
                ctable=ctables[sample_idx],
                label_dtype=label_dtype,
                gt_processor=gt_processor,
                pred_processor=pred_processor,
                device=device
            )

            if callable(callback):
                callback(sample_idx, gt, pred, gt_list_i, pred_list_i)

            gt_lists[sample_idx] = gt_list_i
            pred_lists[sample_idx] = pred_list_i

            if pbar is not None:
                pbar.update()
        
        gt_list = combine_box_lists(gt_lists)
        pred_list = combine_box_lists(pred_lists)
        gt_list.pair_with(pred_list)

        return gt_list, pred_list


    @classmethod
    def evaluate_all_samples_mp(cls, gts, preds, thresholds, criterion='iou', epsilon=0.1,
        filta=None, ctables=None, label_dtype=int, 
        gt_processor=None, pred_processor=None, pbar=None, callback=None, processes=4):
        import torch
        
        if not (callback is None or callable(callback)):
            raise ValueError('Callback must be callable, callback(idx, gt, pred, gt_list, pred_list)')

        if ctables is None:
            ctables = {}

        with mp.Pool(processes) as p:
            args = list(zip(
                np.array_split( np.array(gts), processes ),
                np.array_split( np.array(preds), processes ),
                [thresholds] * processes,
                [criterion] * processes,
                [epsilon] * processes,
                [filta] * processes,
                [None] * processes,
                [label_dtype] * processes,
                [gt_processor] * processes,
                [pred_processor] * processes,
                [None] * (processes-1) + [True],
                [None] * processes,
                np.arange(processes) % torch.cuda.device_count()
            ))
            rets = p.starmap(cls.evaluate_all_samples, args)
            gt_lists = []
            pred_lists = []
            for gt_list, pred_list in rets:
                gt_lists.append(gt_list)
                pred_lists.append(pred_list)
            
        gt_list = combine_box_lists(gt_lists)
        pred_list = combine_box_lists(pred_lists)
        gt_list.pair_with(pred_list)

        return gt_list, pred_list

    
    @classmethod
    def compute_statistics(cls, gt_list, pred_list, n_positions=100):
        gt_list = gt_list[~gt_list.ignored]
        pred_list = pred_list[~pred_list.ignored]

        # Sort by prediction scores (confidence)
        sorted_idx = np.argsort(pred_list.pred_scores)[::-1]
        pred_list = pred_list[sorted_idx]

        # Compute binary indicators and cummulative sums for TP and FP
        tp_mask = pred_list.localized & pred_list.classified
        tp_cum = np.cumsum(tp_mask)
        fp_cum = np.cumsum(~tp_mask)
        tp_count = np.sum(tp_mask)
        fn_count = np.sum(~(gt_list.localized & gt_list.classified))
        gt_count = tp_count + fn_count

        # Interpolate recall positions
        rec = tp_cum / gt_count
        prec = tp_cum / ( tp_cum + fp_cum )
        rec_interp = np.linspace(0, 1, n_positions+1)
        prec_interp = np.interp(rec_interp, rec, prec, right=0)

        prob_fn = fn_count / gt_count

        sigma_99 = 2.58 * np.sqrt( ( prob_fn * (1-prob_fn) ) / gt_count )
        prob_fn_bound = prob_fn + sigma_99

        results = {
            'rec': rec_interp.tolist(),
            'prec': prec_interp.tolist(),
            'ap': np.mean(prec_interp),
            'tp': int(tp_count),
            'fn': int(fn_count),
            'prob_fn': prob_fn,
            'prob_fn_bound': prob_fn_bound
        }

        # ################################################################################
        # # Compute statistics for FP categorization
        # ################################################################################
        # for fp_type in BoxType:
        #     if fp_type == BoxType.IGN or fp_type == BoxType.TP:
        #         continue
                
        #     # Ignore FP of different type and recompute cummulative sum
        #     fp_type_mask = fp_mask.copy()
        #     fp_type_mask[fp >= fp_type] = False
        #     fp_type_cum = np.cumsum(fp_type_mask)

        #     # Compute precision and AP for the current FP type
        #     prec_fp = tp_cum / ( tp_cum + fp_type_cum )
        #     prec_fp_interp = np.interp(rec_interp, rec, prec_fp, right=0)
        #     ap = np.mean(prec_fp_interp)

        #     results[f'fp_{fp_type.name.lower()}'] = fp_type_cum.tolist()
        #     results[f'prec_fp_{fp_type.name.lower()}'] = prec_fp_interp.tolist()
        #     results[f'ap_fp_{fp_type.name.lower()}'] = ap

        
        # ################################################################################
        # # Compute statistics for FN categorization
        # ################################################################################
        # for fn_type in BoxType:
        #     if fn_type == BoxType.IGN or fn_type == BoxType.TP:
        #         continue

        #     fn_type_count = np.sum((fn > BoxType.TP) & (fn < fn_type))

        #     rec_fn = tp_cum / ( tp_count + fn_type_count )
        #     prec_fn_interp = np.interp(rec_interp, rec_fn, prec, right=0)
        #     ap = np.mean(prec_fn_interp)

        #     results[f'fn_{fn_type.name.lower()}'] = ((tp_count + fn_type_count) - tp_cum).tolist()
        #     results[f'prec_fn_{fn_type.name.lower()}'] = prec_fn_interp.tolist()
        #     results[f'ap_fn_{fn_type.name.lower()}'] = ap

        return results


    @classmethod
    def evaluate(cls, gts, preds, thresholds, criterion='iou', epsilon=0.1,
        filters=None, n_positions=100, metrics=None, label_dtype=int,
        gt_processor=None, pred_processor=None, processes=-1, verbose=False, **kwargs):
        """ Evaluate all samples

        Args:
            gts (any): list of ground truth objects. If not in (labels, boxes) format, then `gt_processor` is required.
            preds (any): list of prediction outputs. If not in (labels, scores, boxes) format, then `pred_processor` is required.
            thresholds (list or dict): matching thresholds for different classes.
            criterion (str, optional): box matching criterion. Defaults to 'iou'.
            epsilon (float, optional): minimum matching threshold. Defaults to 0.1.
            filters (list, optional): a list of filter objects for detection evaluation. Defaults to None.
            n_positions (int, optional): number of recall positions for AP evaluation. Defaults to 100.
            gt_processor (callable, optional): a function that transforms `gt`into (labels, boxes) format. Defaults to None.
            pred_processor (callable, optional): a function that transforms `pred` into (labels, scores, boxes) format. Defaults to None.

        Returns:
            dict: a dictionary containing evaluation results
        """

        if filters is None:
            filters = [None]

        pbar = tqdm(total=len(filters)*len(gts), desc='Evaluating detection') if verbose else None

        results = {}
        ctables = {}
        for filta in filters:
            if processes > 0:
                gt_list, pred_list = cls.evaluate_all_samples_mp(
                    gts=gts, preds=preds,
                    thresholds=thresholds,
                    criterion=criterion,
                    epsilon=epsilon,
                    filta=filta,
                    ctables=ctables,
                    label_dtype=label_dtype,
                    gt_processor=gt_processor,
                    pred_processor=pred_processor,
                    pbar=pbar,
                    processes=processes
                )
            else:
                gt_list, pred_list = cls.evaluate_all_samples(
                    gts=gts, preds=preds,
                    thresholds=thresholds,
                    criterion=criterion,
                    epsilon=epsilon,
                    filta=filta,
                    ctables=ctables,
                    label_dtype=label_dtype,
                    gt_processor=gt_processor,
                    pred_processor=pred_processor,
                    pbar=pbar
                )
            stats = cls.compute_statistics(gt_list, pred_list, n_positions=n_positions)

            if metrics is None:
                metrics = cls.default_metrics
            for metric in metrics:
                if filta is None:
                    key = f'{metric}'
                elif hasattr(filta, 'name'):
                    key = f'{metric}_{filta.name}'
                else:
                    key = f'{metric}_{filta.__name__}'
                results[key] = stats[metric]

        return results