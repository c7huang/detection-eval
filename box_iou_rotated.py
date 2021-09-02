import numpy as np

__all__ = ['box_iou_rotated', 'box_iou_rotated_3d']

def box_iou_rotated_numba(boxes1, boxes2, mode='iou'):
    from .box_iou_rotated_numba import rotate_iou_gpu_eval
    return rotate_iou_gpu_eval(boxes1, boxes2, criterion=0 if mode == 'iof' else -1)

def box_iou_rotated_torch(boxes1, boxes2, mode='iou', device=0):
    import torch
    from mmcv.ops import box_iou_rotated
    torch.cuda.set_device(torch.device('cuda', device))
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.from_numpy(boxes1).float().cuda()
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.from_numpy(boxes2).float().cuda()
    return box_iou_rotated(boxes1, boxes2, mode=mode)

def box_iou_rotated(boxes1, boxes2, mode='iou', version='torch', device=0):
    if version == 'numba':
        return box_iou_rotated_numba(boxes1, boxes2, mode=mode)
    if version == 'torch':
        return box_iou_rotated_torch(boxes1, boxes2, mode=mode, device=device)
    raise ValueError(f'Unknown version {version}.')

def box_iou_rotated_3d_numba(boxes1, boxes2, mode='iou', z_axis=2, z_center=0.0):
    from .box_iou_rotated_numba import d3_box_overlap
    return d3_box_overlap(boxes1, boxes2, criterion=0 if mode == 'iof' else -1, z_axis=z_axis, z_center=z_center)

def box_iou_rotated_3d_new(boxes1, boxes2, mode='iou', z_axis=2, z_center=0.0, version='torch', device=0):
    if version == 'torch':
        import torch as alg
        alg.cuda.set_device(alg.device('cuda', device))
        if not isinstance(boxes1, alg.Tensor):
            boxes1 = alg.from_numpy(boxes1).float().cuda()
        if not isinstance(boxes2, alg.Tensor):
            boxes2 = alg.from_numpy(boxes2).float().cuda()
    else:
        import numpy as alg

    bev_axis = np.r_[0:z_axis,z_axis+1:z_axis+3,z_axis+4:7]
    boxes1_bev = boxes1[:,bev_axis]
    boxes2_bev = boxes2[:,bev_axis]
    iof_bev = box_iou_rotated(boxes1_bev, boxes2_bev, mode='iof', version=version).flatten()

    n_boxes1 = boxes1.shape[0]
    n_boxes2 = boxes2.shape[0]
    stack_idx = np.mgrid[:n_boxes1,:n_boxes2].reshape(2, n_boxes1*n_boxes2)

    boxes1_top = boxes1[:,z_axis] + boxes1[:,z_axis+3]*(1-z_center)
    boxes1_bot = boxes1[:,z_axis] - boxes1[:,z_axis+3]*z_center
    boxes1_vol = boxes1[:,3]*boxes1[:,4]*boxes1[:,5]
    boxes1_area = boxes1_vol / boxes1[:,z_axis+3]

    boxes2_top = boxes2[:,z_axis] + boxes2[:,z_axis+3]*(1-z_center)
    boxes2_bot = boxes2[:,z_axis] - boxes2[:,z_axis+3]*z_center
    boxes2_vol = boxes2[:,3]*boxes2[:,4]*boxes2[:,5]

    boxes1_top = boxes1_top[stack_idx[0]]
    boxes1_bot = boxes1_bot[stack_idx[0]]
    boxes1_vol = boxes1_vol[stack_idx[0]]
    boxes1_area = boxes1_area[stack_idx[0]]

    boxes2_top = boxes2_top[stack_idx[1]]
    boxes2_bot = boxes2_bot[stack_idx[1]]
    boxes2_vol = boxes2_vol[stack_idx[1]]

    z_top_min = alg.minimum(boxes1_top, boxes2_top)
    z_bot_max = alg.maximum(boxes1_bot, boxes2_bot)
    intersect_area = iof_bev * boxes1_area
    intersect_height = alg.maximum(z_top_min - z_bot_max, alg.zeros_like(z_top_min))
    intersect_vol = intersect_area * intersect_height

    if mode == 'iou':
        union_vol = boxes1_vol + boxes2_vol - intersect_vol
    elif mode == 'iof':
        union_vol = boxes1_vol
    
    iou_3d = intersect_vol / union_vol
    iou_3d = iou_3d.reshape((n_boxes1, n_boxes2))
    return iou_3d

def box_iou_rotated_3d(boxes1, boxes2, mode='iou', z_axis=2, z_center=0.0, version='torch', device=0):
    if version == 'numba':
        return box_iou_rotated_3d_numba(boxes1, boxes2, mode=mode)
    if version == 'torch':
        return box_iou_rotated_3d_new(boxes1, boxes2, mode=mode, z_axis=z_axis, z_center=z_center, version=version, device=device)
    if version[-3:] == 'new':
        return box_iou_rotated_3d_new(boxes1, boxes2, mode=mode, z_axis=z_axis, z_center=z_center, version=version[:-4], device=device)
    raise ValueError(f'Unknown version {version}.')
