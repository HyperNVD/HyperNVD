import cv2
import numpy as np
from PIL import Image


def get_all_masks(mask_dir, num_frames, resx=None, resy=None, separate_objects_masks=True):
    # masks = get_all_elements(mask_dir)
    # print("mask_dir", mask_dir)
    mask_files = sorted(list(mask_dir.glob('*.png')))

    # print("mask_files", mask_files)
    if len(mask_files) == 0:
        all_masks = np.zeros((resy, resx, num_frames))
        # masks_void = np.zeros_like(masks)
    else:
        mask_files = mask_files[0:num_frames]
        # print(num_frames, len(mask_files))
        obj = np.array(Image.open(mask_files[0]))
        if resy is None:
            resx=obj.shape[1]
            resy=obj.shape[0]
        masks = np.zeros((resy, resx, len(mask_files)))

        for i, obj in enumerate(mask_files):
            mask = np.array(Image.open(obj))
            # interpolation=cv2.INTER_NEAREST gets right nearest interpolation results, cv2.INTER_NEAREST gets bilinear interpolation results
            mask = cv2.resize(mask, (resx, resy), interpolation=cv2.INTER_NEAREST)
            masks[...,i] = mask
        # masks_void = np.zeros_like(masks)
        
        # Separate void and object masks
        for i in range(masks.shape[-1]):
            # masks_void[i, ...] = masks[i, ...] == 255
            masks[masks[..., i] == 255, i] = 0

        print("separate_objects_masks", separate_objects_masks)
        if separate_objects_masks:
            num_objects = int(np.max(masks[...,i]))
            tmp = np.ones((*masks.shape, num_objects))
            tmp = tmp * np.arange(1, num_objects + 1)[None, None, None, :]
            masks = (tmp == masks[..., None])
            masks = masks > 0
            all_masks = masks.astype(int)
        else:
            all_masks = np.expand_dims(masks, axis=3)
            all_masks[masks > 0] == 1

    return all_masks #, masks_void



def compute_iou(pred, target, dim=None):
    """
    Compute region similarity as the Jaccard Index.
    :param pred (binary tensor) prediction
    :param target (binary tensor) ground truth
    :param dim (optional, int) the dimension to reduce across
    :returns jaccard (float) region similarity
    """
    intersect = pred & target
    union = pred | target
    if dim is None:
        intersect = intersect.sum()
        union = union.sum()
    else:
        intersect = intersect.sum(dim)
        union = union.sum(dim)
    return (intersect + 1e-6) / (union + 1e-6)


def compute_multiple_iou(pred_mask, gt_mask):

    # mask_bin = np.zeros_like(pred_mask)
    # mask_bin[pred_mask>0.5]=1

    mask_bin = pred_mask > 0.5
    gt_bin = gt_mask==1
    ious = compute_iou(mask_bin, gt_bin) 

    return ious


if __name__ == "__main__":
    import os
    from pathlib import Path
    data_dir = "./data/boxing-fisheye"
    mask_dir = os.path.join(data_dir, 'gt_mask')
    save_dir = os.path.join(data_dir, 'gt_masks_sep')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images = os.listdir(mask_dir)
    num_frames = len(images)
    all_masks = get_all_masks(Path(mask_dir), num_frames, separate_objects_masks=True)
    num_objects = all_masks.shape[-1]

    for mask_id in range(num_objects):
        save_dir_sub = os.path.join(save_dir, 'object_{:02d}'.format(mask_id))
        if not os.path.exists(save_dir_sub):
            os.makedirs(save_dir_sub)
        for frame_id in range(num_frames):
            mask = all_masks[:,:,frame_id,mask_id]
            mask = mask * 255
            cv2.imwrite(os.path.join(save_dir_sub, "{:05d}.png".format(frame_id)), mask.astype(np.uint8))