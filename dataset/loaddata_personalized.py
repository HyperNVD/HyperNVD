import torch
import torch.utils.data
import numpy as np
import os
from PIL import Image
import sys
import cv2

from utils.unwrap_utils import resize_flow, compute_consistency, get_tuples
from utils.seg_utils import get_all_masks, compute_multiple_iou

import torch
from random import randint

from pathlib import Path

import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
from utils.unwrap_utils import resize_flow, compute_consistency
from utils.seg_utils import get_all_masks, compute_multiple_iou

import torch
import cv2


class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, root, task='unsupervised', subset='val', sequences='all', resolution='480p', codalab=False):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root, annotations_folder, resolution)
        year = '2019' if task == 'unsupervised' and (subset == 'test-dev' or subset == 'test-challenge') else '2017'
        self.imagesets_path = os.path.join(self.root, 'ImageSets/480p')

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'DAVIS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found, download the missing subset '
                                    f'for the {self.task} task from {self.DATASET_WEB}')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder for the {self.task} task not found, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            # mask = None if msk is None else np.array(Image.open(msk))
            yield image

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')


    def get_all_masks(self, sequence, num_frames, resx=None, resy=None, separate_objects_masks=True):
        mask_files = self.sequences[sequence]['masks']

        if len(mask_files) == 0:
            all_masks = np.zeros((resy, resx, num_frames))
        else:
            mask_files = mask_files[0:num_frames]
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
           
            # Separate void and object masks
            for i in range(masks.shape[-1]):
                masks[masks[..., i] == 255, i] = 0

            # print("separate_objects_masks", separate_objects_masks)
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



    def get_sequences(self):
        for seq in self.sequences:
            yield seq




def get_1_item(resy, resx, maximum_number_of_frames,
                data_folder, use_mask_rcnn_bootstrapping,
                filter_optical_flow,
                sep_gtmask=True):

    video_frames_dir = data_folder / 'video_frames'
    flow_dir = data_folder / 'flow'
    mask_dirs = list((data_folder / 'masks').glob('*'))
    gt_mask_dir = data_folder / 'gt_mask'

    input_files = sorted(list(video_frames_dir.glob('*.jpg')) + list(video_frames_dir.glob('*.png')))
    number_of_frames=np.minimum(maximum_number_of_frames, len(input_files))
    
    gt_mask_frames = get_all_masks(gt_mask_dir, number_of_frames, resx, resy, separate_objects_masks=sep_gtmask)
    
    video_frames = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dx = torch.zeros((resy, resx, 3, number_of_frames))
    video_frames_dy = torch.zeros((resy, resx, 3, number_of_frames))

    # objects + 1 background
    if use_mask_rcnn_bootstrapping:
        num_of_objects = len(mask_dirs)
    else:
        num_of_objects = 1

    mask_frames = torch.zeros((resy, resx, number_of_frames, num_of_objects+1))

    optical_flows = torch.zeros((resy, resx, 2, number_of_frames,  1))
    optical_flows_mask = torch.zeros((resy, resx, number_of_frames,  1))
    optical_flows_reverse = torch.zeros((resy, resx, 2, number_of_frames,  1))
    optical_flows_reverse_mask = torch.zeros((resy, resx, number_of_frames, 1))

    
    mask_files_list = [sorted(list(d.glob('*.jpg')) + list(d.glob('*.png'))) for d in mask_dirs]
    for i in range(number_of_frames):
        file1 = input_files[i]
        im = np.array(Image.open(str(file1))).astype(np.float64) / 255.
        if use_mask_rcnn_bootstrapping:
            for j, mask_files in enumerate(mask_files_list):
                # print(i, mask_files)
                mask = np.array(Image.open(str(mask_files[i]))).astype(np.float64) / 255.
                mask = cv2.resize(mask, (resx, resy), interpolation=cv2.INTER_NEAREST)
                mask_frames[:, :, i, j] = torch.from_numpy(mask)

        video_frames[:, :, :, i] = torch.from_numpy(cv2.resize(im[:, :, :3], (resx, resy)))
        video_frames_dy[:-1, :, :, i] = video_frames[1:, :, :, i] - video_frames[:-1, :, :, i]
        video_frames_dx[:, :-1, :, i] = video_frames[:, 1:, :, i] - video_frames[:, :-1, :, i]
    mask_frames[..., -1] = 1 - mask_frames.amax(dim=-1)


    for i in range(number_of_frames - 1):
        file1 = input_files[i]
        j = i + 1
        file2 = input_files[j]

        fn1 = file1.name
        fn2 = file2.name

        flow12_fn = flow_dir / f'{fn1}_{fn2}.npy'
        flow21_fn = flow_dir / f'{fn2}_{fn1}.npy'
        flow12 = np.load(flow12_fn)
        flow21 = np.load(flow21_fn)

        if flow12.shape[0] != resy or flow12.shape[1] != resx:
            flow12 = resize_flow(flow12, newh=resy, neww=resx)
            flow21 = resize_flow(flow21, newh=resy, neww=resx)
        mask_flow = compute_consistency(flow12, flow21) < 1.0
        mask_flow_reverse = compute_consistency(flow21, flow12) < 1.0

        optical_flows[:, :, :, i, 0] = torch.from_numpy(flow12)
        optical_flows_reverse[:, :, :, j, 0] = torch.from_numpy(flow21)

        if filter_optical_flow:
            optical_flows_mask[:, :, i, 0] = torch.from_numpy(mask_flow)
            optical_flows_reverse_mask[:, :, j, 0] = torch.from_numpy(mask_flow_reverse)
        else:
            optical_flows_mask[:, :, i, 0] = torch.ones_like(mask_flow)
            optical_flows_reverse_mask[:, :, j, 0] = torch.ones_like(mask_flow_reverse)
    return video_frames, video_frames_dx, video_frames_dy, optical_flows, optical_flows_reverse, optical_flows_mask, optical_flows_reverse_mask, mask_frames, gt_mask_frames




class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, training, resy, resx, use_mask_rcnn_bootstrapping,
                 filter_optical_flow, rez, frame_num=2, load_flow=False, 
                 load_pl=False, transform=None, subsample_frame_interval=None, 
                 flow_suffix="", zero_ann=False, pl_root=None, sequences='all', 
                 resolution= "480p",  sep_gtmask=True, samples_batch=10000, 
                 uv_mapping_scales=[0.9, 0.9, 0.9, 0.6], load_mask_gt = False):
        super().__init__()



        self.dataset_davis = DAVIS(root=root, task='semi-supervised', subset='train', sequences=sequences, resolution=resolution)

        self.root = root
        self.split = split
        self.training = training
        self.resy = resy
        self.resx = resx
        self.frame_num = frame_num
        self.load_flow = load_flow
        self.load_pl = load_pl
        self.transform = transform
        self.subsample_frame_interval = subsample_frame_interval
        self.flow_suffix = flow_suffix
        self.zero_ann = zero_ann
        self.pl_root = pl_root
        self.use_mask_rcnn_bootstrapping = use_mask_rcnn_bootstrapping
        self.filter_optical_flow = filter_optical_flow
        self.sep_gtmask = sep_gtmask
        self.resolution = resolution
        self.samples_batch = samples_batch
        self.rez = rez
        self.uv_mapping_scales = uv_mapping_scales
        self.load_masks_gt = load_mask_gt
        
        self.sequences = list(self.dataset_davis.get_sequences())

        if not self.training:
            if self.frame_num != 1:
                print(f"You need single frames for evaluaion but have {self.frame_num} frames. It will be set to 1.")
                self.frame_num = 1
                
        # for each video, choose a random offset
        self.offsets = [randint(0, len(list(self.dataset_davis.get_frames(sequence_name))) - self.frame_num - 1) for sequence_name in self.sequences]
        self.offsets = [0] * len(self.sequences)

        self.sequences_data = [self.load_1_instance(sequence_name) for sequence_name in self.sequences]

    def load_image(self, im):
        # im = Image.open(str(path))
        im = np.array(im).astype(np.float64) / 255.
        return im
    
    def resize_image(self, im, resy, resx):
        im = cv2.resize(im[:, :, :3], (self.resx, self.resy))
        return torch.from_numpy(im)

    def __getitem__(self, index):
        data = self.sequences_data[index]
        
        inds_foreground = torch.randint(data['jif_all'].shape[1], (np.int64(self.samples_batch), 1))
        # print("inds_foreground", inds_foreground.shape)

        data['jif_current'] = data['jif_all'][:, inds_foreground]  # size (3, batch, 1)

        data['xyt_current'] = torch.cat((data['jif_current'][0:1] / (self.rez / 2) - 1, data['jif_current'][1:2] / (self.rez / 2) - 1, data['jif_current'][2:3] / (data['number_of_frames'] / 2) - 1), dim=2)  # size (batch, 3)
        # print("jif_all", jif_all.shape)
        # print("jif_current", jif_current.shape)
        # print("xyt_current", xyt_current.shape)

        return data
    
    def load_1_instance(self, sequence_name):
        flow_dir = Path(os.path.join(self.root, 'flow', sequence_name))
        # if self.load_masks_gt:
        masks_dir = Path(os.path.join(self.root, 'Annotations', '480p', sequence_name))
        mask_dirs = masks_dir
        # else: 
            # masks_dir = Path(os.path.join(self.root, 'masks', sequence_name))
            # mask_dirs = masks_dir / 'merged'
        
        input_files = list(self.dataset_davis.get_frames(sequence_name))
        number_of_frames=np.minimum(self.frame_num, len(input_files))
        gt_mask_frames = self.dataset_davis.get_all_masks(sequence=sequence_name, num_frames=number_of_frames, resx=self.resx, resy=self.resy, separate_objects_masks=self.sep_gtmask)
        
        num_of_objects = 1
        
        video_frames = torch.zeros((self.resy, self.resx, 3, number_of_frames))
        video_frames_dx = torch.zeros((self.resy, self.resx, 3, number_of_frames))
        video_frames_dy = torch.zeros((self.resy, self.resx, 3, number_of_frames))

        mask_frames = torch.zeros((self.resy, self.resx, number_of_frames, num_of_objects+1))
        
        mask_files_list = list(mask_dirs.glob('*.png'))
        mask_files_list.sort()

        # offset = self.offsets[self.sequences.index(sequence_name)]
        offset = 0
        for i in range(number_of_frames):
            i_off = i + offset

            # Load the image
            im = self.load_image(input_files[i_off])
            video_frames[:, :, :, i] = self.resize_image(im, self.resy, self.resx)
            video_frames_dy[:-1, :, :, i] = video_frames[1:, :, :, i] - video_frames[:-1, :, :, i]
            video_frames_dx[:, :-1, :, i] = video_frames[:, 1:, :, i] - video_frames[:, :-1, :, i]
            
            # Load the mask
            mask = np.array(Image.open(str(mask_files_list[i_off]))).astype(np.float64) / 255.
            mask = cv2.resize(mask, (self.resx, self.resy), interpolation=cv2.INTER_NEAREST)
            mask_frames[:, :, i, 0] = torch.from_numpy(mask)
        
        mask_frames[..., -1] = 1 - mask_frames.amax(dim=-1)

        optical_flows = torch.zeros((self.resy, self.resx, 2, number_of_frames,  1))
        optical_flows_mask = torch.zeros((self.resy, self.resx, number_of_frames,  1))
        optical_flows_reverse = torch.zeros((self.resy, self.resx, 2, number_of_frames,  1))
        optical_flows_reverse_mask = torch.zeros((self.resy, self.resx, number_of_frames, 1))

        for i in range(number_of_frames - 1):
            j = i + 1
            j_off = i_off + 1 

            fn1 = self.dataset_davis.sequences[sequence_name]['images'][i_off].split("/")[-1]
            fn2 = self.dataset_davis.sequences[sequence_name]['images'][j_off].split("/")[-1]

            flow12_fn = flow_dir / f'{fn1}_{fn2}.npy'
            flow21_fn = flow_dir / f'{fn2}_{fn1}.npy'
            flow12 = np.load(flow12_fn)
            flow21 = np.load(flow21_fn)

            if flow12.shape[0] != self.resy or flow12.shape[1] != self.resx:
                flow12 = resize_flow(flow12, newh=self.resy, neww=self.resx)
                flow21 = resize_flow(flow21, newh=self.resy, neww=self.resx)
            mask_flow = compute_consistency(flow12, flow21) < 1.0
            mask_flow_reverse = compute_consistency(flow21, flow12) < 1.0

            optical_flows[:, :, :, i, 0] = torch.from_numpy(flow12)
            optical_flows_reverse[:, :, :, j, 0] = torch.from_numpy(flow21)

            if self.filter_optical_flow:
                optical_flows_mask[:, :, i, 0] = torch.from_numpy(mask_flow)
                optical_flows_reverse_mask[:, :, j, 0] = torch.from_numpy(mask_flow_reverse)
            else:
                optical_flows_mask[:, :, i, 0] = torch.ones_like(mask_flow)
                optical_flows_reverse_mask[:, :, j, 0] = torch.ones_like(mask_flow_reverse)

            jif_all = get_tuples(number_of_frames, video_frames)  # it does this every time, remember to change it 
            # print("jif_all", jif_all.shape) # [3, 8294400]
                
        return {
            'video_frames': video_frames,
            'video_frames_dx': video_frames_dx,
            'video_frames_dy': video_frames_dy,
            'optical_flows': optical_flows,
            'optical_flows_reverse': optical_flows_reverse,
            'optical_flows_mask': optical_flows_mask,
            'optical_flows_reverse_mask': optical_flows_reverse_mask,
            'mask_frames': mask_frames,
            'gt_mask_frames': gt_mask_frames,
            'jif_all': jif_all,
            'number_of_frames': number_of_frames
        }
        



    def __len__(self):
        return len(self.sequences)



class VideoMAEDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, training, resy, resx, use_mask_rcnn_bootstrapping,
                 filter_optical_flow, rez, frame_num=2, load_flow=False, 
                 load_pl=False, transform=None, subsample_frame_interval=None, 
                 flow_suffix="", zero_ann=False, pl_root=None, sequences='all', 
                 resolution= "480p",  sep_gtmask=True, samples_batch=10000, 
                 uv_mapping_scales=[0.9, 0.9, 0.9, 0.6], load_mask_gt = False):
        super().__init__()

        self.emb_folder = root + "/EMBEDDINGS"
        self.files = os.listdir(self.emb_folder)

        self.info = {name.split(".")[0]:0 for name in self.files}
        for name in self.files:
            idx = int(name.split("_")[-1][:-3])
            n = name.split(".")[0]
            if idx > self.info[n]:
                self.info[n] = idx

        self.names = list(self.info.keys())
        
        self.embs = {name:{} for name in self.names}

        for name in self.names:
            for i in range((self.info[name])):
                self.embs[name][i] = torch.load(os.path.join(self.emb_folder, f"{name}.mp4_{i}.pt"))


        self.dataset_davis = DAVIS(root=root, task='semi-supervised', subset='train', sequences=sequences, resolution=resolution)

        self.root = root
        self.split = split
        self.training = training
        self.resy = resy
        self.resx = resx
        self.frame_num = frame_num
        self.load_flow = load_flow
        self.load_pl = load_pl
        self.transform = transform
        self.subsample_frame_interval = subsample_frame_interval
        self.flow_suffix = flow_suffix
        self.zero_ann = zero_ann
        self.pl_root = pl_root
        self.use_mask_rcnn_bootstrapping = use_mask_rcnn_bootstrapping
        self.filter_optical_flow = filter_optical_flow
        self.sep_gtmask = sep_gtmask
        self.resolution = resolution
        self.samples_batch = samples_batch
        self.rez = rez
        self.uv_mapping_scales = uv_mapping_scales
        self.load_masks_gt = load_mask_gt
        
        self.sequences = list(self.dataset_davis.get_sequences())

        if not self.training:
            if self.frame_num != 1:
                print(f"You need single frames for evaluaion but have {self.frame_num} frames. It will be set to 1.")
                self.frame_num = 1
                
        # for each video, choose a random offset
        self.offsets = [randint(0, len(list(self.dataset_davis.get_frames(sequence_name))) - self.frame_num - 1) for sequence_name in self.sequences]
        self.offsets = [0] * len(self.sequences)

        self.sequences_data = [self.load_1_instance(sequence_name) for sequence_name in self.sequences]

    def load_image(self, im):
        # im = Image.open(str(path))
        im = np.array(im).astype(np.float64) / 255.
        return im
    
    def resize_image(self, im, resy, resx):
        im = cv2.resize(im[:, :, :3], (self.resx, self.resy))
        return torch.from_numpy(im)

    def __getitem__(self, index):
        data = self.sequences_data[index]
        name = data['name']
        start = data['start']
        emb = self.embs[name][start]
        data['emb'] = emb
        
        inds_foreground = torch.randint(data['jif_all'].shape[1], (np.int64(self.samples_batch), 1))
        # print("inds_foreground", inds_foreground.shape)

        data['jif_current'] = data['jif_all'][:, inds_foreground]  # size (3, batch, 1)

        data['xyt_current'] = torch.cat((data['jif_current'][0:1] / (self.rez / 2) - 1, data['jif_current'][1:2] / (self.rez / 2) - 1, data['jif_current'][2:3] / (data['number_of_frames'] / 2) - 1), dim=2)  # size (batch, 3)
        # print("jif_all", jif_all.shape)
        # print("jif_current", jif_current.shape)
        # print("xyt_current", xyt_current.shape)

        return data
    
    def load_1_instance(self, sequence_name):
        flow_dir = Path(os.path.join(self.root, 'flow', sequence_name))
        # if self.load_masks_gt:
        masks_dir = Path(os.path.join(self.root, 'Annotations', '480p', sequence_name))
        mask_dirs = masks_dir
        # else: 
            # masks_dir = Path(os.path.join(self.root, 'masks', sequence_name))
            # mask_dirs = masks_dir / 'merged'
        
        input_files = list(self.dataset_davis.get_frames(sequence_name))
        number_of_frames=np.minimum(self.frame_num, len(input_files))
        gt_mask_frames = self.dataset_davis.get_all_masks(sequence=sequence_name, num_frames=number_of_frames, resx=self.resx, resy=self.resy, separate_objects_masks=self.sep_gtmask)
        
        num_of_objects = 1
        
        video_frames = torch.zeros((self.resy, self.resx, 3, number_of_frames))
        video_frames_dx = torch.zeros((self.resy, self.resx, 3, number_of_frames))
        video_frames_dy = torch.zeros((self.resy, self.resx, 3, number_of_frames))

        mask_frames = torch.zeros((self.resy, self.resx, number_of_frames, num_of_objects+1))
        
        mask_files_list = list(mask_dirs.glob('*.png'))
        mask_files_list.sort()

        offset = self.offsets[self.sequences.index(sequence_name)]
        for i in range(number_of_frames):
            i_off = i + offset

            # Load the image
            im = self.load_image(input_files[i_off])
            video_frames[:, :, :, i] = self.resize_image(im, self.resy, self.resx)
            video_frames_dy[:-1, :, :, i] = video_frames[1:, :, :, i] - video_frames[:-1, :, :, i]
            video_frames_dx[:, :-1, :, i] = video_frames[:, 1:, :, i] - video_frames[:, :-1, :, i]
            
            # Load the mask
            mask = np.array(Image.open(str(mask_files_list[i_off]))).astype(np.float64) / 255.
            mask = cv2.resize(mask, (self.resx, self.resy), interpolation=cv2.INTER_NEAREST)
            mask_frames[:, :, i, 0] = torch.from_numpy(mask)
        
        mask_frames[..., -1] = 1 - mask_frames.amax(dim=-1)

        optical_flows = torch.zeros((self.resy, self.resx, 2, number_of_frames,  1))
        optical_flows_mask = torch.zeros((self.resy, self.resx, number_of_frames,  1))
        optical_flows_reverse = torch.zeros((self.resy, self.resx, 2, number_of_frames,  1))
        optical_flows_reverse_mask = torch.zeros((self.resy, self.resx, number_of_frames, 1))

        for i in range(number_of_frames - 1):
            j = i + 1
            j_off = i_off + 1 

            fn1 = self.dataset_davis.sequences[sequence_name]['images'][i_off].split("/")[-1]
            fn2 = self.dataset_davis.sequences[sequence_name]['images'][j_off].split("/")[-1]

            flow12_fn = flow_dir / f'{fn1}_{fn2}.npy'
            flow21_fn = flow_dir / f'{fn2}_{fn1}.npy'
            flow12 = np.load(flow12_fn)
            flow21 = np.load(flow21_fn)

            if flow12.shape[0] != self.resy or flow12.shape[1] != self.resx:
                flow12 = resize_flow(flow12, newh=self.resy, neww=self.resx)
                flow21 = resize_flow(flow21, newh=self.resy, neww=self.resx)
            mask_flow = compute_consistency(flow12, flow21) < 1.0
            mask_flow_reverse = compute_consistency(flow21, flow12) < 1.0

            optical_flows[:, :, :, i, 0] = torch.from_numpy(flow12)
            optical_flows_reverse[:, :, :, j, 0] = torch.from_numpy(flow21)

            if self.filter_optical_flow:
                optical_flows_mask[:, :, i, 0] = torch.from_numpy(mask_flow)
                optical_flows_reverse_mask[:, :, j, 0] = torch.from_numpy(mask_flow_reverse)
            else:
                optical_flows_mask[:, :, i, 0] = torch.ones_like(mask_flow)
                optical_flows_reverse_mask[:, :, j, 0] = torch.ones_like(mask_flow_reverse)

            jif_all = get_tuples(number_of_frames, video_frames)  # it does this every time, remember to change it 
            # print("jif_all", jif_all.shape) # [3, 8294400]
                
        return {
            'video_frames': video_frames,
            'video_frames_dx': video_frames_dx,
            'video_frames_dy': video_frames_dy,
            'optical_flows': optical_flows,
            'optical_flows_reverse': optical_flows_reverse,
            'optical_flows_mask': optical_flows_mask,
            'optical_flows_reverse_mask': optical_flows_reverse_mask,
            'mask_frames': mask_frames,
            'gt_mask_frames': gt_mask_frames,
            'jif_all': jif_all,
            'number_of_frames': number_of_frames,
            'name': sequence_name,
            'start': offset
        }
        



    def __len__(self):
        return len(self.sequences)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(1)
    
    frames = 5
    sequences = ['bmx-trees', 'breakdance-flare', 'libby', 'scooter-gray', 'soapbox']
    dataset = VideoDataset(root = '/ghome/mpilligua/video_editing/DAVIS', split='trainval-480', resx=768, resy=432, 
                           use_mask_rcnn_bootstrapping = True, filter_optical_flow = True,
                           frame_num=frames, sep_gtmask=True, training=True, sequences=sequences)

    item = dataset[1]
    
    fig, ax = plt.subplots(2, frames, figsize=(frames*4, 4))
    # plt.title(seq)
    frames = item['video_frames']
    masks = item['mask_frames']
    for i in range(frames.shape[-1]):
        ax[0][i].imshow(frames[:, :, :, i])
        ax[0][i].axis('off')
        
        ax[1][i].imshow(masks[:, :, i, 0]*255)
        ax[1][i].axis('off')

    plt.tight_layout()
    os.makedirs('/ghome/mpilligua/video_editing/hypersprites-no-residual-no-hyphash/test3/', exist_ok=True)
    plt.savefig(f'/ghome/mpilligua/video_editing/hypersprites-no-residual-no-hyphash/test3/bear.png')
    
