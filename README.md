# HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks

### [Project Page](https://hypernvd.github.io/) | [Paper](https://arxiv.org/abs/2503.17276)

## Installation

Our code is compatible and validate with Python 3.9.16, PyTorch 1.13.1, and CUDA 11.7.

```
conda create -n hashing-nvd python=3.9
conda activate hashing-nvd
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install matplotlib tensorboard scipy  scikit-image tqdm
pip install opencv-python imageio-ffmpeg gdown

# if desired you can also download wandb for logging
# pip install wandb
CC=gcc-9 CXX=g++-9 python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install easydict
CC=gcc-9 CXX=g++-9 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Data preparations

To train our model we use the DAVIS-2016 dataset that you can download from here: [DAVIS](https://davischallenge.org/davis2016/code.html)

### Video frames

The video frames follows the format of [DAVIS](https://davischallenge.org/) dataset. The file type of images should be all either in png or jpg and named as `00000.jpg`, `00001.jpg`, ...

### Preprocess optical flow

We extract the optical flow using [RAFT](https://arxiv.org/abs/2003.12039). The submodule can be linked by the following command:

```
git submodule update --init
cd thirdparty/RAFT/
./download_models.sh
cd ../..
```

To create optical flow for the whole DAVIS dataset, run:

```
python preprocess_optical_flow.py --path2DAVIS "" --max_long_edge 768
```

The script will automatically generate the corresponding backward and forward optical flow and store the npy files in the right directory.

### Preprocess MAE embeddings

We extract the embeddings of the DAVIS dataset with the following command:

```
python prepare_dataset/create_videos_DAVIS.py --path2DAVIS ""
python prepare_dataset/EmbedAllVideoMAE.py --path2DAVIS ""
```
This scripts will automatically generate the .mp4 and embeddings of the davis dataset and store them in path2DAVIS/MP4 and path2DAVIS/EMBEDDIGNS respectively.

## Train with multiple videos

To train our model like we propose in our paper, you can use the following command: 

```
python train.py config/train_30_videos.py
```

You will need to change the "path2DAVIS" inside the config file. 
You can download our checkpoint for our model trained with 15 videos and 30 videos in the following links.

## Finetune a new videos

To finetune our model in a new video, run:

```
python finetune.py config/finetune_car-shadow.py
```

You need to replace the `path2DAVIS` to the folder of DAVIS dataset.

The config file and checkpoint file will be stored to the assigned result folder.

## Citation

If you find our work useful in your research, please consider citing:

```
@misc{pilligua2025hypernvdacceleratingneuralvideo,
      title={HyperNVD: Accelerating Neural Video Decomposition via Hypernetworks}, 
      author={Maria Pilligua and Danna Xue and Javier Vazquez-Corral},
      year={2025},
      eprint={2503.17276},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.17276}, 
}
```

## Acknowledgement

We thank [Hashing-nvd](https://github.com/vllab/hashing-nvd/) and [Layered Neural Atlases](https://github.com/ykasten/layered-neural-atlases) for using their code implementation as our code base. We modify the code structures to meet our requirements.
