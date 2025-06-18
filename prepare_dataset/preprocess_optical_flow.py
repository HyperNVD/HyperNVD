from pathlib import Path
import argparse
import numpy as np
from raft_wrapper import RAFTWrapper

from tqdm import tqdm
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess image sequence')
    parser.add_argument('--path2DAVIS', type=str, help='Path to DAVIS dataset')
    parser.add_argument('--max_long_edge', type=int,default='2000', help='maximum image dimension to process without resizing')

    args = parser.parse_args()

    path2DAVIS = args.path2DAVIS
    data_path = Path(f'{path2DAVIS}/JPEGImages/480p/')
    lst = os.listdir(data_path)
    
    for sequence_name in lst:
        files = sorted((data_path / sequence_name).glob('*.jpg'))
        out_flow_dir = Path(f'{path2DAVIS}/trainval-480p/DAVIS/flow/' + sequence_name)
        out_flow_dir.mkdir(exist_ok=True)
        raft_wrapper = RAFTWrapper(
            model_path='thirdparty/RAFT/models/raft-things.pth', max_long_edge=args.max_long_edge)
        
        for i, file1 in enumerate(tqdm(files,desc='computing flow')):
            if i < len(files) - 1:
                file2 = files[i + 1]
                fn1 = file1.name
                fn2 = file2.name
                out_flow12_fn = out_flow_dir / f'{fn1}_{fn2}.npy'
                out_flow21_fn = out_flow_dir / f'{fn2}_{fn1}.npy'

                overwrite=False
                if not out_flow12_fn.exists() and not out_flow21_fn.exists() or overwrite:
                    im1, im2 = raft_wrapper.load_images(str(file1), str(file2))
                    flow12 = raft_wrapper.compute_flow(im1, im2)
                    flow21 = raft_wrapper.compute_flow(im2, im1)
                    np.save(out_flow12_fn, flow12)
                    np.save(out_flow21_fn, flow21)
