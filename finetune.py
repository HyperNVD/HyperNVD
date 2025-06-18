import torch
import torch.optim as optim
import numpy as np
import random
import sys

from evaluate_finetune import eval
from datetime import datetime
from utils.unwrap_utils import *

import logging
from utils.config_utils import config_load, config_save

from pathlib import Path

from network.hypernvd import HyperVDN
from network.nvd_net import VDNetwork
from losses_finetune import get_losses
from torch.utils.data import DataLoader
from dataset.loaddata_personalized import VideoMAEDataset
import os
import time

best_psnr = 0

def seed_all(seed):
    if not seed:
        seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
   
def compute_losses(loss_funcs, data, output, rgb_output, index,
                    uv_mapping_scales, model, device, iteration, num_of_maps, 
                    samples, resx, resy, loss_cfg, use_alpha_gt):
    
    video_frames = data['video_frames'][0].to(device)
    video_frames_dx = data['video_frames_dx'][0].to(device)
    video_frames_dy = data['video_frames_dy'][0].to(device)
    optical_flows = data['optical_flows'][0].to(device)
    optical_flows_reverse = data['optical_flows_reverse'][0].to(device)
    optical_flows_mask = data['optical_flows_mask'][0].to(device)
    optical_flows_reverse_mask = data['optical_flows_reverse_mask'][0].to(device)
    masks_inital = data['mask_frames'][0].to(device)
    masks_gt = data['gt_mask_frames'][0].to(device)
    jif_current = data['jif_current'][0].to(device)
    
    rgb_textures = output['rgb']
    alpha = output['alpha']
    uvs = output['uv']
    rgb_residuals = output['residual']
    
    losses = {}
    # RGB loss
    if loss_funcs.get('rgb'):
        rgb_GT = video_frames[jif_current[1], jif_current[0], :, jif_current[2]].permute(1,0,2).to(device)
        losses['rgb'] = loss_funcs['rgb'](rgb_GT, rgb_output)

    # gradient loss
    if not use_alpha_gt:
        if loss_funcs.get('gradient'):
            losses['gradient'] = loss_funcs['gradient'](
                video_frames_dx, video_frames_dy,
                jif_current, rgb_output,
                device, model, index)

    # sparsity loss
    if loss_funcs.get('sparsity'):
        losses['sparsity'] = loss_funcs['sparsity'](rgb_textures, alpha)

    # alpha bootstrapping loss
    if not use_alpha_gt:
        if loss_funcs.get('alpha_bootstrapping'):
            if iteration <= loss_cfg['alpha_bootstrapping']['stop_iteration']:
                alpha_GT = masks_inital[jif_current[1], jif_current[0], jif_current[2],:].permute(1,0,2).to(device)
                losses['alpha_bootstrapping'] = loss_funcs['alpha_bootstrapping'](alpha_GT, alpha)

        # alpha regularization loss
        if loss_cfg['alpha_reg']['weight'] != 0:
            losses['alpha_reg'] = loss_funcs['alpha_reg'](alpha)

        # optical flow alpha loss
        if loss_funcs.get('flow_alpha'):
            losses['flow_alpha'] = loss_funcs['flow_alpha'](
                optical_flows, optical_flows_mask,
                optical_flows_reverse, optical_flows_reverse_mask,
                jif_current, alpha, device, model, index)

    # optical flow loss
    if loss_funcs.get('optical_flow'):
        loss_optical_flow = loss_funcs['optical_flow'](
                                    optical_flows, optical_flows_mask,
                                    optical_flows_reverse, optical_flows_reverse_mask,
                                    jif_current, uvs, uv_mapping_scales,
                                    device, model, index, True, alpha)
        for i in range(num_of_maps):
            losses['optical_flow_%d'%i] = loss_optical_flow[i]

    # rigidity loss
    if loss_funcs.get('rigidity'):
        for layer in range(num_of_maps):
            losses['rigidity_%d'%layer] = loss_funcs['rigidity'](
                                                jif_current,
                                                uvs[layer], uv_mapping_scales[layer],
                                                device, layer, model, index)

    # global rigidity loss
    if loss_funcs.get('global_rigidity'):
        if iteration <= loss_cfg['global_rigidity']['stop_iteration']:
            for layer, funcs in enumerate(loss_funcs['global_rigidity']):
                losses['global_rigidity_%d'%layer] = funcs(
                                                    jif_current,
                                                    uvs[layer], uv_mapping_scales[layer],
                                                    device, layer, model, index)

    # residual regularization loss
    if loss_funcs.get('residual_reg'):
        for i in range(num_of_maps):
            layer_net=getattr(model, f"layer_{i}")
            if layer_net.residual_net is not None:
                losses['residual_reg_%d'%i] = loss_funcs['residual_reg'](rgb_residuals[i])

    # residual consistent loss
    if loss_funcs.get('residual_consistent'):
        for i in range(num_of_maps):
            layer_net=getattr(model, f"layer_{i}")
            if layer_net.residual_net is not None:
                losses['residual_consistent_%d'%i] = loss_funcs['residual_consistent'](samples, resx, resy, device, i, model, index)
    
    return losses, sum(losses.values())
    
def evaluate(iteration, model, data, index, number_of_frames, num_of_maps, rez, eval_cfg, results_folder, device, batch_size, batch_id, best_psnr):
    if iteration is None: 
        eval_save_dir = str(results_folder/"initial_masks"/('%01d'%int(index)))
        iteration = 0
    else: 
        eval_save_dir = str(results_folder/('%06d'%iteration)/('%01d'%int(index)))
    
    
    if iteration is None or iteration % eval_cfg['interval'] == 0: # and iteration > 0:  
        jif_all = data['jif_all'][0]
        video_frames = data['video_frames'][0]
        masks_gt = data['gt_mask_frames'][0]
        
        psnr, psnr_no_residual, iou = eval(model, index,
                                            jif_all, video_frames, masks_gt, 
                                            number_of_frames, rez, eval_cfg['samples_batch'],
                                            num_of_maps, eval_save_dir, device, None,
                                            iteration = iteration)

        # uncomment if you want to use wandb for logging
        # wandb.log({f"evaluation/PSNR_{index.item()}": psnr, "iteration": iteration})

        if psnr > best_psnr[index]:
            best_psnr[index] = psnr
            os.makedirs(f"{results_folder}/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"{results_folder}/checkpoints/best_{index.item()}.pth")
            print(f"Model saved at iteration {iteration} with PSNR {psnr}")

    model.train()
        
        
def innit_logger(results_folder, cfg):
    logging.basicConfig(filename=f'{results_folder}/logger.log', level=logging.INFO, format='%(asctime)s %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)
    logging.info('Started')
    
def main(cfg):
    seed_all(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = cfg['sequences']
        
    resx = np.int64(cfg["resx"])
    resy = np.int64(cfg["resy"])
    iters_num = cfg["iters_num"]
    loss_cfg = cfg["losses"]
    lr= cfg["learning_rate"]
    
    samples = cfg["samples_batch"]
    eval_cfg = cfg["evaluation"]

    results_folder_name = cfg["results_folder_name"]
    folder_suffix = cfg["folder_suffix"]

    uv_mapping_scales = cfg["uv_mapping_scales"]
    
    if not isinstance(uv_mapping_scales, list):
        uv_mapping_scales = [uv_mapping_scales] * len(cfg["sequences"])
    
    mappings_cfg = cfg["model_mapping"]
    alpha_cfg = cfg.get("alpha", None)
    hypernet_cfg = cfg["hypernet"]
    prior_cfg = cfg["prior"]

    results_folder = Path(f'{results_folder_name}/{folder_suffix}_{datetime.utcnow().strftime("%d-%m_%H:%M:%S")}')
    os.makedirs(results_folder, exist_ok=True)

    config_save(cfg, '%s/config.py'%results_folder)
    innit_logger(results_folder, cfg)

    logging.info("Using Seed: %d" % cfg["seed"])
    num_of_maps = len(mappings_cfg)
    logging.info('number of layers: %d' % num_of_maps)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_of_frames = cfg["maximum_number_of_frames"]
    rez = np.maximum(resx, resy)
    
    dataset = VideoMAEDataset(root = cfg['path2DAVIS'], split='trainval-480', resx=resx, resy=resy, rez=rez,
                           use_mask_rcnn_bootstrapping = True, filter_optical_flow = True,
                           frame_num=number_of_frames, sep_gtmask=True, training=True, sequences=sequences, 
                           samples_batch = cfg["samples_batch"], uv_mapping_scales=uv_mapping_scales, load_mask_gt = cfg["use_alpha_gt"])
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    hypernet =  HyperVDN(mappings_cfg, alpha_cfg, hypernet_cfg, prior_cfg, num_instances=len(cfg["sequences"]), 
                     bound=1, device=device, clip_mapping=False).to(device)

    hypernet.load_state_dict(torch.load(cfg['ckpt']))
    
    model = VDNetwork(mappings_cfg, alpha_cfg, num_mlp_layers=4, hidden_dim=256, bound=1)
    model = model.to(device)

    data = next(iter(dataloader))
    params = hypernet.get_params(data['emb'])
    topop = []
    extra = {k: v.clone() for k,v in params.items()}
    for k,v in params.items():
        if 'embeddings.weight' in k:
            del extra[k]
            extra[k.replace('embeddings.weight', 'embeddings')] = v.reshape(-1, 2)
        else:
            extra[k] = v[0]
    params = extra
    
    model.load_state_dict(extra)
    
    best_psnr = [0] * len(sequences)
    evaluate(0, model, data, torch.LongTensor([0]), number_of_frames, num_of_maps, rez, eval_cfg, 
                    results_folder, device, None, 0, best_psnr)

    optimizer_parameters = model.get_optimizer_list()
    optimizer = torch.optim.Adam(optimizer_parameters)

    loss_funcs = get_losses(loss_cfg, rez, number_of_frames, num_of_maps)

    # uncomment if you want to use wandb for logging
    # wandb.init(project='hypersprites', name="hypersprites-" + folder_suffix, config=cfg)
    model.train()

    for iteration in range(iters_num+1):
        loss = 0
        losses = {}
        start_iter = time.time()            
        for i, data in enumerate(dataloader):
            index = torch.LongTensor([i]).to(device)
            xyt_current = data['xyt_current'][0].to(device)

            output = model(xyt_current)
            rgb_output = sum([(output['rgb'][i] * output['residual'][i]) * output['alpha'][:,:,[i]] for i in range(len(output['rgb']))])
            
            it_losses, it_loss = compute_losses(loss_funcs, data, output, rgb_output, index,
                                            uv_mapping_scales, model, device, iteration, num_of_maps, 
                                            samples, resx, resy, loss_cfg, cfg["use_alpha_gt"])

            loss += it_loss / len(dataloader)
            losses = {name: (losses.get(name, 0) + l)/len(dataloader) for name, l in it_losses.items()}

            evaluate(iteration, model, data, index, number_of_frames, num_of_maps, rez, eval_cfg, 
                    results_folder, device, None, i, best_psnr)

            # uncomment if you want to use wandb for logging
            # wandb.log({f"batchLoss": it_loss, **{f"batchLoss_{name}": l for name, l in it_losses.items()}, "iteration": iteration, 'lr': optimizer.param_groups[0]['lr']})        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("----------------------------------------------------------")
        print(f"Iteration {iteration}, loss {loss.item()}, took {time.time() - start_iter:.2f} seconds")
        # wandb.log({"loss": loss.item(), "iteration": iteration, 'lr': optimizer.param_groups[0]['lr']})

        if iteration % eval_cfg['save_interval'] == 0:
            os.makedirs(f"{results_folder}/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"{results_folder}/checkpoints/last.pth")
            print(f"Model saved at iteration {iteration}")

    # uncomment if you want to use wandb for logging
    # wandb.finish()

if __name__ == "__main__":
    main(config_load(sys.argv[1]))
    