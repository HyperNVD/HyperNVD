import cv2
import os
import torch
import torchvision
import imageio
import skimage.metrics
import skimage.measure
import numpy as np


from PIL import Image
from tqdm import tqdm
from utils.seg_utils import compute_multiple_iou
from network.metamodules import get_subdict
from dataset.loaddata_personalized import VideoMAEDataset

# taken from https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
# sample coordinates x,y from image im.
def bilinear_interpolate_numpy(im, x, y):
    x = x.flatten()
    y = y.flatten()
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    img = (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

    return np.expand_dims(img, axis=0)


def eval_data_gen(model, index, jif_all, number_of_frames, rez, samples_batch, num_of_maps, texture_size, device, mask_frames):
    '''
    Given the whole model settings.
    Return:
        - rec_masks: Maximum alpha value sampled of UV map of each layer.
        - xyts: Normalized spatial and temporal location of each frame.
        - alphas: Alpha value of each layer of each frame.
            `[ndarray, ndarray, ...]`, len = number of frames
        - uvs: Color of UV map of each layer of each frame.
            `[[uv1, uv2, ...], [uv1, uv2, ...], ...]`, len = number of frames
        - residuals: Residual value of each layer of each frame, corresponding to video coordinate.
            `[[residual1, residual2, ...], [residual1, residual2, ...], ...]`, len = number of frames
        - rgbs: Color of each layer of each frame, corresponding to video coordinate.
            `[[rgb1, rgb2, ...], [rgb1, rgb2, ...], ...]`, len = number of frames
    '''
    model.eval()
    with torch.no_grad():
        samples_batch = min(samples_batch, jif_all.shape[1])
        
        rec_x = jif_all[0, :, None] / (rez / 2) - 1.0
        rec_y = jif_all[1, :, None] / (rez / 2) - 1.0
        rec_t = jif_all[2, :, None] / (number_of_frames / 2) - 1.0
        rec_xyt = torch.cat((rec_x, rec_y, rec_t), dim=1).unsqueeze(0)
        batch_xyt = rec_xyt.split(samples_batch, dim=1)
        
        # print(mask_frames.shape)
        if mask_frames is not None:
            mask_frames = mask_frames[jif_all[1], jif_all[0], jif_all[2]].squeeze(1).to(device)
            batch_alphas = mask_frames.split(samples_batch, dim=0)
        
        # init results
        rec_masks = np.zeros((num_of_maps, *texture_size), dtype=np.uint8)
        xyts = list()
        alphas = list()
        uvs = list()
        residuals = list()
        rgbs = list()

        # run eval: split by batch size
        pbar = tqdm(range(number_of_frames), 'Generating')
        progress = 0
        print(len(batch_xyt), samples_batch)
        for idx in range(len(batch_xyt)):
            now_xyt = batch_xyt[idx].to(device)
            progress += len(now_xyt) * number_of_frames
            if pbar.n != int(progress / len(rec_xyt)):
                pbar.update(int(progress / len(rec_xyt)) - pbar.n)

            xyts.append(now_xyt.cpu().numpy())
            
            # rec_alpha = model_alpha(now_xyt)
            # rec_maps, rec_residuals, rec_rgbs = zip(*[i(now_xyt, True, True) for i in model_F_mappings])
            
            output = model(now_xyt, emb='saved')
            rec_maps, rec_residuals, rec_rgbs, rec_alpha = output['uv'], output['residual'], output['rgb'],output['alpha']
            if rec_alpha is not None:
                alphas.append(rec_alpha.cpu().numpy())
            else: 
                alphas.append(batch_alphas[idx].cpu().unsqueeze(0).numpy())
            
            # print(rec_maps[0].shape, alphas[0].shape)
            
            uvs.append(np.stack([i.cpu().numpy() for i in rec_maps]))
            residuals.append(np.stack([i.cpu().numpy() for i in rec_residuals]))
            rgbs.append(np.stack([i.cpu().numpy() for i in rec_rgbs]))
            rec_idxs = [np.clip(np.floor((i * 0.5 + 0.5).cpu().numpy() * 1000).astype(np.int64), 0, 999) for i in rec_maps]
            for i in range(num_of_maps):
                _idx = np.stack((rec_idxs[i][:,:, 1], rec_idxs[i][:,:, 0]))
                for d in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    _idx_now = _idx + np.array(d)[:, None, None] # 2,1,10000  2,1
                    _idx_now[0] = np.clip(_idx_now[0], 0, texture_size[1]-1)
                    _idx_now[1] = np.clip(_idx_now[1], 0, texture_size[0]-1)
                    mask_now = (alphas[-1][..., i] * 255).astype(np.uint8)
                    mask_now = np.max((mask_now, rec_masks[i][_idx_now[0],_idx_now[1]]), axis=0)
                    rec_masks[i][_idx_now[0], _idx_now[1]] = mask_now
        pbar.close()

    # re-split the data by frame number
    xyts = np.split(np.concatenate(xyts, axis=1), number_of_frames, axis=1)
    alphas = np.split(np.concatenate(alphas, axis=1), number_of_frames, axis=1)
    uvs = np.split(np.concatenate(uvs, axis=2), number_of_frames, axis=2)
    residuals = np.split(np.concatenate(residuals, axis=2), number_of_frames, axis=2)
    rgbs = np.split(np.concatenate(rgbs, axis=2), number_of_frames, axis=2)
    return rec_masks, xyts, alphas, uvs, residuals, rgbs



def eval(model, index,
        jif_all, video_frames, gt_masks, 
        number_of_frames, rez, samples_batch,
        num_of_maps, save_dir, device, writer, iteration):
    
    print('Start evaluation.')
    os.makedirs(save_dir, exist_ok=True)
    texture_size = (1000, 1000)
    with torch.no_grad():
        # init results
        texture_maps = list()
        _m = np.array(Image.open('checkerboard.png'))[..., :3] # canviar
        # _m = np.array(Image.open('/ghome/mpilligua/video_editing/hypersprites-no-residual-no-hyphash/checkerboard.png'))[..., :3]
        edit_textures = [_m for _ in range(num_of_maps)]
        # generate necessary evaluation components
        rec_masks, xyts, alphas, uvs, residuals, rgbs = eval_data_gen(model, index, 
                                        jif_all, number_of_frames, rez, 
                                        samples_batch, num_of_maps, 
                                        texture_size, device, mask_frames = gt_masks)
        # print(residuals)
        # exit(0)

        # write results
        print('Synthesizing: textures')
        grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, texture_size[1]), torch.linspace(-1, 1, texture_size[0])) #, indexing='ij')
        grid_xy = torch.hstack((grid_y.reshape(-1, 1), grid_x.reshape(-1, 1))).unsqueeze(0).to(device)
        
        # output = model(grid_xy, index)
        params = model.params
        for i in range(num_of_maps):
            # print(get_subdict(params,f'layer_{i}'))
            texture_map = getattr(model.net, f'layer_{i}').texture(grid_xy.detach(), params=get_subdict(params,f'layer_{i}'))
            texture_map = texture_map.detach().cpu().numpy().reshape(*texture_size, 3)
            texture_map = (texture_map * 255).astype(np.uint8)
            texture_maps.append(texture_map)
            Image.fromarray(np.concatenate((texture_map, rec_masks[i][..., None]), axis=-1)).save(os.path.join(save_dir, 'tex%d.png'%i))

        # print(len(alphas), alphas[0].shape)
        print('Synthesizing: alpha masks')
        _write_alpha(save_dir, alphas, video_frames.shape[:2], num_of_maps, iteration, save_video=True, tb_writer=writer)

        print('Synthesizing: residuals')
        _write_residual(save_dir, residuals, video_frames.shape[:2], num_of_maps, iteration, tb_writer=writer)

        print('Synthesizing: reconstruction videos')
        psnr, psnr_no_residual, iou = _write_video(save_dir, rgbs, residuals, alphas, video_frames.cpu().numpy(), gt_masks, num_of_maps, iteration, tb_writer=writer)

        print('Synthesizing: edited videos')
        _write_edited(save_dir, texture_maps, edit_textures, 0.5, uvs, residuals, alphas, video_frames.shape[:2])

    return psnr, psnr_no_residual, iou


def _write_alpha(save_dir, alphas, video_size, num_layers, iteration, save_video=True, tb_writer=None):
    if save_video:
        writers = [imageio.get_writer(os.path.join(save_dir, 'alpha%d.mp4'%i), fps=10) for i in range(num_layers)]
        for alpha in alphas:
            alpha = alpha.reshape(*video_size, num_layers)
            for i in range(num_layers):
                writers[i].append_data((alpha[..., [i]] * 255).astype(np.uint8))
        for i in writers: i.close()
    else:
        for frame_id, alpha in enumerate(alphas):
            alpha = alpha.reshape(*video_size, num_layers)
            for obj_id in range(num_layers):
                cv2.imwrite(os.path.join(save_dir,"{:2d}_{:05d}.png".format(obj_id, frame_id)),
                             (alpha[..., [obj_id]] * 255).astype(np.uint8))
    
    if tb_writer:
        # alphas_vis = np.concatenate(alphas)
        # alphas_vis = alphas_vis.reshape(len(alphas), *video_size, num_layers)
        # alphas_vis = torch.from_numpy(alphas_vis).unsqueeze(0).permute(0,1,4,2,3)
        # for i in range(num_layers-1):
        #     tb_writer.add_video('evaluation/alpha_%d'%i, alphas_vis[:,:,i:i+1,:,:], global_step=iteration, fps=10)

        alpha_vis = alphas[0]
        alpha_vis = alpha_vis.reshape( *video_size, num_layers)
        alpha_vis = torch.from_numpy(alpha_vis)
        for i in range(num_layers-1):
            tb_writer.add_image('evaluation/alpha_%d'%i, alpha_vis[...,i], global_step=iteration, dataformats='HW')


def _write_residual(save_dir, residuals, video_size, num_layers, iteration, tb_writer):
    writers = [imageio.get_writer(os.path.join(save_dir, 'residual%d.mp4'%i), fps=10) for i in range(num_layers)]
    for residual in residuals:
        for i in range(num_layers):
            writers[i].append_data(np.clip(residual[i]*128, 0, 255).astype(np.uint8).reshape(*video_size, 3))
    for i in writers: i.close()
    if tb_writer:
        res_vis = residuals[0].reshape(-1, *video_size, 3)
        res_vis = torch.from_numpy(np.clip(res_vis*128, 0, 255).astype(np.uint8))
        for i in range(num_layers):
            tb_writer.add_image('evaluation/residual_%d'%i, res_vis[i,...], global_step=iteration, dataformats='HWC')


def _write_video(save_dir, rgbs, residuals, alphas, video_frames, gt_masks, num_layers, iteration, write_compare=True, tb_writer=None):
    writer = imageio.get_writer(os.path.join(save_dir, 'rec.mp4'), fps=10)
    writer_no_residual = imageio.get_writer(os.path.join(save_dir, 'rec_no_residual.mp4'), fps=10)
    if write_compare:
        writer_compare = imageio.get_writer(os.path.join(save_dir, 'comp.mp4'), fps=10)
    writers_error = [imageio.get_writer(os.path.join(save_dir, 'maskerror%d.mp4'%i), fps=10) for i in range(num_layers)]

    psnr = np.zeros((len(rgbs), 1))
    psnr_no_residual = np.zeros((len(rgbs), 1))

    num_object = gt_masks.shape[-1]
    iou = np.zeros((len(rgbs), num_object))
    # print(gt_masks.size())
    

    for t, (rgb, residual, alpha) in enumerate(zip(rgbs, residuals, alphas)):
        output_res = sum([(rgb[i] * residual[i]) * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1])
        writer.append_data((np.clip(output_res, 0, 1) * 255).astype(np.uint8))
        psnr[t] = skimage.metrics.peak_signal_noise_ratio(
            video_frames[:, :, :, t],
            output_res,
            data_range=1)
        output = sum([rgb[i] * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1])
        writer_no_residual.append_data((np.clip(output, 0, 1) * 255).astype(np.uint8))
        psnr_no_residual[t] = skimage.metrics.peak_signal_noise_ratio(
            video_frames[:, :, :, t],
            output,
            data_range=1)
        
        if tb_writer:
            for i in range(num_layers):
                texture_vis = (rgb[i] * residual[i]) * alpha[..., [i]]
                texture_vis = texture_vis.reshape(video_frames.shape[:-1])
                texture_vis = torch.from_numpy(np.clip(texture_vis, 0, 1))
                tb_writer.add_image('evaluation/texture_%d'%i, texture_vis, global_step=iteration, dataformats='HWC')
        
        # print(alpha.shape)
        if not gt_masks.all():
            alpha_map = alpha.reshape((video_frames.shape[0], video_frames.shape[1], -1))
            # print(alpha_map.shape, gt_mask.shape)
            for i in range(num_object):
                mask_pred = np.zeros_like(alpha_map[:,:,i])
                mask_pred[alpha_map[:,:,i] > 0.5] = 1
                mask_gt = gt_masks[:,:,t,i]

                iou[t,i] = compute_multiple_iou(mask_pred, mask_gt)
                error_map = np.zeros_like(mask_pred)
                error_map = np.where(mask_pred!=mask_gt, 1, 0)
                writers_error[i].append_data((error_map*255).astype(np.uint8))
            
        if write_compare:
            comp = np.empty((output.shape[0]*2, output.shape[1]*2, 3))
            # GT: top-left
            comp[:output.shape[0], :output.shape[1]] = video_frames[..., t]
            # w/ residual: top-right
            comp[:output.shape[0], output.shape[1]:] = output_res
            # w/o residual: bottom-left
            comp[output.shape[0]:, :output.shape[1]] = output
            # residual only: bottom-right
            comp[output.shape[0]:, output.shape[1]:] = sum([residual[i] * alpha[..., [i]] for i in range(num_layers)]).reshape(video_frames.shape[:-1]) * 0.5
            writer_compare.append_data((np.clip(comp, 0, 1) * 255).astype(np.uint8))
    
    writer.close()
    writer_no_residual.close()
    if write_compare: writer_compare.close()
    for i in writers_error: i.close()

    return psnr.mean(), psnr_no_residual.mean(), iou.mean(axis=0)



def _write_edited(save_dir, maps1, maps2, ratio, uvs, residuals, alphas, video_size):
    # writer_all = imageio.get_writer(os.path.join(save_dir, 'edited_all.mp4'), fps=10)
    writer_all_residual = imageio.get_writer(os.path.join(save_dir, 'edited_all_+residual.mp4'), fps=10)
    # rgb_all = np.zeros((len(uvs), *video_size, 3))
    residual_all = np.zeros((len(uvs), *video_size, 3))
    for i in range(len(maps1)):
        im = np.clip(np.where(
            maps2[i], maps1[i] * ratio + maps2[i] * (1-ratio), maps1[i]
        ), 0, 255)
        # writer = imageio.get_writer(os.path.join(save_dir, 'edited%d.mp4'%i), fps=10)
        writer_residual = imageio.get_writer(os.path.join(save_dir, 'edited%d+residual.mp4'%i), fps=10)
        for t, (uv, alpha) in enumerate(zip(uvs, alphas)):
            rgb = bilinear_interpolate_numpy(im, (uv[i][:,:, 0]*0.5+0.5)*im.shape[1], (uv[i][:,:, 1]*0.5+0.5)*im.shape[0])
            # writer.append_data((rgb).reshape(*video_size, 3).astype(np.uint8))
            writer_residual.append_data(np.clip((rgb * residuals[t][i]), 0, 255).reshape(*video_size, 3).astype(np.uint8))
            # rgb_all[t] += (rgb*alpha[..., [i]]).reshape(*video_size, 3)
            residual_all[t] += ((rgb * residuals[t][i]) * alpha[..., [i]]).reshape(*video_size, 3)
        # writer.close()
        writer_residual.close()
    for t in range(len(uvs)):
        # writer_all.append_data(rgb_all[t].astype(np.uint8))
        writer_all_residual.append_data(np.clip(residual_all[t], 0, 255).astype(np.uint8))
    # writer_all.close()
    writer_all_residual.close()



