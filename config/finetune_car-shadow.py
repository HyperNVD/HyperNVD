results_folder_name = 'results/'
path2DAVIS = '/data/124-1/datasets/DAVIS-2016/DAVIS/'
ckpt = './checkpoints/30_videos.pth'

maximum_number_of_frames = 25
resx = 768
resy = 432
iters_num = 80000
samples_batch = 150000
learning_rate = 0.0001

sequences = ['car-shadow']
uv_mapping_scales = [0.9, 0.9, 0.9, 0.6]
folder_suffix = 'finetune-car-shadow'
seperate_obj_gtmask = False
seed=42
finetune = True

logger = dict(                              
    period = 500,
    log_time = True,
    log_loss = True,
    log_alpha = True)

use_alpha_gt = False

evaluation = dict(
    interval = 250,
    save_interval = 1000,
    samples_batch = 150000)

losses = dict(
    rgb = dict(weight = 5),
    gradient = dict(weight = 1),
    sparsity = dict(weight = 1),
    alpha_bootstrapping = dict(weight = 2,
                          stop_iteration = 500),
    alpha_reg = dict(weight = 0.1),
    flow_alpha = dict(weight = 0.05),
    optical_flow = dict(weight = 0.01),
    rigidity = dict(weight = 0.001,
                    derivative_amount = 1),
    global_rigidity = dict(weight = [0.005, 0.05],
                        stop_iteration = 5000,
                        derivative_amount = 100),
    residual_reg = dict(weight = 0.5),
    residual_consistent = dict(weight = 0.1))

config_xyt = {
    'otype': 'HashGrid',
    'n_levels': 16,
    'per_level_scale': 1.25,
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2, # level_dim
    'desired_resolution': 2048,}

config_uv = {
    'otype': 'HashGrid',
    'n_levels': 16,
    'per_level_scale': 1.25,
    'base_resolution': 16,
    'log2_hashmap_size': 15,
    'n_features_per_level': 2,
    'desired_resolution': 2048,}

model_mapping = [{
    'pretrain': True,
    'texture': {
        'encoding_config': config_uv,
        'num_layers': 4,
        'num_hidden_neurons': 64,
        },
    'residual': {
        'encoding_config': config_xyt,
        'num_layers': 4,
        'num_hidden_neurons': 64,
        },
    'encoding_config': None,
    'num_layers': 4,
    'num_hidden_neurons': 256,
}, {
    'pretrain': True,
    'texture': {
        'encoding_config': config_uv,
        'num_layers': 4,
        'num_hidden_neurons': 64,
        },
    'residual': {
        'encoding_config': config_xyt,
        'num_layers': 4,
        'num_hidden_neurons': 64,
        },
    'encoding_config': None,
    'num_layers': 4,
    'num_hidden_neurons': 256,
}]

alpha = {
    'encoding_config': config_xyt,
    'num_encoding_out': 32,
    'num_layers': 4,
    'num_hidden_neurons': 64,}

hypernet={
    'num_hidden_neurons': 128, 
    'num_layers': 1, 
    'hn_in': 768, 
    'activation_type': 'relu'
}

prior={
    'clip_hidden_dim': 128, 
}
