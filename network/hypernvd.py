import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaSequential, MetaLinear

from .metamodulesMAE import FCBlock, BatchLinear, HyperNetwork, get_subdict
from .encoding import get_encoder
from torchmeta.modules import MetaModule


class HyperVDN(nn.Module):
    '''
    Module class for HyP-video decomposition model
    '''
    def __init__(self,
                 mappings_cfg,
                 alpha_cfg,
                 hypernet_cfg,
                 prior_cfg,
                 num_instances=1, 
                 bound=1,
                 device='cuda',
                 clip_mapping=False,
                 std=0.01,
                 ):
        super().__init__()

        self.bound = bound

        # video decomposition network
        self.net = VDNetwork(mappings_cfg,
                            alpha_cfg,
                            num_mlp_layers=4,
                            hidden_dim=256,
                            bound=1)

        ## hypernetwork
        self.hypernet = HyperNetwork(hyper_in_features=hypernet_cfg['hn_in'],
                                     hyper_hidden_layers=hypernet_cfg['num_layers'],
                                     hyper_hidden_features=hypernet_cfg['num_hidden_neurons'],
                                     hypo_module=self.net, 
                                     activation=hypernet_cfg['activation_type'])


    def get_params(self, emb):
        self.params  = self.hypernet(emb)
        return self.params


    def forward(self, xyt, emb, image=None, **kwargs):
        if emb == 'saved': 
            pred_params = self.params
        else: 
            pred_params = self.get_params(emb)
        pred_output = self.net(xyt, params=pred_params, **kwargs)

        return pred_output

    def get_optimizer_list(self):
        optimizer_list = list()
        optimizer_list.extend(self.hypernet.get_optimizer_list())
        optimizer_list.extend(self.net.get_optimizer_list())
        return optimizer_list


class VDNetwork(MetaModule):
    def __init__(self,
                 mappings_cfg,
                 alpha_cfg,
                 num_mlp_layers=4,
                 hidden_dim=64,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__()
        
        self.num_img_layers = len(mappings_cfg)
        self.num_mlp_layers = num_mlp_layers
        self.hidden_dim = hidden_dim

        xyt_dim = 3
        uv_dim = 2 
        rgb_dim = 3

        for i, map_cfg in enumerate(mappings_cfg):
            layer = MappingNetwork(map_cfg,
                                    xyt_dim=xyt_dim,
                                    uv_dim=uv_dim,
                                    rgb_dim=rgb_dim,
                                    bound=bound)
            setattr(self, f"layer_{i}", layer)
        

        if alpha_cfg is not None:
            self.alpha = AlphaNetwork(alpha_cfg,
                                    self.num_img_layers,
                                    xyt_dim=xyt_dim,
                                    bound=bound)
        else: 
            self.alpha = None

    def forward(self, xyt, params=None):
        uvs = []
        rgbs = []
        weights = []
        for i in range(self.num_img_layers):
            uv, rgb, weight = getattr(self,f'layer_{i}')(xyt, params=get_subdict(params,f'layer_{i}'))
            uvs.append(uv)
            rgbs.append(rgb)
            weights.append(weight)

        if self.alpha is not None:
            mask = self.alpha(xyt, params=get_subdict(params,'alpha'))
        else: 
            mask = None
        return {'uv': uvs, 'rgb': rgbs, 'residual': weights, 'alpha': mask}

    def get_optimizer_list(self):
        optimizer_list = list()
        for i in range(self.num_img_layers):
            optimizer_list.extend(getattr(self,f'layer_{i}').get_optimizer_list())
        
        optimizer_list.extend(self.alpha.get_optimizer_list())
        return optimizer_list

class MappingNetwork(MetaModule):
    def __init__(self,
                 map_cfg,
                 xyt_dim=3,
                 uv_dim=2,
                 rgb_dim=3,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound

        text_cfg = map_cfg['texture']
        resi_cfg = map_cfg['residual']

        self.mapping_enc, self.mapping_net, self.optimizer_list_mapping = build_network(xyt_dim, 
                                                            uv_dim, 
                                                            map_cfg['num_layers'], 
                                                            map_cfg['num_hidden_neurons'], 
                                                            map_cfg['encoding_config'])
        self.texture_enc, self.texture_net, self.optimizer_list_texture = build_network(uv_dim, 
                                                            rgb_dim, 
                                                            text_cfg['num_layers'], 
                                                            text_cfg['num_hidden_neurons'], 
                                                            text_cfg['encoding_config'])
        self.residual_enc, self.residual_net, self.optimizer_list_residual = build_network(uv_dim+1, 
                                                            rgb_dim, 
                                                            resi_cfg['num_layers'], 
                                                            resi_cfg['num_hidden_neurons'], 
                                                            resi_cfg['encoding_config'])
    
    
    def forward(self, xyt, params=None):
        t = xyt[:,:,-2:-1]
        uv = self.mapping(xyt, params)
        rgb = self.texture(uv, params)
        weight = self.residual(uv, t, params)
        return uv, rgb, weight


    def mapping(self, x, params=None):
        uv = self.mapping_net(x, params=get_subdict(params,f'mapping_net'))
        uv = torch.tanh(uv.to(torch.float32)) 
        return uv
    

    def texture(self, x, params=None):
        # x: [N, 3] in [-bound, bound]
        x = self.texture_enc(x, 
                             bound=self.bound, 
                             params=get_subdict(params,f'texture_enc.embeddings'))
        rgb = self.texture_net(x, params=get_subdict(params,f'texture_net'))
        rgb = (torch.tanh(rgb.to(torch.float32)) + 1.0) * 0.5
        return rgb


    def residual(self, uv, t, params=None):
        # x: [N, 3] in [-bound, bound]      
        uvt = torch.cat([uv, t], dim=2)
        uvt = self.residual_enc(uvt, 
                              bound=self.bound, 
                              params=get_subdict(params,f'residual_enc.embeddings'))
        rgb_weight = self.residual_net(uvt, params=get_subdict(params,f'residual_net'))
        rgb_weight = F.softplus(rgb_weight).to(torch.float32)
        return rgb_weight
    

    def get_optimizer_list(self):
        optimizer_list = list()
        optimizer_list.extend(self.optimizer_list_mapping)
        optimizer_list.extend(self.optimizer_list_texture)
        optimizer_list.extend(self.optimizer_list_residual)
        return optimizer_list



class AlphaNetwork(MetaModule):
    def __init__(self,
                 alpha_cfg,
                 num_img_layers,
                 xyt_dim=3,
                 bound=1,
                 eps=1e-5,
                 **kwargs,
                 ):
        super().__init__()
        self.bound = bound
        self.eps = eps

        ### alpha network, input: xyt, output: 1
        self.alpha_enc, self.alpha_net, self.optimizer_list_alpha = build_network(xyt_dim, 
                                                        num_img_layers-1, 
                                                        alpha_cfg['num_layers'], 
                                                        alpha_cfg['num_hidden_neurons'], 
                                                        alpha_cfg['encoding_config'])

    def forward(self, x, params=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        x = self.alpha_enc(x, bound=self.bound, 
                           params=get_subdict(params,'alpha_enc.embeddings'))
        mask = self.alpha_net(x, params=get_subdict(params,'alpha_net'))
        alpha = mask.to(torch.float32)
        alpha = (torch.tanh(alpha) + 1.0) * 0.5 # normalize to [0, 1]
        alpha_hie = [alpha[:, :, [0]]]
        coeff = (1.0 - alpha[:, :, [0]])
        for i in range(1, alpha.shape[-1]):
            alpha_hie.append(alpha[:, :, [i]] * coeff)
            coeff = coeff * (1.0 - alpha[:, :, [i]])
        alpha_hie.append(coeff)
        alpha = torch.clamp(torch.cat(alpha_hie, dim=2), self.eps, 1-self.eps)

        return alpha


    def get_optimizer_list(self):
        return self.optimizer_list_alpha

def build_network(in_dim, out_dim, num_mlp_layers, hidden_dim, enc_cfg):   
    if enc_cfg is None:
        encoder = None
        in_dim_enc = in_dim

    else:
        encoding = enc_cfg['otype']
        encoder, in_dim_enc = get_encoder(input_dim=in_dim, 
                                        encoding=encoding,
                                        num_levels=enc_cfg['n_levels'], 
                                        level_dim=enc_cfg['n_features_per_level'], 
                                        base_resolution=enc_cfg['base_resolution'], 
                                        log2_hashmap_size=enc_cfg['log2_hashmap_size'], 
                                        desired_resolution=enc_cfg['desired_resolution'])
    model = MetaSequential()
    for l in range(num_mlp_layers):
        if l == 0:
            in_d = in_dim_enc
        else:
            in_d = hidden_dim
        
        if l == num_mlp_layers - 1:
            out_d = out_dim
        else:
            out_d = hidden_dim

        model.add_module(f"layer_{l}", BatchLinear(in_d, out_d, bias=False))
        if l != num_mlp_layers - 1:
            model.add_module(f"act_{l}", nn.ReLU(inplace=True))

    optimizer_list = list()
    if encoder is not None:
        optimizer_list.append({'params': encoder.parameters(), 'lr': 1e-3})
    optimizer_list.append({'params': model.parameters(), 'lr': 5*(1e-5)})
    
    return encoder, model, optimizer_list