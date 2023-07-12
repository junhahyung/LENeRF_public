import torch
import dnnlib
from torch.nn import Module
from training.networks_stylegan2 import MappingNetwork


class Mapper(Module):
    def __init__(self, num_layers, z_dim=512, c_dim=0, w_dim=512, **kwargs):
        super().__init__()
        if 'layer_features' in kwargs:
            layer_features = kwargs['layer_features']
        else:
            layer_features = None

        self.net = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=None, num_layers=num_layers, layer_features=layer_features)
        self.net.requires_grad_(False)
        last_layer = list(self.net.modules())[-1]
        last_layer.weight.mul_(0.01)
        self.net.requires_grad_(True)


    def forward(self, x):
        out = self.net(x, None)
        return out


'''
class VanillaMapper(Module):
    def __init__(self, **opts):
        super().__init__()

        self.opts = dnnlib.EasyDict(opts)

        self.opts.z_dim = 512 * self.opts.num_ws
        self.opts.w_dim = 512 * self.opts.num_ws
        self.opts.layer_features = self.opts.hidden_dim
        self.mapping = Mapper(**self.opts)

    def forward(self, x):
        # x: (b, num_ws, 512)
        bs = x.shape[0]
        num_ws = x.shape[1]

        x = x.view(bs, -1)
        out = self.mapping(x)
        out = out.reshape(bs, num_ws, -1)

        return out
'''


class LevelsMapper(Module):

    def __init__(self, **opts):
        super(LevelsMapper, self).__init__()

        opts = dnnlib.EasyDict(opts)
        self.opts = opts
        self.mapper_type = opts.mapper_type
        self.num_ws = opts.num_ws

        if not opts.no_coarse_mapper:
            self.course_mapping = Mapper(**opts)
        if not opts.no_medium_mapper:
            self.medium_mapping = Mapper(**opts)
        if not opts.no_fine_mapper:
            self.fine_mapping = Mapper(**opts)

        if self.mapper_type == 'ec':
            self.ec = EmbedCond(self.num_ws)
            self.embed = None

        self.start_fine = self.opts.get('start_fine', 8)
        print(f'====start_fine: {self.start_fine}====')
        self.end_fine = self.opts.get('end_fine', self.num_ws) # 14 == use all
        print(f'====end_fine: {self.end_fine}====')
        print(f'====num_layers: {self.num_ws}====')

    def forward(self, x):
        if self.mapper_type == 'ec':
            x = self.ec(self.embed, x)

        bs = x.shape[0]
        feat_dim = x.shape[-1]

        x_coarse = x[:, :4, :].reshape(-1, feat_dim)
        x_medium = x[:, 4:self.start_fine, :].reshape(-1,feat_dim)
        x_fine = x[:, self.start_fine:self.end_fine, :].reshape(-1, feat_dim)

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)

        out_feat_dim = x_coarse.shape[-1]
        x_coarse = x_coarse.reshape(bs, -1, out_feat_dim)
        x_medium = x_medium.reshape(bs, -1, out_feat_dim)
        x_fine = x_fine.reshape(bs, -1, out_feat_dim)

        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        if self.end_fine != self.num_ws:
            assert out.shape[1] == self.end_fine
            out = torch.cat([out, torch.zeros_like(out[:,:self.num_ws-self.end_fine,:])], dim=1)
            assert out.shape[1] == self.num_ws

        return out


class EmbedCond(Module):
    def __init__(self, num_ws):
        super().__init__()
        self.opts = dnnlib.EasyDict()
        self.opts.num_layers = 4
        self.mappers = torch.nn.ModuleList()

        for i in range(num_ws):
            self.mappers.append(Mapper(**self.opts))

        self.final_opts = dnnlib.EasyDict()
        self.final_opts.num_layers = 1
        self.final_opts.z_dim = 1024

        self.final = Mapper(**self.final_opts)

    def forward(self, emb, ws):
        out_list = []
        bs, nws, _ = ws.shape
        for mapper in self.mappers:
            out = mapper(emb)
            out_list.append(out)

        out = torch.stack(out_list, dim=1)

        out = out.repeat(bs, 1, 1)
        out = torch.cat([out, ws], dim=-1) # (bs, nws, 1024)
        out = self.final(out.reshape(bs*nws, -1)).reshape(bs, nws, -1) # (bs, nws, 512)

        return out
