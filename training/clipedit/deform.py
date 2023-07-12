import torch
import torch.nn as nn
from training.networks_stylegan2 import FullyConnectedLayer, normalize_2nd_moment


class Deform_mlp(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, thres=0.0, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.feat_dim = feat_dim
        assert feat_dim == 3

        self.ws_net = nn.Sequential(
                FullyConnectedLayer(512, hidden_dim, activation='lrelu'),
                FullyConnectedLayer(hidden_dim, hidden_dim//2, activation='lrelu'))

        self.network = nn.Sequential(
                FullyConnectedLayer(feat_dim + hidden_dim, hidden_dim, activation='lrelu'),
                FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                FullyConnectedLayer(hidden_dim, feat_dim, activation='linear'),
                #FullyConnectedLayer(hidden_dim, 1, activation='sigmoid')
                )
        self.network.requires_grad_(False)
        last_layer = list(self.network.modules())[-1]
        last_layer.weight.mul_(0.01)
        self.network.requires_grad_(True)


    def forward(self, x, ws=None, ws_edit=None):
        x = x.reshape(-1, self.feat_dim)
        if ws is not None:
            assert ws_edit is not None
            b, l, dim = ws.shape
            all_b, _ = x.shape
            assert all_b % b == 0

            ws = ws.view(-1, dim)
            ws_edit = ws_edit.view(-1, dim)
            ws_feat = self.ws_net(ws).reshape(b, l, self.hidden_dim//2).mean(dim=1) # (b, hidden_dim)
            ws_feat = ws_feat.unsqueeze(1).repeat(1,all_b//b,1).reshape(-1, self.hidden_dim//2)
            ws_edit_feat = self.ws_net(ws_edit).reshape(b, l, self.hidden_dim//2).mean(dim=1) # (b, hidden_dim)
            ws_edit_feat = ws_edit_feat.unsqueeze(1).repeat(1,all_b//b,1).reshape(-1, self.hidden_dim//2)
            out = torch.cat([x, ws_feat, ws_edit_feat], dim=-1)

        out = self.network(out)
        # add residual
        out = out + x
        out = out.reshape(b, all_b//b, self.feat_dim)

        # 이거 1-out을 왜 해줘야할지???
        return out
