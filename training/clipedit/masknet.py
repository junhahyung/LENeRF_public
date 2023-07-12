import torch
import torch.nn as nn
from training.networks_stylegan2 import FullyConnectedLayer, normalize_2nd_moment


class Mask_mlp(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, thres=0.0, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        '''
        self.network = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid())
        '''

        if kwargs['masknet_type'] == 'ws_condition':
            self.ws_net = nn.Sequential(
                    FullyConnectedLayer(512, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, hidden_dim//2, activation='lrelu'))

            self.network = nn.Sequential(
                    FullyConnectedLayer(feat_dim + hidden_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, 1, activation='linear'),
                    #FullyConnectedLayer(hidden_dim, 1, activation='sigmoid')
                    )
        else:
            self.network = nn.Sequential(
                    FullyConnectedLayer(feat_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, hidden_dim, activation='lrelu'),
                    FullyConnectedLayer(hidden_dim, 1, activation='linear')
                    #FullyConnectedLayer(hidden_dim, 1, activation='sigmoid')
                    )


        self.thres = thres
        self.sigmoid = nn.Sigmoid()

        '''
        self.inv_thres = kwargs.get('inv_thres', None)
        if self.inv_thres:
            print('=== use inv thres ===')
        '''

        self.sigmoid_beta = kwargs.get('sigmoid_beta', None)


    def forward(self, x, ws=None, ws_edit=None):
        x = normalize_2nd_moment(x)

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
            x = torch.cat([x, ws_feat, ws_edit_feat], dim=-1)

        out = self.network(x)
        if self.sigmoid_beta is not None:
            out = self.sigmoid_beta*out
        out = self.sigmoid(out)

        # 이거 1-out을 왜 해줘야할지???
        out = 1. - out
        '''
        if self.inv_thres:
            ones = torch.ones_like(out)
            out = torch.where(out>self.inv_thres, ones, out)
        '''

        if not self.training:
            #out = (out > self.thres).long()
            zeros = torch.zeros_like(out)
            out = torch.where(out>self.thres, out, zeros)
        return out


class Mask_attention(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, thres=0.0, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        assert kwargs['masknet_type'] == 'attention'

        self.feat_in = FullyConnectedLayer(feat_dim, hidden_dim, activation='lrelu')
        self.ws_in = FullyConnectedLayer(512, hidden_dim, activation='lrelu')
        self.fuse = FullyConnectedLayer(hidden_dim*2, hidden_dim, activation='lrelu')
        self.sa_net = torch.nn.MultiheadAttention(hidden_dim,1,batch_first=True)
        '''
        self.final = nn.Sequential(
                FullyConnectedLayer(hidden_dim*2, hidden_dim, activation='lrelu'),
                FullyConnectedLayer(hidden_dim, 1, activation='sigmoid'),
        )
        '''
        self.final = nn.Sequential(
                FullyConnectedLayer(hidden_dim*2, 1, activation='sigmoid'),
        )


        self.thres = thres
        self.feat_dim = feat_dim
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.sigmoid_beta = kwargs.get('sigmoid_beta', None)

        '''
        self.inv_thres = kwargs.get('inv_thres', None)
        if self.inv_thres:
            print('=== use inv thres ===')
        '''

    def forward(self, x, x_edit, ws, ws_edit):
        x = normalize_2nd_moment(x)
        x_edit = normalize_2nd_moment(x_edit)
        bs, l, _ = ws.shape
        all_b, _ = x.shape
        assert all_b % bs == 0

        x = torch.stack([x, x_edit], dim=1) #(all_b,2,feat_dim)
        x = self.feat_in(x.view(-1, self.feat_dim))

        ws = torch.stack([ws, ws_edit], dim=1) # (b,2,14,512)
        ws = self.ws_in(ws.view(-1, 512)).view(bs, 2, l, self.hidden_dim).mean(dim=2) # (bs, 2, hidden_dim)
        ws = ws.unsqueeze(1).repeat(1,all_b//bs, 1,1).reshape(-1, self.hidden_dim) #(all_b*2, hidden_dim)

        x = self.fuse(torch.cat([x, ws], dim=-1)).view(all_b, 2, self.hidden_dim)

        x = self.lrelu(self.sa_net(x,x,x)[0])

        out = self.final(x.reshape(all_b, -1))

        # 이거 1-out을 왜 해줘야할지???
        out = 1. - out

        '''
        if self.inv_thres:
            ones = torch.ones_like(out)
            out = torch.where(out>self.inv_thres, ones, out)
        '''

        if not self.training:
            #out = (out > self.thres).long()
            zeros = torch.zeros_like(out)
            out = torch.where(out>self.thres, out, zeros)
        return out
