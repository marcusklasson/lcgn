import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg

class GraphAttentionLayer(nn.Module):
    """
    GAT layer, from to https://arxiv.org/abs/1710.10903
    see Pytorch code: https://github.com/Diego999/pyGAT/blob/master/models.py
    """

    def __init__(self, in_features, out_features, dropout=0.0, concat=False, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, x_loc, entity_num):
        h = torch.matmul(x_loc, self.W) #torch.mm(x_loc, self.W)
        N = h.size()[1]

        
        #a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=2).view(-1, N, N, 2*self.out_features)
        #e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        #zero_vec = -9e15*torch.ones_like(e)
        #attention = torch.where(entity_num.double() > 0, e, zero_vec)
        #attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        attention = F.softmax(e, dim=2)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.build_loc_ctx_init()

        self.gat_layer = GraphAttentionLayer(in_features=cfg.CTX_DIM,
                             out_features=cfg.CTX_DIM, concat=False)

    def build_loc_ctx_init(self):
        assert cfg.STEM_LINEAR != cfg.STEM_CNN
        if cfg.STEM_LINEAR:
            self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)
            self.x_loc_drop = nn.Dropout(1 - cfg.stemDropout)
        elif cfg.STEM_CNN:
            self.cnn = nn.Sequential(
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.D_FEAT, cfg.STEM_CNN_DIM, (3, 3), padding=1),
                nn.ELU(),
                nn.Dropout(1 - cfg.stemDropout),
                ops.Conv(cfg.STEM_CNN_DIM, cfg.CTX_DIM, (3, 3), padding=1),
                nn.ELU())

    def forward(self, images, batch_size, entity_num):
        x_loc = self.loc_init(images)
        x_out = self.gat_layer(x_loc, entity_num)
        #x_out = self.propagate_message(x_loc, entity_num)
        return x_out

    def loc_init(self, images):
        if cfg.STEM_NORMALIZE:
            images = F.normalize(images, dim=-1)
        if cfg.STEM_LINEAR:
            x_loc = self.initKB(images)
            x_loc = self.x_loc_drop(x_loc)
        elif cfg.STEM_CNN:
            images = torch.transpose(images, 1, 2)  # N(HW)C => NC(HW)
            x_loc = images.view(-1, cfg.D_FEAT, cfg.H_FEAT, cfg.W_FEAT)
            x_loc = self.cnn(x_loc)
            x_loc = x_loc.view(-1, cfg.CTX_DIM, cfg.H_FEAT * cfg.W_FEAT)
            x_loc = torch.transpose(x_loc, 1, 2)  # NC(HW) => N(HW)C
        if cfg.STEM_RENORMALIZE:
            x_loc = F.normalize(x_loc, dim=-1)

        return x_loc
