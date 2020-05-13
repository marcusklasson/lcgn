import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import ops as ops
from .config import cfg


class SpatialGNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.build_loc_ctx_init()
        #self.build_extract_textual_command()
        self.build_propagate_message()

        adj_matrix = build_adjacency_matrix(cfg.W_FEAT, cfg.H_FEAT) # defines neighbors in graph
        self.adj = torch.from_numpy(adj_matrix.astype(np.float32)).cuda()


    def build_propagate_message(self):
        self.W = ops.Linear(cfg.CTX_DIM, cfg.CTX_DIM, bias=False)
        self.initKB = ops.Linear(cfg.D_FEAT, cfg.CTX_DIM)

    def forward(self, images, batch_size, entity_num):
        x_loc = self.loc_init(images)
        support = self.W(x_loc)
        x_out = torch.matmul(self.adj, support)
        #x_out = self.propagate_message(x_loc, entity_num)
        return x_out

    def loc_init(self, images):
        x_loc = self.initKB(images)
        return x_loc

def build_adjacency_matrix(nx, ny, self_loop=True, neighbors=4):
    """ Building adjacency matrix
    Only supports 4-neighborhood.
    TODO: Support for 8 neighborhood
    """

    if self_loop:
        adj_matrix = np.eye((nx*ny), dtype=np.int64)
    else:
        adj_matrix = np.zeros((nx*ny, nx*ny), dtype=np.int64)

    x = np.linspace(0, nx-1, nx, dtype=np.int64)
    y = np.linspace(0, ny-1, ny, dtype=np.int64)
    xv, yv = np.meshgrid(x, y)
    grid = np.concatenate((np.expand_dims(yv, axis=2), np.expand_dims(xv, axis=2)), axis=-1)

    adj_row_idx = 0
    for i in range(nx):
        for j in range(ny):
            
            if i > 0: # check if upper 
                neighbor_idx = j + (i-1)*nx # row-major order
                adj_matrix[adj_row_idx, neighbor_idx] = 1

            if i < (nx-1): # check if lower
                neighbor_idx = j + (i+1)*nx # row-major order
                adj_matrix[adj_row_idx, neighbor_idx] = 1

            if j > 0: # check if left
                neighbor_idx = (j-1) + i*nx # row-major order
                adj_matrix[adj_row_idx, neighbor_idx] = 1

            if j < (ny-1): # check if right
                neighbor_idx = (j+1) + i*nx # row-major order
                adj_matrix[adj_row_idx, neighbor_idx] = 1

            # increment row in adjacency matrix
            adj_row_idx += 1

    return adj_matrix