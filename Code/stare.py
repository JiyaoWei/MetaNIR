import torch
import numpy as np
import torch.nn.functional as F

from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MessagePassing
import logging
import logging.config
from torch import nn

import inspect
import torch_scatter
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

np.set_printoptions(precision=4)


class MessagePassing(torch.nn.Module):

    def __init__(self):
        super(MessagePassing, self).__init__()

    def propagate(self, edge_index, aggr='add', x=None, edge_type=None,
                                    rel_embed=None, edge_norm=None, mode=None,
                                    ent_embed=None, qualifier_ent=None,
                                    qualifier_rel=None,qual_index=None):

        assert aggr in ['add', 'mean', 'max']

        size = x.size(0)
        message_args = [x[edge_index[1]], edge_type, rel_embed, edge_norm, mode, ent_embed, qualifier_ent, qualifier_rel, qual_index]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size) 

        return out


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    # return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
    return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), n=a.shape[-1])

def rotate(h, r):
    # re: first half, im: second half
    # assume embedding dim is the last dimension
    d = h.shape[-1]
    h_re, h_im = torch.split(h, d // 2, -1)
    r_re, r_im = torch.split(r, d // 2, -1)
    return torch.cat([h_re * r_re - h_im * r_im,
                        h_re * r_im + h_im * r_re], dim=-1)


def scatter_(name, src, index, dim_size=None):

    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name == 'max' else 0
    out = op(src, index, 0, None, dim_size)
    # out = op(src, index, 0, None, dim_size, fill_value)
    if isinstance(out, tuple):
        out = out[0]

    if name == 'max':
        out[out == fill_value] = 0

    return out

class StarEConv(MessagePassing):
    """ The important stuff. """

    def __init__(self, in_channels, out_channels, act=lambda x: x, device='cpu'):
        super(self.__class__, self).__init__()

        self.act = act
        self.device = device

        self.loop_w = self.get_param((in_channels, out_channels))  # (100,200)
        self.in_w = self.get_param((in_channels, out_channels))  # (100,200)
        self.out_w = self.get_param((in_channels, out_channels))  # (100,200)
        self.w_rel = self.get_param((in_channels, out_channels))  # (100,200)
        self.loop_rel = self.get_param((1, in_channels))  # (1,100)
        
        self.qual_aggregate= 'sum' # self.p['STAREARGS']['QUAL_AGGREGATE']
        self.opn = 'sub' # self.p['STAREARGS']['OPN']
        self.qual_opn = 'sub' # self.p['STAREARGS']['QUAL_OPN']
        self.qual_n = 'mean' # self.p['STAREARGS']['QUAL_N']
        self.triple_qual_weight = 0.8 # self.p['STAREARGS']['TRIPLE_QUAL_WEIGHT']
        
        if self.qual_aggregate== 'sum' or self.qual_aggregate== 'mul':
            self.w_q = self.get_param((in_channels, in_channels))  # new for quals setup
        elif self.qual_aggregate== 'concat':
            self.w_q = self.get_param((2 * in_channels, in_channels))  # need 2x size due to the concat operation

        self.gcn_drop = 0.1 # self.p['STAREARGS']['GCN_DROP']
        self.drop = torch.nn.Dropout(self.gcn_drop)
        self.bn = torch.nn.BatchNorm1d(out_channels)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape)).to(self.device)
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    # 如果只对x进行投影，效果会逐渐提升，但非常的慢
    def forward(self, x, edge_index, edge_type, rel_embed, quals=None):
        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

        num_quals = quals.size(1) // 2
        self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:]
        self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]
        self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:]

        self.in_norm = self.compute_norm(self.in_index, num_ent)
        self.out_norm = self.compute_norm(self.out_index, num_ent)

        in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                qualifier_rel=self.in_index_qual_rel,
                                qual_index=self.quals_index_in)

        out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                    rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                    ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                    qualifier_rel=self.out_index_qual_rel,
                                    qual_index=self.quals_index_out)
        loop_res = torch.mm(self.rel_transform(x, self.loop_rel), self.loop_w) / 3
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        # out = self.bn(out)
        return self.act(out), torch.mm(rel_embed, self.w_rel)


    def rel_transform(self, ent_embed, rel_embed):
        if self.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed.expand_as(ent_embed))
        elif self.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.opn == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return trans_embed

    def qual_transform(self, qualifier_ent, qualifier_rel):

        if self.qual_opn == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.qual_opn == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.qual_opn == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.qual_opn == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel)
        else:
            raise NotImplementedError

        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):

        if self.qual_aggregate== 'sum':
            qualifier_emb = torch.einsum('ij,jk -> ik', self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]), self.w_q)
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb      # [N_EDGES / 2 x EMB_DIM]
        elif self.qual_aggregate== 'concat':
            qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_q)                         # [N_EDGES / 2 x EMB_DIM]
        elif self.qual_aggregate== 'mul':
            qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1), self.w_q)
            return rel_part_emb * qualifier_emb
        else:
            raise NotImplementedError

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):

        # Step 1: embedding
        qualifier_emb_rel = rel_embed[qualifier_rel]
        qualifier_emb_ent = ent_embed[qualifier_ent]
        rel_part_emb = rel_embed[edge_type]
        
        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,
                                            qualifier_rel=qualifier_emb_rel)

        # Pass it through a aggregate layer
        return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.triple_qual_weight, qual_index=qual_index)


    def message(self, x_j, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None):
        weight = getattr(self, '{}_w'.format(mode))

        rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                qualifier_rel, edge_type, qual_index)
        # rel_emb = rel_embed[edge_type]
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)


    @staticmethod
    def compute_norm(edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()  # Identity matrix where we know all entities are there
        # 每个实体在row位置上出现的次数
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5}
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability
        
        norm = deg_inv[row] * edge_weight * deg_inv[col]  # Norm parameter D^{-0.5} *

        return norm

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        if self.qual_n == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges)
        elif self.qual_n == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output