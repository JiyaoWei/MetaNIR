import torch
from torch import nn
from torch.nn.init import xavier_normal_
import dgl
import dgl.function as fn


class EntgraphCompConv(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.1, opn='sum', device='cpu', bias="True", bn="True", activation='tanh'):
        super(EntgraphCompConv, self).__init__()
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == 'elu':
            self.act = nn.ELU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        # self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        # self.w_ent = self.get_param([in_channels, out_channels])
        # self.w_edge = self.get_param([in_channels, out_channels])
        
        self.w_rel = nn.Linear(in_channels, out_channels).to(self.device)
        xavier_normal_(self.w_rel.weight.data)
        self.w_ent = nn.Linear(in_channels, out_channels).to(self.device)
        xavier_normal_(self.w_ent.weight.data)
        self.w_edge = nn.Linear(in_channels, out_channels).to(self.device)
        xavier_normal_(self.w_edge.weight.data)
        
        # self.loop_w = self.get_param([in_channels, out_channels])
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding
        self.loop_w = nn.Linear(in_channels, out_channels).to(self.device)
        xavier_normal_(self.loop_w.weight.data)
        # self.loop_rel = nn.Linear(1, in_channels).to(self.device)

        self.drop = nn.Dropout(drop_rate).to(self.device)
        self.bias = nn.Parameter(torch.zeros(out_channels)).to(self.device) if bias == "True" else None
        self.bn = torch.nn.BatchNorm1d(out_channels).to(self.device) if bn == "True" else None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape)).to(self.device)
        nn.init.xavier_normal_(param)
        return param

    # edges.src是源节点的特征, edges.dst是目标节点的特征, edges.data是边的特征
    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        msg = self.w_edge(edge_data)
        return {'msg': msg}
    
    def message_func_2(self, edges):
        edge_type = edges.data['type']
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        msg = self.w_ent(edge_data)
        # msg = torch.matmul(edge_data, self.w_ent)
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h'])}

    def comp(self, h, edge_data):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), n=a.shape[-1])
        
        def rotate(h, r):
            # re: first half, im: second half
            # assume embedding dim is the last dimension
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)
            
        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'sum':
            return h + edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        elif self.opn == 'rotate':
            return rotate(h, edge_data)
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, x, rel_repr, g, num_ent, all_node_num):
        self.rel = rel_repr
        
        g = g.local_var()
        x_f = torch.zeros(all_node_num, self.in_channels).to(self.device)
        x_f[:num_ent] = x
        
        g.ndata['h'] = x_f
        g.update_all(self.message_func, fn.mean(msg='msg', out='h'), self.reduce_func) # 设置节点特征为所有入边上信息之和
        edge_type = g.edata['type']
        
        g= dgl.reverse(g)
        g.edata['type'] = edge_type

        g.update_all(self.message_func_2, fn.mean(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h')[:num_ent]/2 + self.loop_w(self.comp(x, self.loop_rel))[:num_ent]/2
            
        if self.bias is not None:
            x = x + self.bias
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)#, self.w_rel(self.rel)
    