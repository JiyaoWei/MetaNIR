import torch.nn as nn
import torch
import dgl.function as fn
import torch.nn.functional as F
from egraph import EntgraphCompConv
from rgraph import RelgraphConv
from torch_scatter import scatter_add, scatter_mean
from torch.nn.init import xavier_normal_



class ExtGNN(nn.Module):
    # knowledge extrapolation with GNN
    def __init__(self, args):
        super(ExtGNN, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.rel_w = nn.Parameter(torch.Tensor(args.rel_dim, self.args.rel_dim))
        nn.init.xavier_uniform_(self.rel_w, gain=nn.init.calculate_gain('relu'))

        if self.args.ent_transfer_method == 'egraph':
            self.e_conv_layers0 = nn.ModuleList()
            for _ in range(self.args.egraph_num_layers):
                self.e_conv_layers0.append(EntgraphCompConv(self.args.dim, self.args.egraph_gcn_dim, self.args.egraph_dropout, opn=self.args.egraph_opn, device=self.args.gpu, bias=self.args.egraph_bias, bn=self.args.egraph_bn, activation=self.args.egraph_act).to(self.args.gpu))

        if self.args.rel_transfer_method == 'rgraph':
            self.r_conv_layers0 = nn.ModuleList()
            self.meta_r_embeddings = nn.Embedding(16, self.args.dim).to(self.args.gpu)
            xavier_normal_(self.meta_r_embeddings.weight)
            for _ in range(self.args.rgraph_num_layers):
                self.r_conv_layers0.append(RelgraphConv(self.args.num_rel_bases, self.args.rgraph_gcn_dim, drop_rate=self.args.rgraph_dropout, device=self.args.gpu, bias=self.args.rgraph_bias, bn=self.args.rgraph_bn, activation=self.args.rgraph_act).to(self.args.gpu))


    # base2rel_feat，初始化关系嵌入（伴随mrel_feat）时需要用到
    def forward(self, ent_g, rel_g, base2rel_feat, ent_feat, rel_feat, mrel_feat):
        if self.args.gnnent == 'True' and self.args.gnnrel == 'True':
            for i in range(len(self.r_conv_layers0)):
                rel_feat = self.r_conv_layers0[i](rel_feat, mrel_feat, base2rel_feat, rel_g)
                
            all_node_num = ent_g.num_nodes()
            for i in range(len(self.e_conv_layers0)):
                ent_feat = self.e_conv_layers0[i](ent_feat, rel_feat, ent_g, self.args.num_ent, all_node_num)
            
            return ent_feat, rel_feat
        elif self.args.gnnent == 'False' and self.args.gnnrel == 'True':
            for i in range(len(self.r_conv_layers0)):
                rel_feat = self.r_conv_layers0[i](rel_feat, mrel_feat, base2rel_feat, rel_g)

            return ent_feat, rel_feat
        elif self.args.gnnent == 'True' and self.args.gnnrel == 'False':
            all_node_num = ent_g.num_nodes()
            for i in range(len(self.e_conv_layers0)):
                ent_feat = self.e_conv_layers0[i](ent_feat, rel_feat, ent_g, self.args.num_ent, all_node_num)
            rel_feat = torch.matmul(rel_feat, self.rel_w)
            
            return ent_feat, rel_feat
        else:
            return ent_feat, rel_feat