from model import Model
from data import *
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import numpy as np
from trainer import Trainer
import os
import copy
import math
from utils import *
from stare import StarEConv
from torch.nn.init import xavier_normal_



class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, bias: bool, use_node: bool) -> None:
        super().__init__()
        self.heads = heads
        self.use_node = use_node       

        if self.use_node is True:
            self.layer_e_s=nn.Linear(hidden_dim,hidden_dim)
            self.layer_r_s=nn.Linear(hidden_dim,hidden_dim)
            self.layer_e_o=nn.Linear(hidden_dim,hidden_dim)
            self.layer_r_o=nn.Linear(hidden_dim,hidden_dim)
            self.layer_a=nn.Linear(hidden_dim,hidden_dim)
            self.layer_v=nn.Linear(hidden_dim,hidden_dim)
        else:
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)


    def forward(self, x : torch.Tensor):
        shape = x.shape[:-1]

        if self.use_node is False:
            x = self.linear(x)
        else:
            device=x.device
            max_seq_len=x.size(1)
            mask_r_s = torch.tensor([1]+[0]*(max_seq_len-1)).to(device)
            mask_e_s = torch.tensor([0,1]+[0]*(max_seq_len-2)).to(device)
            mask_r_o = torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to(device)
            mask_e_o = torch.tensor([0,0,0,1]+[0]*(max_seq_len-4)).to(device)
            mask_a = torch.tensor([0,0,0,0]+[1,0]*int(((max_seq_len-4)/2))).to(device)
            mask_v = torch.tensor([0,0,0,0]+[0,1]*int(((max_seq_len-4)/2))).to(device)

            x_r_s=self.layer_r_s(torch.mul(x,mask_r_s[:,None].expand(-1,x.size(-1))))
            x_e_s=self.layer_e_s(torch.mul(x,mask_e_s[:,None].expand(-1,x.size(-1))))
            x_r_o=self.layer_r_o(torch.mul(x,mask_r_o[:,None].expand(-1,x.size(-1))))
            x_e_o=self.layer_e_o(torch.mul(x,mask_e_o[:,None].expand(-1,x.size(-1))))
            x_a=self.layer_a(torch.mul(x,mask_a[:,None].expand(-1,x.size(-1))))
            x_v=self.layer_v(torch.mul(x,mask_v[:,None].expand(-1,x.size(-1))))
                            
            x=(x_r_s+x_e_s+x_r_o+x_e_o+x_a+x_v) 
      
        return x.reshape(*shape, self.heads, -1)


class MultiHeadAttention_seven(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, use_edge: bool, remove_mask: bool, bias: bool, use_node: bool) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.query = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.key = PrepareForMultiHeadAttention(hidden_dim, heads, bias, use_node)
        self.value = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.pos = PrepareForMultiHeadAttention(hidden_dim, heads, True, use_node)
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.use_edge = use_edge
        self.remove_mask = remove_mask
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # u
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim)) # v

    def get_mask(self, graph: torch.Tensor):
        return graph.unsqueeze(1).repeat(1, self.heads, 1, 1)

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        # query/key/value: (batch, seq_len, hidden_dim)
        # batch, seq_len, head_num, hid
        # graph: (batch, kinds, query, key)
        shape = query.shape[:-1]
        query = self.query(query)   # (batch, seq_len, head, hidden)
        key = self.key(key)         # (batch, seq_len, head, hidden)
        value = self.value(value)   # (batch, seq_len, head, hidden)
        seq_len = query.size(1)
        if self.use_edge is True:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key.unsqueeze(0).repeat(query.shape[0],1,1,1)) + torch.einsum("bkqd,bkhd->bhqk", edge_query.unsqueeze(0).repeat(query.shape[0],1,1,1), key) + torch.einsum("bkqd,bqkd->bqk", edge_query.unsqueeze(0).repeat(query.shape[0],1,1,1), edge_key.unsqueeze(0).repeat(query.shape[0],1,1,1)).unsqueeze(1)
            # batch_size,head,seq_len,seq_len
            scores = scores * self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(4,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False     
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value.unsqueeze(0).repeat(query.shape[0],1,1,1))
            x = x.reshape(*shape, -1)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key)
            scores *= self.scale
            mask = self.get_mask(graph)
            if self.remove_mask is True:
                for i in range(3,seq_len,2):
                    if i==3:
                        mask[:,:,i:(i+2),(i+2):]=False
                    elif i==(seq_len-2):
                        mask[:,:,i:(i+2),3:i]=False
                    else:
                        mask[:,:,i:(i+2),(i+2):]=False
                        mask[:,:,i:(i+2),3:i]=False  
            scores = scores.masked_fill(mask == 0, -100000)
            attn = self.softmax(scores)
            attn = self.dropout(attn)
            x = torch.einsum("bhqk,bkhd->bqhd", attn, value)
            x = x.reshape(*shape, -1)

        return self.output(x)  # (batch, query, hidden_dim)


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation) -> None:
        super().__init__()
        act = None
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layer(x)
    
        
class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout_prob: float, activation: str, use_edge: bool, remove_mask: bool, use_node: bool, bias=True, times=4) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadAttention_seven(hidden_dim, heads, dropout_prob, use_edge, remove_mask, bias, use_node)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, 2048, hidden_dim, activation)
        # self.ffn = FeedForward(hidden_dim, hidden_dim * times, hidden_dim, activation)

    def forward(self, x: torch.Tensor, graph: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, graph=graph, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x


class MetaTrainer(Trainer):
    def __init__(self, args):
        super(MetaTrainer, self).__init__(args)
        # dataset
        if self.args.scorer_func == 'MNKGE':
            if args.metalearning == 'True':
                self.train_subgraph_iter = TrainSubgraphDataset(args)
                if args.pretraining == 'True':
                    self.train_iter = TrainDataset(args)
            else:
                self.train_iter = TrainDataset(args)
            # model
        elif self.args.scorer_func in ['maker', 'pmpi', 'ingram']:
            self.train_subgraph_iter = TrainBinarySubgraphDataset(args)
            # kv形式的
            self.data_loader_eval = BaselineBinaryTestDataset(args)
        else:    
            if self.args.scorer_func in ['i_hahe', 'i_gran', 'i_hytransformer', 'i_stare']:
                self.data_loader_train = MaskTrainDatasetMarginLoss(args)
            else:
                self.data_loader_train = TrainDatasetMarginLoss(args)
            # BaselineTestDataset n-ary基线
            self.data_loader_eval = BaselineTestDataset(args)
        
        self.args = args
        self.model = Model(args).to(args.gpu)

        if self.args.scorer_func == 'i_hahe':
            edge_labels = []
            max_aux = 8 - 2
            edge_labels.append([0, 1, 2] + [3,4] * max_aux )
            edge_labels.append([1, 0, 5] + [6,7] * max_aux )
            edge_labels.append([2, 5, 0] + [8,9] * max_aux )
            for idx in range(max_aux):
                edge_labels.append([3,6,8] + [11,12] * idx + [0,10] + [11,12] * (max_aux - idx - 1))
                edge_labels.append([4,7,9] + [12,13] * idx + [10,0] + [12,13] * (max_aux - idx - 1))
            self.edge_labels = torch.LongTensor(edge_labels).to(self.args.gpu)
        elif self.args.scorer_func in ['i_gran']:
            edge_labels = []
            max_aux = 8 - 2
            edge_labels.append([0, 1, 2] + [3, 0] * max_aux)
            edge_labels.append([1] + [0] * (max_aux*2+2))
            edge_labels.append([2] + [0] * (max_aux*2+2))
            for idx in range(max_aux):
                edge_labels.append([3, 0, 0] + [0, 0] * idx + [0, 4] + [0, 0] * (max_aux - idx - 1))
                edge_labels.append([0, 0, 0] + [0, 0] * idx + [4, 0] + [0, 0] * (max_aux - idx - 1))
            self.edge_labels = torch.LongTensor(edge_labels).to(self.args.gpu)


        if self.args.scorer_func == 'i_stare':
            self.e_conv_layers0 = nn.ModuleList()
            for _ in range(2):
                self.e_conv_layers0.append(StarEConv(in_channels=self.args.dim, out_channels=self.args.egraph_gcn_dim, act=nn.ELU(), device=self.args.gpu).to(self.args.gpu))
        if self.args.scorer_func == 'i_hahe':
            self.e_conv_layers0 = nn.ModuleList()
            self.e_conv_layers1 = nn.ModuleList()
            from torch_geometric.nn import GATv2Conv
            for _ in range(2):
                self.e_conv_layers0.append(GATv2Conv(self.args.dim, self.args.dim//4, heads=4, dropout=0.1).to(self.args.gpu))
                self.e_conv_layers1.append(GATv2Conv(self.args.dim, self.args.dim//4, heads=4, dropout=0.1).to(self.args.gpu))
        
        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        # args for controlling training
        self.num_step = args.num_step
        self.log_per_step = args.log_per_step
        self.check_per_step = args.check_per_step
        self.early_stop_patience = args.early_stop_patience

        self.output_norm = nn.LayerNorm(args.dim).to(self.args.gpu)
        self.output_act = nn.GELU().to(self.args.gpu)
        self.output_linear = nn.Linear(args.dim, args.dim).to(self.args.gpu)
        mask_embbing = nn.Embedding(1, args.dim).to(self.args.gpu)
        self.mask_emb = nn.init.xavier_uniform_(mask_embbing.weight, gain=nn.init.calculate_gain('relu'))
        
        self.layers = nn.ModuleList()
        if self.args.scorer_func in ['maker', 'cvt_dicgrl']:
            self.epsilon = 2.0
            self.gamma = torch.Tensor([args.gamma])
            self.rel_reduce = nn.Linear(args.dim, int(args.dim*1/2)).to(self.args.gpu)
            nn.init.xavier_uniform_(self.rel_reduce.weight, gain=nn.init.calculate_gain('relu'))
            self.embedding_range = torch.Tensor([(self.gamma.item() + self.epsilon) / args.dim])
            self.cvt_e_w = nn.Linear(self.args.dim, self.args.dim).to(self.args.gpu)
            xavier_normal_(self.cvt_e_w.weight)
            self.cvt_r_w = nn.Linear(self.args.dim, self.args.dim).to(self.args.gpu)
            xavier_normal_(self.cvt_r_w.weight)
            self.margin_loss_func = nn.MarginRankingLoss(margin=float(8), reduction="sum").to(self.args.gpu)  #
        if self.args.scorer_func in ['MNKGE']:
            decoder_activation, use_edge, remove_mask, use_node, bias = 'gelu', True, False, True, True
            mask_embbing = nn.Embedding(1, self.args.dim).to(self.args.gpu)
            self.mask_emb = nn.init.xavier_uniform_(mask_embbing.weight, gain=nn.init.calculate_gain('relu'))
            for _ in range(self.args.Trans_layers):
                self.layers.append(TransformerLayer(self.args.Trans_hid_dim, self.args.Trans_heads, self.args.Trans_drop, decoder_activation, use_edge, remove_mask, use_node, bias)).to(self.args.gpu)
            self.input_norm = nn.LayerNorm(args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop).to(self.args.gpu)
            self.position_embeddings = nn.Embedding(16, args.dim).to(self.args.gpu)
            self.input_norm = nn.LayerNorm(args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop).to(self.args.gpu)
            
            self.edge_query_embedding = nn.Embedding(19, self.args.Trans_hid_dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.edge_key_embedding = nn.Embedding(19, self.args.Trans_hid_dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.edge_value_embedding = nn.Embedding(19, self.args.Trans_hid_dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
        elif self.args.scorer_func in ['i_hytransformer', 'i_stare']:
            self.TransformerEncoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.args.dim, self.args.Trans_heads, self.args.Trans_hid_dim, self.args.Trans_drop), self.args.Trans_layers).to(self.args.gpu)
            self.input_norm = nn.LayerNorm(self.args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop).to(self.args.gpu)
            self.position_embeddings = nn.Embedding(8*2-1, self.args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop).to(self.args.gpu)
        elif self.args.scorer_func in ['i_gran']:
            self.input_norm = nn.LayerNorm(self.args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop).to(self.args.gpu)
            self.edge_key_embedding = nn.Embedding(5, self.args.dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.edge_value_embedding = nn.Embedding(5, self.args.dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.layers = nn.ModuleList()
            for _ in range(self.args.Trans_layers):
                self.layers.append(TransformerLayer_gran(args, self.args.dim, self.args.Trans_hid_dim, self.args.Trans_heads, self.args.Trans_drop, model='i_gran'))
        elif self.args.scorer_func == 'i_hahe':
            self.input_norm = nn.LayerNorm(self.args.dim).to(self.args.gpu)
            self.input_dropout = nn.Dropout(p=self.args.Trans_drop)
            self.layers = nn.ModuleList()
            self.edge_query_embedding = nn.Embedding(14, self.args.dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.edge_key_embedding = nn.Embedding(14, self.args.dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            self.edge_value_embedding = nn.Embedding(14, self.args.dim // self.args.Trans_heads, padding_idx=0).to(self.args.gpu)
            for _ in range(self.args.Trans_layers):
                self.layers.append(TransformerLayer_gran(args, self.args.dim, self.args.Trans_hid_dim, self.args.Trans_heads, self.args.Trans_drop, model='i_hahe')).to(self.args.gpu)
        elif self.args.scorer_func == 'i_hinge':
            self.softloss_func = torch.nn.Softplus()
            self.num_filters = 200
            self.i_FCN_net = torch.nn.Linear(self.num_filters*(self.args.dim-2), 1).to(self.args.gpu)
            # nn.init.xavier_uniform_(self.i_FCN_net.weight, gain=nn.init.calculate_gain('relu'))
            zeros_(self.i_FCN_net.bias.data)
            self.conv1 = torch.nn.Conv2d(1, self.num_filters, (3, 3)).to(self.args.gpu)
            zeros_(self.conv1.bias.data)
            self.batchNorm1 = torch.nn.BatchNorm2d(self.num_filters, momentum=0.1).to(self.args.gpu)
            truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
            self.conv2 = torch.nn.Conv2d(1, self.num_filters, (5, 3)).to(self.args.gpu)
            zeros_(self.conv2.bias.data)
            self.batchNorm2 = torch.nn.BatchNorm2d(self.num_filters, momentum=0.1).to(self.args.gpu)
            truncated_normal_(self.conv2.weight, mean=0.0, std=0.1)
        elif self.args.scorer_func == 'i_hyconve':
            self.conv_layer_2 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 3)).to(self.args.gpu)
            self.conv_layer_3 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 4)).to(self.args.gpu)
            self.conv_layer_4 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 5)).to(self.args.gpu)
            self.conv_layer_5 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 6)).to(self.args.gpu)
            self.conv_layer_6 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 7)).to(self.args.gpu)
            self.conv_layer_7 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 8)).to(self.args.gpu)
            self.conv_layer_8 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 9)).to(self.args.gpu)
            self.conv_layer_9 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=(1, 1, 10)).to(self.args.gpu)
            self.arity_lst = [2, 3, 4, 5, 6, 7, 8, 9]
            self.fc_pos = nn.Linear(in_features=self.arity_lst[-1], out_features=9).to(self.args.gpu)
            self.fc_rel_2 = nn.Linear(in_features=self.args.dim, out_features=3).to(self.args.gpu)
            self.pool = torch.nn.MaxPool3d((2, 1, 1))
            self.pool1d = torch.nn.MaxPool2d((1, 2))

            self.inp_drop = nn.Dropout(0.2)
            self.dropout = nn.Dropout(0.2)
            self.dropout_3d = nn.Dropout(0.2)
            self.dropout_2d = nn.Dropout(0.2)
            self.nonlinear = nn.ReLU()
            self.args.emb_dim1 = 64
            self.args.emb_dim2 = self.args.dim // self.args.emb_dim1
            self.lmbda = 0.4
            self.conv_size = (self.args.emb_dim1 * self.args.emb_dim2) * 4 // 2
            self.conv_size_1d = (self.args.dim) * 3 // 4
            self.fc_layer = nn.Linear(in_features=self.conv_size, out_features=1).to(self.args.gpu)
            self.fc_2 = nn.Linear(in_features=2*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_3 = nn.Linear(in_features=3*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_4 = nn.Linear(in_features=4*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_5 = nn.Linear(in_features=5*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_6 = nn.Linear(in_features=6*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_7 = nn.Linear(in_features=7*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_8 = nn.Linear(in_features=8*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)
            self.fc_9 = nn.Linear(in_features=9*self.conv_size_1d, out_features=self.conv_size).to(self.args.gpu)

            self.bn1 = nn.BatchNorm3d(num_features=1).to(self.args.gpu)
            self.bn2 = nn.BatchNorm3d(num_features=4).to(self.args.gpu)
            self.bn3 = nn.BatchNorm2d(num_features=1).to(self.args.gpu)
            self.bn4 = nn.BatchNorm1d(num_features=self.conv_size).to(self.args.gpu)
            self.criterion = nn.Softplus()

            # nn.init.xavier_uniform_(self.conv_layer_2.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_3.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_4.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_5.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_6.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_7.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_8.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.conv_layer_9.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_layer.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_rel_2.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_2.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_3.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_4.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_5.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_6.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_7.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_8.weight.data, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(self.fc_9.weight.data, gain=nn.init.calculate_gain('relu'))
        elif self.args.scorer_func == 'i_shrinke':
            self.bcelogitloss = torch.nn.BCEWithLogitsLoss()
            self.min_fc = nn.Sequential(
                torch.nn.Linear(7*self.args.dim, self.args.dim),
                torch.nn.Sigmoid()
            ).to(self.args.gpu)
            self.max_fc = nn.Sequential(
                torch.nn.Linear(7*self.args.dim, self.args.dim),
                torch.nn.Sigmoid()
            ).to(self.args.gpu)
            self.diag_w = nn.Linear(self.args.dim, self.args.dim).to(self.args.gpu)
            self.offset_w = nn.Linear(self.args.dim, self.args.dim).to(self.args.gpu)
        elif self.args.scorer_func == 'i_neuinfer':
            self.bce_loss_func = torch.nn.functional.binary_cross_entropy_with_logits
            self.hrtFCNs_layers=2
            self.hrtavFCNs_layers=2
            self.g_theta_dim=1000
            self.weight=0.8
            self.layer_1_norm = nn.LayerNorm(self.g_theta_dim).to(self.args.gpu)
            self.layer_2_norm = nn.LayerNorm(self.args.dim*2).to(self.args.gpu)
            self.layer_3_norm = nn.LayerNorm(self.g_theta_dim).to(self.args.gpu)
            if self.hrtavFCNs_layers == 1:
                self.g_1_linear = nn.Linear(self.args.dim*5, self.g_theta_dim).to(self.args.gpu)
                # nn.init.xavier_uniform_(self.g_1_linear.weight, gain=nn.init.calculate_gain('relu'))
            else:
                self.g_2_linear = nn.Linear(self.args.dim*5, self.args.dim*2).to(self.args.gpu)
                self.g_3_linear = nn.Linear(self.args.dim*2, self.g_theta_dim).to(self.args.gpu)
                # nn.init.xavier_uniform_(self.g_2_linear.weight, gain=nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(self.g_3_linear.weight, gain=nn.init.calculate_gain('relu'))
            if self.hrtFCNs_layers == 1:
                self.t_1_linear = nn.Linear(self.args.dim*3, self.g_theta_dim).to(self.args.gpu)
                # nn.init.xavier_uniform_(self.t_1_linear.weight, gain=nn.init.calculate_gain('relu'))
            else:
                self.t_2_linear = nn.Linear(self.args.dim*3, self.args.dim*2).to(self.args.gpu)
                self.t_3_linear = nn.Linear(self.args.dim*2, self.g_theta_dim).to(self.args.gpu)
                # nn.init.xavier_uniform_(self.t_2_linear.weight, gain=nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(self.t_3_linear.weight, gain=nn.init.calculate_gain('relu'))
                
            self.hrt_score_linear = nn.Linear(self.g_theta_dim, 1).to(self.args.gpu)
            # nn.init.xavier_uniform_(self.hrt_score_linear.weight, gain=nn.init.calculate_gain('relu'))
            self.Tkv_score_linear = nn.Linear(self.g_theta_dim, 1).to(self.args.gpu)
            # nn.init.xavier_uniform_(self.Tkv_score_linear.weight, gain=nn.init.calculate_gain('relu'))
        elif self.args.scorer_func == 'ingram':
            dim_ent, hid_dim_ratio_ent, dim_rel, hid_dim_ratio_rel, num_bin = self.args.dim, 8, self.args.dim, 4, 10
            
            # my_model = InGram(dim_ent = d_e, hid_dim_ratio_ent = hdr_e, dim_rel = d_r, hid_dim_ratio_rel = hdr_r, \
				# num_bin = B, num_layer_ent = args.num_layer_ent, num_layer_rel = args.num_layer_rel, # num_head = args.num_head)
            num_layer_ent=2
            num_layer_rel=2
            num_head = 8
            bias = True
            
            layers_ent = []
            layers_rel = []
            layer_dim_ent = hid_dim_ratio_ent * dim_ent
            layer_dim_rel = hid_dim_ratio_rel * dim_rel
            for _ in range(num_layer_ent):
                layers_ent.append(InGramEntityLayer(layer_dim_ent, layer_dim_ent, layer_dim_rel, \
                                                    bias = bias, num_head = num_head))
            for _ in range(num_layer_rel):
                layers_rel.append(InGramRelationLayer(layer_dim_rel, layer_dim_rel, num_bin, \
                                                    bias = bias, num_head = num_head))
            res_proj_ent = []
            for _ in range(num_layer_ent):
                res_proj_ent.append(nn.Linear(layer_dim_ent, layer_dim_ent, bias = bias))
            
            res_proj_rel = []
            for _ in range(num_layer_rel):
                res_proj_rel.append(nn.Linear(layer_dim_rel, layer_dim_rel, bias = bias))

            self.res_proj_ent = nn.ModuleList(res_proj_ent)
            self.res_proj_rel = nn.ModuleList(res_proj_rel)
            self.bias = bias
            self.ent_proj1 = nn.Linear(dim_ent, layer_dim_ent, bias = bias)
            self.ent_proj2 = nn.Linear(layer_dim_ent, dim_ent, bias = bias)
            self.layers_ent = nn.ModuleList(layers_ent)
            self.layers_rel = nn.ModuleList(layers_rel)

            self.rel_proj1 = nn.Linear(dim_rel, layer_dim_rel, bias = bias)
            self.rel_proj2 = nn.Linear(layer_dim_rel, dim_rel, bias = bias)
            self.rel_proj = nn.Linear(dim_rel, dim_ent, bias = bias)
            self.num_layer_ent = num_layer_ent
            self.num_layer_rel = num_layer_rel
            self.act = nn.ReLU()
            # nn.init.xavier_normal_(self.ent_proj1.weight, gain = nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(self.ent_proj2.weight, gain = nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(self.rel_proj1.weight, gain = nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(self.rel_proj2.weight, gain = nn.init.calculate_gain('relu'))
            # nn.init.xavier_normal_(self.rel_proj.weight, gain = nn.init.calculate_gain('relu'))
            # for layer_idx in range(self.num_layer_ent):
            #     nn.init.xavier_normal_(self.res_proj_ent[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
            # for layer_idx in range(self.num_layer_rel):
            #     nn.init.xavier_normal_(self.res_proj_rel[layer_idx].weight, gain = nn.init.calculate_gain('relu'))
            if self.bias:
                nn.init.zeros_(self.ent_proj1.bias)
                nn.init.zeros_(self.ent_proj2.bias)
                nn.init.zeros_(self.rel_proj1.bias)
                nn.init.zeros_(self.rel_proj2.bias)
                nn.init.zeros_(self.rel_proj.bias)
                for layer_idx in range(self.num_layer_ent):
                    nn.init.zeros_(self.res_proj_ent[layer_idx].bias)
                for layer_idx in range(self.num_layer_rel):
                    nn.init.zeros_(self.res_proj_rel[layer_idx].bias)
                
    def get_curr_state(self):
        state = {'model': self.model.state_dict()}
        return state

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.name + '.best'), map_location=self.args.gpu)
        self.model.load_state_dict(state['model'])

    def cvt_transe_scorer(self, embeddings):
        cvt_emb_e = torch.mean(torch.stack(embeddings[1::2],dim=0).squeeze(1), dim=0)
        cvt_emb_r = torch.mean(torch.stack(embeddings[0::2],dim=0).squeeze(1), dim=0)
        cvt_emb = (self.cvt_e_w(cvt_emb_e) + self.cvt_r_w(cvt_emb_r))/2
        score = self.transe_fun(cvt_emb, embeddings[0], embeddings[1])
        for _, ele in enumerate(embeddings[2::2]):
            score = torch.stack([score, self.transe_fun(cvt_emb, embeddings[2+2*_], embeddings[2+2*_+1])]).mean(dim=0)
        return score
    
    def margin_loss(self, score, label):
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.gpu)
        # 期望p_score小于n_score，因此y为-1
        loss = self.margin_loss_func(p_score, n_score, y)/score.size(0)
        return loss
    
    def get_cvt_transe_loss(self, pred_type, fact, pred_indexs, mask_output, ent_emb, rel_emb):
        r_embedding_h = torch.index_select(rel_emb, 0, fact[0])
        h_embedding = torch.index_select(ent_emb, 0, fact[1])
        r_embedding = torch.index_select(rel_emb, 0, fact[2])
        t_embedding = torch.index_select(ent_emb, 0, fact[3])
        embeddings = [r_embedding_h, h_embedding, r_embedding, t_embedding]
        for _, ele in enumerate(fact[4::2]):
            embeddings.append(torch.index_select(rel_emb, 0, fact[4+2*_]))
            embeddings.append(torch.index_select(ent_emb, 0, fact[4+2*_+1]))
        
        for _, ele in enumerate(fact[4::2]):
            embeddings.append(torch.index_select(rel_emb, 0, fact[4+2*_]))
            embeddings.append(torch.index_select(ent_emb, 0, fact[4+2*_+1]))

        score = self.cvt_transe_scorer(embeddings)
        loss = self.margin_loss(score, mask_output)
        return loss

    def split_emb(self, emb, split_list):
        # 为什么顺序排列
        split_list = [np.sum(split_list[0: i], dtype=np.int) for i in range(len(split_list) + 1)]
        emb_split = [emb[split_list[i]: split_list[i + 1]] for i in range(len(split_list) - 1)]
        return emb_split

    def process_epoch_train(self):
        self.model.train()
        '''Start training'''
        total_loss = 0.0
        self.data_loader_train.reset()
        g = self.data_loader_train.g.to(self.args.gpu)
        pattern_g = self.data_loader_train.pattern_g.to(self.args.gpu)
        tmp_seen_ents = self.data_loader_train.tmp_seen_ents.to(self.args.gpu)
        tmp_seen_rels = self.data_loader_train.tmp_seen_rels.to(self.args.gpu)
        
        if self.args.scorer_func in ['i_stare', 'i_hahe']:
            edge_index_stare = self.data_loader_train.edge_index_stare
            edge_type_stare = self.data_loader_train.edge_type_stare
            quals_stare = self.data_loader_train.quals_stare
            edge_index_eH = self.data_loader_train.edge_index_eH
            num_training_fact = self.data_loader_train.num_training_fact
            
        for idx_b, batch in enumerate(self.data_loader_train):
            '''get loss'''
            if self.args.scorer_func in ['i_hahe', 'i_gran', 'i_hytransformer', 'i_stare']:
                bf, by, bk, td = batch
                fact = [ele.to(self.args.gpu) for ele in bf]
                label = by.to(self.args.gpu) if by is not None else by
                mask_position = bk
                ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
                if self.args.scorer_func in ['i_stare', 'i_hahe']:
                    loss, socre = self.loss_func(fact, ent_emb, rel_emb, label, mask_position, edge_index_stare, edge_type_stare, quals_stare, edge_index_eH, num_training_fact)
                else:
                    loss, socre = self.loss_func(fact, ent_emb, rel_emb, label, mask_position)
                batch_loss = loss.mean().float()
            else:
                bf, by = batch
                fact = [ele.to(self.args.gpu) for ele in bf]
                label = by.to(self.args.gpu)
                ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
                loss, socre = self.loss_func(fact, ent_emb, rel_emb, label, None)
                batch_loss = loss.mean().float()
        
            '''update'''
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()
            # break
            
        if self.args.scorer_func in ['i_hahe', 'i_gran', 'i_hytransformer', 'i_stare']:
            return total_loss/(idx_b+1)
        return total_loss
    
    def process_epoch_eval(self, istest=False):
        self.model.eval()
        detail_results = {'total':dict(), 'entity':dict(), 'relation':dict(), 'b_entity':dict(), 'b_relation':dict(), 'n_entity':dict(), 'n_relation':dict(), 'urel_ent':dict(), 'uent_ent':dict(), 'uboth_ent':dict(), 'urel_rel':dict(), 'uent_rel':dict(), 'uboth_rel':dict()}
        
        for sign, dic in detail_results.items():
            for k in ['mrr', 'hits1', 'hits3', 'hits5', 'hits10', 'count']:
                dic[k] = 0
        '''start evaluation'''
        if istest:
            self.args.valid = False
        else:
            self.args.valid = True
        
        if self.args.eval_times == 5 and istest:# and self.args.dataset == 'IWikiPeople':
            eval_times = 5
        else:
            eval_times = 1
        for _ in range(eval_times):
            for type in ['urel', 'uent', 'uboth']:
                self.args.utype = type
                self.data_loader_eval.reset()
                edge_index_stare = self.data_loader_eval.edge_index_stare
                edge_type_stare = self.data_loader_eval.edge_type_stare
                quals_stare = self.data_loader_eval.quals_stare
                edge_index_eH = self.data_loader_eval.edge_index_eH
                num_training_fact = self.data_loader_eval.num_training_fact
                for step, batch in enumerate(self.data_loader_eval):
                    # self.args.eval_batch_size * step // 
                    fact, k, label, truth = batch
                    if fact == 0:
                        continue
                    if self.args.scorer_func in ['i_hinge', 'i_neuinfer', 'i_hyconve', 'i_shrinke']:
                        eles = [ele for ele in fact]
                    else:
                        eles = [torch.LongTensor(ele).to(self.args.gpu) for ele in fact]
                        label = label.to(self.args.gpu)
                    if self.args.valid:
                        stage = 'Valid'
                    else:
                        stage = 'Test'
                    pred = self.predict(eles, k, stage, edge_index_stare, edge_type_stare, quals_stare, edge_index_eH, num_training_fact, truth, label)
                    # if self.args.scorer_func == 'maker':
                    #     b_range = torch.arange(pred.size()[1])#, device=self.args.gpu
                    # else:
                    b_range = torch.arange(pred.size()[0])#, device=self.args.gpu
                    # truth = truth.to(pred.device)
                    if self.args.num_sample_cand == 0:
                        target_pred = pred[b_range, truth]
                        pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                        pred[b_range, truth] = target_pred
                    else:
                        truth = 0
                    '''rank all candidate entities'''
                    ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, truth]
                    '''get results'''
                    ranks = ranks.float()
                    
                    signs = ['total']
                    if self.args.scorer_func in ['maker','ReDA', 'cvt_dicgrl']:
                        if k%2 == 1:
                            signs.append('entity')
                            if len(fact) == 4:
                                signs.append('b_entity')
                            else:
                                signs.append('n_entity')
                            if type == 'urel':
                                signs.append('urel_ent')
                            elif type == 'uent':
                                signs.append('uent_ent')
                            elif type == 'uboth':
                                signs.append('uboth_ent')
                        else:
                            signs.append('relation')
                            if len(fact) == 4:
                                signs.append('b_relation')
                            else:
                                signs.append('n_relation')
                            if type == 'urel':
                                signs.append('urel_rel')
                            elif type == 'uent':
                                signs.append('uent_rel')
                            elif type == 'uboth':
                                signs.append('uboth_rel')
                    else:
                        if k%2 == 0:
                            signs.append('entity')
                            if len(fact) == 3:
                                signs.append('b_entity')
                            else:
                                signs.append('n_entity')
                            if type == 'urel':
                                signs.append('urel_ent')
                            elif type == 'uent':
                                signs.append('uent_ent')
                            elif type == 'uboth':
                                signs.append('uboth_ent')
                        else:
                            signs.append('relation')
                            if len(fact) == 3:
                                signs.append('b_relation')
                            else:
                                signs.append('n_relation')
                            if type == 'urel':
                                signs.append('urel_rel')
                            elif type == 'uent':
                                signs.append('uent_rel')
                            elif type == 'uboth':
                                signs.append('uboth_rel')
                    for sign in signs:
                        detail_results[sign]['count'] = torch.numel(ranks) + detail_results[sign].get('count', 0.0)
                        detail_results[sign]['mr'] = torch.sum(ranks).item() + detail_results[sign].get('mr', 0.0)
                        detail_results[sign]['mrr'] = torch.sum(1.0 / ranks).item() + detail_results[sign].get('mrr', 0.0)
                        for k in range(10):
                            detail_results[sign]['hits{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + detail_results[sign].get(
                                'hits{}'.format(k + 1), 0.0)            
                    
        for sign in ['total', 'entity', 'relation', 'b_entity', 'b_relation', 'n_entity', 'n_relation', 'urel_ent', 'uent_ent', 'uboth_ent', 'urel_rel', 'uent_rel', 'uboth_rel']:
            count = float(detail_results[sign]['count'])
            for key, val in detail_results[sign].items():
                if count == 0:  continue
                detail_results[sign][key] = round(val / count, 4)
        return detail_results

    def predict(self, eles, k, stage='Valid', edge_index_stare=None, edge_type_stare=None, quals_stare=None, edge_index_eH=None, num_training_fact=None, truth=None, label=None):
        '''
        Scores all candidate facts for evaluation
        :param head: subject entity id
        :param rel: relation id
        :param stage: object entity id
        :return: scores of all candidate facts
        '''

        '''get entity and relation embeddings'''
        # ent_embeddings, rel_embeddings = self.get_eval_emb(self.data_loader_eval.g, self.data_loader_eval.pattern_g, self.data_loader_eval.train_ents, self.data_loader_eval.train_rels)
        ent_embeddings, rel_embeddings = self.model(self.data_loader_eval.g, self.data_loader_eval.pattern_g, self.data_loader_eval.train_ents, self.data_loader_eval.train_rels)
        if self.args.scorer_func in ['i_hahe', 'i_stare']:
            ent_embeddings, rel_embeddings = self.neighbor_aggregator(ent_embeddings, rel_embeddings, edge_index_stare, edge_type_stare, quals_stare, edge_index_eH, num_training_fact)
            
        if self.args.scorer_func not in ['i_hinge', 'i_hyconve', 'i_shrinke', 'i_hytransformer', 'i_hahe', 'i_gran', 'i_neuinfer', 'i_stare']:
            if self.args.scorer_func in ['maker', 'cvt_dicgrl']:
                querys, eles_c = self.get_cvt_query_embeddings(eles, ent_embeddings, rel_embeddings, k, stage)
            else:
                eles_ = self.get_query_embeddings(eles, ent_embeddings, rel_embeddings, k, stage)
        
        if self.args.scorer_func == 'maker':
            # querys = self.get_querys(eles, k, stage)
            # querys, eles_c = self.get_cvt_query_embeddings(eles, ent_embeddings, rel_embeddings, k, stage)
            if self.args.num_sample_cand != 0:
                total_candidates = range(querys[0].shape[1])
                total_negatives = [a for a in total_candidates if label[0][a] == 0]
                eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                eval_candidates[0] = truth[0]
                querys = [_[0][eval_candidates].unsqueeze(0) for _ in querys]
            score = 8 - self.cvt_transe_scorer(querys)
            # score = None
            # for i in range(0, querys[0].shape[0], 128):
            #     cur_query = [torch.LongTensor(querys_ele[i:i+128]).to(self.args.gpu) for querys_ele in querys]
            #     cur_query_embeddings = self.get_embeddings(cur_query, ent_embeddings, rel_embeddings)
            #     cur_score = self.hinge_scorer(cur_query_embeddings)
            #     if i == 0:
            #         score = cur_score.cpu().detach()
            #     else:
            #         score = torch.cat([score, cur_score.cpu().detach()], 0)
            # score = score.unsqueeze(0).squeeze(2)
            # score = 8 - self.cvt_transe_scorer(eles_)
            # score = 8 - self.cvt_rotate_scorer(eles_)
        elif self.args.scorer_func == 'cvt_dicgrl':
            score = 8 - self.cvt_dicgrl_scorer(eles_, eles_c).unsqueeze(0)
        elif self.args.scorer_func == 'i_hinge':
            querys = self.get_querys(eles, k, stage)
            
            if self.args.num_sample_cand != 0:
                total_candidates = range(len(querys[0]))
                total_negatives = [a for a in total_candidates if label[0][a] == 0]
                eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                eval_candidates[0] = truth[0]
                querys = [_[eval_candidates] for _ in querys]
            
            score = None
            for i in range(0, querys[0].shape[0], 128):
                cur_query = [torch.LongTensor(querys_ele[i:i+128]).to(self.args.gpu) for querys_ele in querys]
                cur_query_embeddings = self.get_embeddings(cur_query, ent_embeddings, rel_embeddings)
                cur_score = self.hinge_scorer(cur_query_embeddings)
                if i == 0:
                    score = cur_score.cpu().detach()
                else:
                    score = torch.cat([score, cur_score.cpu().detach()], 0)
            score = score.unsqueeze(0).squeeze(2)
        elif self.args.scorer_func == 'i_hyconve':
            querys = self.get_querys(eles, k, stage)

            if self.args.num_sample_cand != 0:
                total_candidates = range(len(querys[0]))
                total_negatives = [a for a in total_candidates if a not in label]
                eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                eval_candidates[0] = truth[0]
                querys = [_[eval_candidates] for _ in querys]
            
            score = None
            for i in range(0, querys[0].shape[0], 128):
                cur_query = [torch.LongTensor(querys_ele[i:i+128]).to(self.args.gpu) for querys_ele in querys]
                cur_query_embeddings = self.get_embeddings(cur_query, ent_embeddings, rel_embeddings)
                cur_score = self.hyconve_scorer(cur_query_embeddings)
                if i == 0:
                    score = cur_score.cpu().detach()
                else:
                    score = torch.cat([score, cur_score.cpu().detach()], 0)
            score = score.unsqueeze(0)
        elif self.args.scorer_func == 'i_shrinke':
            querys = self.get_querys(eles, k, stage)
            
            if self.args.num_sample_cand != 0:
                total_candidates = range(len(querys[0]))
                total_negatives = [a for a in total_candidates if label[0][a] == 0]
                eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                eval_candidates[0] = truth[0]
                querys = [_[eval_candidates] for _ in querys]
                
            score = None
            for i in range(0, querys[0].shape[0], 128):
                cur_query = [torch.LongTensor(querys_ele[i:i+128]).to(self.args.gpu) for querys_ele in querys]
                cur_query_embeddings = self.get_embeddings(cur_query, ent_embeddings, rel_embeddings)
                cur_score = self.shrinke_scorer(cur_query_embeddings)
                if i == 0:
                    score = cur_score.cpu().detach()
                else:
                    score = torch.cat([score, cur_score.cpu().detach()], 0)
            score = score.unsqueeze(0).squeeze(2)
        elif self.args.scorer_func == 'i_neuinfer':
            querys = self.get_querys(eles, k, stage)
            
            if self.args.num_sample_cand != 0:
                total_candidates = range(len(querys[0]))
                total_negatives = [a for a in total_candidates if label[0][a] == 0]
                eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                eval_candidates[0] = truth[0]
                querys = [_[eval_candidates] for _ in querys]
                
            score = None
            for i in range(0, querys[0].shape[0], 128):
                cur_query = [torch.LongTensor(querys_ele[i:i+128]).to(self.args.gpu) for querys_ele in querys]
                cur_query_embeddings = self.get_embeddings(cur_query, ent_embeddings, rel_embeddings)
                cur_score = self.neuinfer_scorer(cur_query_embeddings)
                if i == 0:
                    score = cur_score.cpu().detach()
                else:
                    score = torch.cat([score, cur_score.cpu().detach()], 0)
            score = score.unsqueeze(0)
        elif self.args.scorer_func in ['i_hytransformer', 'i_gran', 'i_hahe','i_stare']:
            eles_ = self.get_mask_query_embeddings(eles, ent_embeddings, rel_embeddings)
            if k%2 == 0:
                num_ent = self.data_loader_eval.num_ent
                score = self.transformer_scorer(eles_, mask_position=k, rel_embeddings=rel_embeddings, ent_embeddings=ent_embeddings)[:,:num_ent]
            else:
                num_rel = self.data_loader_eval.num_rel
                score = self.transformer_scorer(eles_, mask_position=k, rel_embeddings=rel_embeddings, ent_embeddings=ent_embeddings)[:,:num_rel]

            if self.args.num_sample_cand != 0:
                new_score = []
                for _ in range(len(score)):
                    total_candidates = range(len(score[_]))
                    # total_negatives = [a for a in total_candidates if label[_][a] == 0]
                    indices = (label[_] == 0).nonzero(as_tuple=True)[0]
                    total_negatives = [total_candidates[i] for i in indices]
                    
                    eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                    eval_candidates[0] = truth[_].item()
                    tmp_score = score[_][eval_candidates]
                    new_score.append(tmp_score.unsqueeze(0))
                score = torch.cat(new_score, dim=0)

        return score

    def neighbor_aggregator(self, ent_embeddings, rel_embeddings, edge_index_stare, edge_type_stare, quals_stare, edge_index_eH, num_training_fact):
        # return ent_embeddings, rel_embeddings
        if self.args.scorer_func == 'i_stare':
            for i in range(len(self.e_conv_layers0)):
                ent_embeddings, rel_embeddings = self.e_conv_layers0[i](x=ent_embeddings, edge_index=edge_index_stare,
                                                                        edge_type=edge_type_stare, rel_embed=rel_embeddings,
                                                                        quals=quals_stare)
        elif self.args.scorer_func == 'i_hahe':
            num_fact = num_training_fact
            hyper_e_embeddings = torch.zeros(num_fact, self.args.dim).to(self.args.gpu)
            for i in range(len(self.e_conv_layers0)):
                tmp = self.e_conv_layers0[i]((ent_embeddings, hyper_e_embeddings), torch.index_select(edge_index_eH.transpose(1,0), 0, torch.LongTensor([0, 1]).to(self.args.gpu)))
                hyper_e_embeddings = hyper_e_embeddings + tmp
                tmp = self.e_conv_layers1[i]((hyper_e_embeddings, ent_embeddings), torch.index_select(edge_index_eH.transpose(1,0), 0, torch.LongTensor([1, 0]).to(self.args.gpu)))
                ent_embeddings = ent_embeddings + tmp
        return ent_embeddings, rel_embeddings

    def get_query_embeddings(self, eles, ent_embeddings, rel_embeddings, k, stage):
        if k%2 == 0:
            num_ent = self.data_loader_eval.num_ent
        else:
            num_rel = self.data_loader_eval.num_rel

        eles_ = []
        if k == 0:
            e = ent_embeddings[:num_ent].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
        else:
            if k % 2 == 0:
                e = torch.index_select(ent_embeddings, 0, eles[0]).unsqueeze(1).repeat(1,num_ent,1)
            else:
                e = torch.index_select(ent_embeddings, 0, eles[0]).unsqueeze(1).repeat(1,num_rel,1)
        eles_.append(e)
        for _ in range(1, len(eles)):
            if _ < k:
                if _%2 == 0:
                    e = torch.index_select(ent_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
                else:
                    e = torch.index_select(rel_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
            elif _ == k:
                if k%2==0:
                    e = ent_embeddings[:num_ent].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
                else:
                    e = rel_embeddings[:num_rel].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
            elif _ > k:
                if _%2 == 0:
                    e = torch.index_select(ent_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
                else:
                    e = torch.index_select(rel_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
            eles_.append(e)
        return eles_
    
    def get_cvt_query_embeddings(self, eles, ent_embeddings, rel_embeddings, k, stage):
        if k%2 == 1:
            num_ent = self.data_loader_eval.num_ent
        else:
            num_rel = self.data_loader_eval.num_rel

        eles_ = []
        eles_c = []
        if k == 0:
            e_c = torch.arange(num_rel, dtype=torch.long, device=self.args.gpu).unsqueeze(0)
            e = rel_embeddings[:num_rel].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
        else:
            if k % 2 == 1:
                e_c = eles[0].unsqueeze(0).repeat(1, num_ent)
                e = torch.index_select(rel_embeddings, 0, eles[0]).unsqueeze(1).repeat(1,num_ent,1)
            else:
                e_c = eles[0].unsqueeze(0).repeat(1, num_rel)
                e = torch.index_select(rel_embeddings, 0, eles[0]).unsqueeze(1).repeat(1,num_rel,1)
        eles_.append(e)
        eles_c.append(e_c)
        for _ in range(1, len(eles)):
            if _ < k:
                if _%2 == 1:
                    e_c = eles[_].unsqueeze(0).repeat(1, e.shape[1])
                    e = torch.index_select(ent_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
                else:
                    e_c = eles[_].unsqueeze(0).repeat(1, e.shape[1])
                    e = torch.index_select(rel_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
            elif _ == k:
                if k%2==1:
                    # e_c = ent_embeddings[:num_ent].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
                    e_c = torch.arange(num_ent, dtype=torch.long, device=self.args.gpu).unsqueeze(0)
                    e = ent_embeddings[:num_ent].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
                else:
                    # e_c = rel_embeddings[:num_rel].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
                    e_c = torch.arange(num_rel, dtype=torch.long, device=self.args.gpu).unsqueeze(0)
                    e = rel_embeddings[:num_rel].unsqueeze(0).repeat(eles[0].shape[0], 1, 1)
            elif _ > k:
                if _%2 == 1:
                    e_c = eles[_].unsqueeze(0).repeat(1, e.shape[1])
                    e = torch.index_select(ent_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
                else:
                    e_c = eles[_].unsqueeze(0).repeat(1, e.shape[1])
                    e = torch.index_select(rel_embeddings, 0, eles[_]).unsqueeze(1).repeat(1,e.shape[1],1)
            eles_.append(e)
            eles_c.append(e_c)
        return eles_, eles_c

    def get_mask_query_embeddings(self, eles, ent_embeddings, rel_embeddings):
        eles_ = []
        for _ in range(0, len(eles)):
            if _%2 == 0:
                e = torch.index_select(ent_embeddings, 0, eles[_])
            else:
                e = torch.index_select(rel_embeddings, 0, eles[_])
            eles_.append(e)
        return eles_

    def get_kv_mask_query_embeddings(self, eles, ent_embeddings, rel_embeddings):
        eles_ = []
        for _ in range(0, len(eles)):
            if _%2 == 1:
                e = torch.index_select(ent_embeddings, 0, eles[_])
            else:
                e = torch.index_select(rel_embeddings, 0, eles[_])
            eles_.append(e)
        return eles_

    def get_querys(self, eles, k, stage):
        if k%2 == 0:
            num_ent = self.data_loader_eval.num_ent
        else:
            num_rel = self.data_loader_eval.num_rel

        eles_ = []
        if k == 0:
            e = np.arange(0, num_ent)
        else:
            if k % 2 == 0:
                e = eles[0].repeat(num_ent)
            else:
                e = eles[0].repeat(num_rel)
        eles_.append(e)
        for _ in range(1, len(eles)):
            if _ < k:
                e = eles[_].repeat(e.shape[0])
            elif _ == k:
                if k%2==0:
                    e = np.arange(0, num_ent)
                else:
                    e = np.arange(0, num_rel)
            elif _ > k:
                e = eles[_].repeat(e.shape[0])
            eles_.append(e)
        return eles_
    
    def get_embeddings(self, cur_query, ent_embeddings, rel_embeddings):
        cur_query_embeddings = []
        for _, ele in enumerate(cur_query):
            if _ % 2 == 0:
                cur_query_embeddings.append(torch.index_select(ent_embeddings, 0, ele))
            else:
                cur_query_embeddings.append(torch.index_select(rel_embeddings, 0, ele))
        return cur_query_embeddings
    
    def pretrain_one_epoch(self):
        batch_loss = 0
        self.train_iter.reset()
        g = self.train_iter.g.to(self.args.gpu)
        pattern_g = self.train_iter.pattern_g.to(self.args.gpu)
        tmp_seen_ents = self.train_iter.tmp_seen_ents.to(self.args.gpu)
        tmp_seen_rels = self.train_iter.tmp_seen_rels.to(self.args.gpu)
        
        for _, batch in enumerate(self.train_iter):
            ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
            pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels = batch
            
            pred_facts = pred_facts.to(self.args.gpu).transpose(1,0)
            pred_indexs = pred_indexs.to(self.args.gpu)
            mask_outputs = mask_outputs.to(self.args.gpu)
            mask_inputs = mask_inputs.to(self.args.gpu)
            edge_labels = edge_labels.to(self.args.gpu)
            
            result = self.get_transformer_scorer(pred_facts, pred_indexs, mask_inputs, mask_outputs, edge_labels, ent_emb, rel_emb)
            
            entities, relations = (query_types == 1), (query_types == -1)

            label_entity = mask_outputs[entities] * (self.args.entity_soft / (self.train_iter.ents_num - 1))
            label_entity[torch.arange(label_entity.shape[0]), mask_labels[entities]] = 1 - self.args.entity_soft
            label_relation = mask_outputs[relations] * (self.args.relation_soft / (self.train_iter.rels_num - 1))
            label_relation[torch.arange(label_relation.shape[0]), mask_labels[relations]] = 1 - self.args.relation_soft            
            loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction='none')
            loss = torch.cat((loss1, loss2)).mean()
            batch_loss += loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        batch_loss /= (_+1)
        return batch_loss
    
    def train_one_epoch(self):
        batch_loss = 0
        self.train_iter.reset()
        g = self.train_iter.g.to(self.args.gpu)
        pattern_g = self.train_iter.pattern_g.to(self.args.gpu)
        tmp_seen_ents = self.train_iter.tmp_seen_ents.to(self.args.gpu)
        tmp_seen_rels = self.train_iter.tmp_seen_rels.to(self.args.gpu)
        
        for _, batch in enumerate(self.train_iter):
            if self.args.scorer_func != 'MNKGE':
                ent_emb, rel_emb = self.model.ent_feat, self.model.rel_feat
            else:
                ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
            pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels = batch
            
            pred_facts = pred_facts.to(self.args.gpu).transpose(1,0)
            pred_indexs = pred_indexs.to(self.args.gpu)
            mask_outputs = mask_outputs.to(self.args.gpu)
            mask_inputs = mask_inputs.to(self.args.gpu)
            edge_labels = edge_labels.to(self.args.gpu)
            
            result = self.get_transformer_scorer(pred_facts, pred_indexs, mask_inputs, mask_outputs, edge_labels, ent_emb, rel_emb)
            
            entities, relations = (query_types == 1), (query_types == -1)

            label_entity = mask_outputs[entities] * (self.args.entity_soft / (self.train_iter.ents_num - 1))
            label_entity[torch.arange(label_entity.shape[0]), mask_labels[entities]] = 1 - self.args.entity_soft
            label_relation = mask_outputs[relations] * (self.args.relation_soft / (self.train_iter.rels_num - 1))
            label_relation[torch.arange(label_relation.shape[0]), mask_labels[relations]] = 1 - self.args.relation_soft            
            loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction='none')
            loss = torch.cat((loss1, loss2)).mean()
            batch_loss += loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        batch_loss /= (_+1)
        return batch_loss, ent_emb, rel_emb
    
    def train_one_step(self):
        batch_loss = 0
        self.train_subgraph_iter.reset()
        for _, data in enumerate(self.train_subgraph_iter):
            if self.args.adjustment == 'True':
                g, pattern_g, pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels, tmp_seen_ents, tmp_seen_rels, sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels = data
            else:
                g, pattern_g, pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels, tmp_seen_ents, tmp_seen_rels = data
            g = g.to(self.args.gpu)
            pattern_g = pattern_g.to(self.args.gpu)
            tmp_seen_ents = tmp_seen_ents.to(self.args.gpu)
            tmp_seen_rels = tmp_seen_rels.to(self.args.gpu)
            ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
            
            pred_facts = pred_facts.to(self.args.gpu).transpose(1,0)
            pred_indexs = pred_indexs.to(self.args.gpu)
            mask_outputs = mask_outputs.to(self.args.gpu)
            mask_inputs = mask_inputs.to(self.args.gpu)
            edge_labels = edge_labels.to(self.args.gpu)
            
            if self.args.adjustment == 'True':
                sup_pred_facts = sup_pred_facts.to(self.args.gpu).transpose(1,0)
                sup_pred_indexs = sup_pred_indexs.to(self.args.gpu)
                # 哪些candidates有用，包含 True、False
                sup_mask_outputs = sup_mask_outputs.to(self.args.gpu)
                sup_mask_inputs = sup_mask_inputs.to(self.args.gpu)
                
                ent_emb.retain_grad()
                rel_emb.retain_grad()
                result = self.get_transformer_scorer(sup_pred_facts, sup_pred_indexs, sup_mask_inputs, sup_mask_outputs, edge_labels, ent_emb, rel_emb)
                self.optimizer.zero_grad()
                # query_type为1，表示预测的是实体
                entities, relations = (sup_query_types == 1), (sup_query_types == -1)

                # 负样例
                sup_label_entity = sup_mask_outputs[entities] * (self.args.entity_soft / (self.train_subgraph_iter.ents_num - 1))
                if self.args.pred_truth == 'True':
                    ent_sup_mask_labels = []
                    for _, ele in enumerate(sup_query_types):
                        if ele == 1:
                            ent_sup_mask_labels.append(sup_mask_labels[_])
                    for _ in range(sup_label_entity.shape[0]):
                        sup_label_entity[_, ent_sup_mask_labels[_]] = 1 - self.args.entity_soft
                else:
                    sup_label_entity[torch.arange(sup_label_entity.shape[0]), sup_mask_labels[entities]] = 1 - self.args.entity_soft
                
                sup_label_relation = sup_mask_outputs[relations] * (self.args.relation_soft / (self.train_subgraph_iter.rels_num - 1))
                # 如果用adjustment，mask_label需要完全为真？
                if self.args.pred_truth == 'True':
                    rel_sup_mask_labels = []
                    for _, ele in enumerate(sup_query_types):
                        if ele == -1:
                            rel_sup_mask_labels.append(sup_mask_labels[_])
                    for _ in range(sup_label_relation.shape[0]):
                        sup_label_relation[_, rel_sup_mask_labels[_]] = 1 - self.args.relation_soft
                else:
                    sup_label_relation[torch.arange(sup_label_relation.shape[0]), sup_mask_labels[relations]] = 1 - self.args.relation_soft            
                
                loss1 = torch.nn.functional.cross_entropy(result[entities], sup_label_entity, reduction='none')
                loss2 = torch.nn.functional.cross_entropy(result[relations], sup_label_relation, reduction='none')
                loss = torch.cat((loss1, loss2)).mean()
                loss.backward(retain_graph=True)

                ent_grad_meta = ent_emb.grad
                rel_grad_meta = rel_emb.grad
                # rel_grad_meta 明显大于 ent_grad_meta
                ent_emb_q = ent_emb - self.args.ent_beta*ent_grad_meta
                rel_emb_q = rel_emb - self.args.rel_beta*rel_grad_meta                
                
                if self.args.adjustment_type != 'all':
                    ent_emb_q[tmp_seen_ents] = ent_emb[tmp_seen_ents]
                    rel_emb_q[tmp_seen_rels] = rel_emb[tmp_seen_rels]
            else:
                # ent_emb好多一样的
                ent_emb_q = ent_emb
                rel_emb_q = rel_emb
            
            
            result = self.get_transformer_scorer(pred_facts, pred_indexs, mask_inputs, mask_outputs, edge_labels, ent_emb_q, rel_emb_q)
            
            entities, relations = (query_types == 1), (query_types == -1)

            label_entity = mask_outputs[entities] * (self.args.entity_soft / (self.train_subgraph_iter.ents_num - 1))
            label_entity[torch.arange(label_entity.shape[0]), mask_labels[entities]] = 1 - self.args.entity_soft
            # if self.args.pred_truth == 'True':
            #     ent_mask_labels = []
            #     for _, ele in enumerate(query_types):
            #         if ele == 1:
            #             ent_mask_labels.append(mask_labels[_])
            #     for _ in range(label_entity.shape[0]):
            #         label_entity[_, ent_mask_labels[_]] = 1 - self.args.entity_soft
            # else:
            #     label_entity[torch.arange(label_entity.shape[0]), mask_labels[entities]] = 1 - self.args.entity_soft
            
            label_relation = mask_outputs[relations] * (self.args.relation_soft / (self.train_subgraph_iter.rels_num - 1))
            label_relation[torch.arange(label_relation.shape[0]), mask_labels[relations]] = 1 - self.args.relation_soft            
            # if self.args.pred_truth == 'True':
            #     rel_mask_labels = []
            #     for _, ele in enumerate(query_types):
            #         if ele == -1:
            #             rel_mask_labels.append(mask_labels[_])
            #     for _ in range(label_relation.shape[0]):
            #         label_relation[_, rel_mask_labels[_]] = 1 - self.args.relation_soft
            # else:
            #     label_relation[torch.arange(label_relation.shape[0]), mask_labels[relations]] = 1 - self.args.relation_soft         
            
            loss1 = torch.nn.functional.cross_entropy(result[entities], label_entity, reduction='none')
            loss2 = torch.nn.functional.cross_entropy(result[relations], label_relation, reduction='none')
            loss = torch.cat((loss1, loss2)).mean()
            batch_loss += loss

            if self.args.adjustment == 'False':
                self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # break
        batch_loss /= (_+1)
        return batch_loss, ent_emb_q, rel_emb_q
    
    def binary_train_one_step(self):
        batch_loss = 0
        self.train_subgraph_iter.reset()
        for _, data in enumerate(self.train_subgraph_iter):
            g, pattern_g, tmp_seen_ents, tmp_seen_rels, pred_facts, label = data
            g = g.to(self.args.gpu)
            pattern_g = pattern_g.to(self.args.gpu)
            tmp_seen_ents = tmp_seen_ents.to(self.args.gpu)
            tmp_seen_rels = tmp_seen_rels.to(self.args.gpu)
            ent_emb, rel_emb = self.model(g, pattern_g, tmp_seen_ents, tmp_seen_rels)
            
            pred_facts = [ele.to(self.args.gpu) for ele in pred_facts]
            label = label.to(self.args.gpu) if label is not None else label
            # pred_facts = pred_facts.to(self.args.gpu).transpose(1,0)
            loss, socre = self.loss_func(pred_facts, ent_emb, rel_emb, label, None)
            if self.args.adjustment == 'False':
                self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            batch_loss += loss.mean().float()
            # break
        batch_loss /= (_+1)
        return batch_loss, ent_emb, rel_emb

    def loss_func(self, fact, ent_embeddings, rel_embeddings, label=None, mask_position=None, edge_index_stare=None, edge_type_stare=None, quals_stare=None, edge_index_eH=None, num_training_fact=None):
        
        if self.args.scorer_func in ['i_stare', 'i_hahe']:
            ent_embeddings, rel_embeddings = self.neighbor_aggregator(ent_embeddings, rel_embeddings, edge_index_stare, edge_type_stare, quals_stare, edge_index_eH, num_training_fact)
        
        if self.args.scorer_func in ["maker", "cvt_dicgrl"]:
            r_embedding_h = torch.index_select(rel_embeddings, 0, fact[0])
            h_embedding = torch.index_select(ent_embeddings, 0, fact[1])
            r_embedding = torch.index_select(rel_embeddings, 0, fact[2])
            t_embedding = torch.index_select(ent_embeddings, 0, fact[3])
            embeddings = [r_embedding_h, h_embedding, r_embedding, t_embedding]
            for _, ele in enumerate(fact[4::2]):
                embeddings.append(torch.index_select(rel_embeddings, 0, fact[4+2*_]))
                embeddings.append(torch.index_select(ent_embeddings, 0, fact[4+2*_+1]))

            if self.args.scorer_func == 'maker':
                # score = self.cvt_rotate_scorer(embeddings)
                score = self.cvt_transe_scorer(embeddings)
                loss = self.margin_loss(score, label)
                return loss, -1
            elif self.args.scorer_func == 'cvt_dicgrl':
                score = self.cvt_dicgrl_scorer(embeddings, fact)
                loss = self.margin_loss(score, label)
                return loss, -1
        else:
            h_embedding = torch.index_select(ent_embeddings, 0, fact[0])
            r_embedding = torch.index_select(rel_embeddings, 0, fact[1])
            t_embedding = torch.index_select(ent_embeddings, 0, fact[2])
            embeddings = [h_embedding, r_embedding, t_embedding]
            for _, ele in enumerate(fact[3::2]):
                embeddings.append(torch.index_select(rel_embeddings, 0, fact[3+2*_]))
                embeddings.append(torch.index_select(ent_embeddings, 0, fact[3+2*_+1]))
            if self.args.scorer_func == 'i_hinge':
                score = self.hinge_scorer(embeddings)
                loss = self.hinge_loss(score, label)
                return loss.mean(), score
            elif self.args.scorer_func == 'i_hyconve':
                loss = self.hyconve_loss(embeddings, label)
                return loss.mean(), -1
            elif self.args.scorer_func == 'i_shrinke':
                score = self.shrinke_scorer(embeddings)
                loss = self.shrinke_loss(score, label)
                return loss.mean(), score
            elif self.args.scorer_func == 'i_neuinfer':
                score = self.neuinfer_scorer(embeddings)
                loss = self.neuinfer_loss(score, label)
                return loss, -1
            elif self.args.scorer_func in ["i_gran", 'i_hahe', "i_hytransformer", 'i_stare']:
                score = self.transformer_scorer(embeddings, mask_position, rel_embeddings, ent_embeddings)
                loss = torch.nn.functional.cross_entropy(score, label, reduction='none')
                return loss.mean(), score
    
    def cvt_transe_scorer(self, embeddings):
        cvt_emb_e = torch.mean(torch.stack(embeddings[1::2],dim=0).squeeze(1), dim=0)
        cvt_emb_r = torch.mean(torch.stack(embeddings[0::2],dim=0).squeeze(1), dim=0)
        cvt_emb = (self.cvt_e_w(cvt_emb_e) + self.cvt_r_w(cvt_emb_r))/2
        score = self.transe_fun(cvt_emb, embeddings[0], embeddings[1]).unsqueeze(0)
        for _, ele in enumerate(embeddings[2::2]):
            score = torch.cat([score, self.transe_fun(cvt_emb, embeddings[2+2*_], embeddings[2+2*_+1]).unsqueeze(0)])
        return score.mean(dim=0)
    
    def cvt_rotate_scorer(self, embeddings):
        cvt_emb_e = torch.mean(torch.stack(embeddings[1::2],dim=0).squeeze(1), dim=0)
        cvt_emb_r = torch.mean(torch.stack(embeddings[0::2],dim=0).squeeze(1), dim=0)
        cvt_emb = (self.cvt_e_w(cvt_emb_e) + self.cvt_r_w(cvt_emb_r))/2
        score = self.rotate_fun(cvt_emb, embeddings[0], embeddings[1])
        for _, ele in enumerate(embeddings[2::2]):
            score = torch.stack([score, self.rotate_fun(cvt_emb, embeddings[2+2*_], embeddings[2+2*_+1])]).mean(dim=0)
        score = self.gamma.item() - score.sum(dim=2)
        return score
    
    def cvt_ingram_scorer(self, embeddings):
        layer_emb_ent = self.ent_proj1(emb_ent)
        layer_emb_rel = self.rel_proj1(emb_rel)
        
        for layer_idx, layer in enumerate(self.layers_rel):
            layer_emb_rel = layer(layer_emb_rel, relation_triplets) + \
                            self.res_proj_rel[layer_idx](layer_emb_rel)
            layer_emb_rel = self.act(layer_emb_rel)
        
        for layer_idx, layer in enumerate(self.layers_ent):

            layer_emb_ent = layer(layer_emb_ent, layer_emb_rel, triplets) + \
                            self.res_proj_ent[layer_idx](layer_emb_ent)
            layer_emb_ent = self.act(layer_emb_ent)


        return self.ent_proj2(layer_emb_ent), self.rel_proj2(layer_emb_rel)
    
    def cvt_dicgrl_scorer(self, embeddings, fact):
        embeddings = [emb.squeeze(0) for emb in embeddings]
        fact = [e.squeeze(0) for e in fact]
        cvt_emb_e = torch.mean(torch.stack(embeddings[1::2],dim=0).squeeze(1), dim=0)
        cvt_emb_r = torch.mean(torch.stack(embeddings[0::2],dim=0).squeeze(1), dim=0)
        cvt_emb = (self.cvt_e_w(cvt_emb_e) + self.cvt_r_w(cvt_emb_r))/2

        head_o = cvt_emb
        rel = embeddings[0]
        tail_o = embeddings[1]
        head_ori = head_o.view(-1, self.K, self.emb_s)
        tail_ori = tail_o.view(-1, self.K, self.emb_s)
        # calculate k attention
        tmp = self.Rel_attention(fact[0])  # [b_s, k]
        att = self.softmax(tmp)
        # choose top n
        sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
        top_indices_in = sorted_indices_in[:, :self.top_n].to(self.args.gpu)
        head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
        head = head_ori.view(-1, self.top_n * self.emb_s)
        tail = tail_ori.view(-1, self.top_n * self.emb_s)
        x = head + rel - tail
        score = torch.norm(x, p=1, dim=1)
        
        for _, ele in enumerate(embeddings[2::2]):
            head_o = cvt_emb
            tail_o = embeddings[1+2*_+2]
            rel = embeddings[2*_+2]
            head_ori = head_o.view(-1, self.K, self.emb_s)
            tail_ori = tail_o.view(-1, self.K, self.emb_s)
            # calculate k attention
            tmp = self.Rel_attention(fact[2*_+2])  # [b_s, k]
            att = self.softmax(tmp)
            # choose top n
            sorted_att, sorted_indices_in = torch.sort(att, dim=-1, descending=True)
            top_indices_in = sorted_indices_in[:, :self.top_n].to(self.args.gpu)
            head_ori = head_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
            tail_ori = tail_ori.gather(1, top_indices_in.unsqueeze(-1).expand(-1, self.top_n, self.emb_s))
            head = head_ori.view(-1, self.top_n * self.emb_s)
            tail = tail_ori.view(-1, self.top_n * self.emb_s)
            x = head + rel - tail
            score = torch.stack([score, torch.norm(x, p=1, dim=1)]).mean(dim=0)
        return score
    
    def margin_loss(self, score, label):
        p_score, n_score = self.split_pn_score(score, label)
        y = torch.Tensor([-1]).to(self.args.gpu)
        # 期望p_score小于n_score，因此y为-1
        loss = self.margin_loss_func(p_score, n_score, y)/score.size(0)
        return loss
    # Transfomrer, gran, HINGE, GRAN
    
    def hinge_loss(self, score, label):
        score = score.squeeze(1) * label * (-1)
        return self.softloss_func(score).mean()
    
    def shrinke_loss(self, pred, true_label):
        return self.bcelogitloss(pred.squeeze(1), true_label.float())
    
    def shift(self, v, sh):
        y = torch.cat((v[:, sh:], v[:, :sh]), dim=1)
        return y

    def conv3d_process(self, batch):
        if len(batch) == 3:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_2(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 4:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_3(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 5:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_4(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 6:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e5 = batch[5].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_5(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 7:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e5 = batch[5].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e6 = batch[6].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_6(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 8:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e5 = batch[5].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e6 = batch[6].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e7 = batch[7].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_7(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 9:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e5 = batch[5].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e6 = batch[6].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e7 = batch[7].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e8 = batch[8].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_8(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        if len(batch) == 10:
            r = batch[0].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e1 = batch[1].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e2 = batch[2].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e3 = batch[3].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e4 = batch[4].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e5 = batch[5].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e6 = batch[6].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e7 = batch[7].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e8 = batch[8].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            e9 = batch[9].view(-1, 1, self.args.emb_dim1, self.args.emb_dim2)
            cube = torch.cat((r, e1, e2, e3, e4, e5, e6, e7, e8, e9), dim=1)
            x = cube.permute(0, 2, 3, 1)
            x = x.unsqueeze(1)
            x = self.bn1(x)
            x = self.conv_layer_9(x)
            x = x.permute(0, 4, 1, 2, 3)
            x = self.pool(x)

        # x =  self.nonlinear(x)
        # x = self.bn2(x)
        x = x.view(-1, self.conv_size)
        # x = self.nonlinear(x)

        # x = self.dropout(x)
        x = self.dropout_3d(x)

        return x

    def convolve(self, e_emb, r_emb, pos):

        x = e_emb
        x = self.inp_drop(x)

        k1 = self.fc_rel_2(r_emb)
        k1 = k1.view(-1, 1, 3, 1, 1)
        k1 = k1.view(e_emb.size(0)*3, 1, 1, 1)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k1, groups=e_emb.size(0))


        one_hot_target = (pos == torch.arange(self.arity_lst[-1]).reshape(self.arity_lst[-1])).float().to(self.args.gpu)
        poses = one_hot_target.repeat(r_emb.shape[0]).view(-1, self.arity_lst[-1])
        one_hot_target.requires_grad = False
        poses.requires_grad = False

        k = self.fc_pos(poses)
        k = k.view(e_emb.size(0)*3, 3, 1, 1)
        x = F.conv2d(x, k, groups=e_emb.size(0), stride=2)
        x = x.view(e_emb.size(0), 1, 3, 1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.bn3(x)
        # x = self.dropout_2d(x)
        return x

    def conv2d_process(self, batch):
        if len(batch) == 3:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            # re1 = F.conv2d(e1, r).view(-1, 1, 1, self.args.dim)
            # re2 = F.conv2d(e2, r).view(-1, 1, 1, self.args.dim)
            # conv_r = F.conv1d(r, self.rel_filters)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2), dim=1)

            x = self.pool1d(x)

            # x = self.nonlinear(x)
            x = x.view(e1.shape[0], -1)
            # x = torch.tanh(x)
            x = self.nonlinear(x)
            x = self.dropout(x)

            x = self.fc_2(x)
            # x = self.nonlinear(x)
            return x

        if len(batch) == 4:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_3(x)
            # x = self.nonlinear(x)

            return x

        if len(batch) == 5:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)

            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_4(x)
            # x = self.nonlinear(x)

            return x

        if len(batch) == 6:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            e5 = batch[5].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)

            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5), dim=1)
            x = self.pool1d(x)


            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_5(x)
            # x = self.nonlinear(x)

            return x

        if len(batch) == 7:
            r = batch[0].view(-1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            e5 = batch[5].view(-1, 1, 1, self.args.dim)
            e6 = batch[6].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_6(x)
            # x = self.nonlinear(x)
            return x

        if len(batch) == 8:
            r = batch[0].view(-1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            e5 = batch[5].view(-1, 1, 1, self.args.dim)
            e6 = batch[6].view(-1, 1, 1, self.args.dim)
            e7 = batch[7].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_7(x)
            # x = self.nonlinear(x)
            return x


        if len(batch) == 9:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            e5 = batch[5].view(-1, 1, 1, self.args.dim)
            e6 = batch[6].view(-1, 1, 1, self.args.dim)
            e7 = batch[7].view(-1, 1, 1, self.args.dim)
            e8 = batch[8].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3)
            conv_e8 = self.convolve(e8, r, 7).permute(0, 2, 1, 3)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_8(x)

            return x

        if len(batch) == 10:
            r = batch[0].view(-1, 1, 1, self.args.dim)
            e1 = batch[1].view(-1, 1, 1, self.args.dim)
            e2 = batch[2].view(-1, 1, 1, self.args.dim)
            e3 = batch[3].view(-1, 1, 1, self.args.dim)
            e4 = batch[4].view(-1, 1, 1, self.args.dim)
            e5 = batch[5].view(-1, 1, 1, self.args.dim)
            e6 = batch[6].view(-1, 1, 1, self.args.dim)
            e7 = batch[7].view(-1, 1, 1, self.args.dim)
            e8 = batch[8].view(-1, 1, 1, self.args.dim)
            e9 = batch[9].view(-1, 1, 1, self.args.dim)
            conv_e1 = self.convolve(e1, r, 0).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e2 = self.convolve(e2, r, 1).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e3 = self.convolve(e3, r, 2).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e4 = self.convolve(e4, r, 3).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e5 = self.convolve(e5, r, 4).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e6 = self.convolve(e6, r, 5).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e7 = self.convolve(e7, r, 6).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e8 = self.convolve(e8, r, 7).permute(0, 2, 1, 3).to(self.args.gpu)
            conv_e9 = self.convolve(e9, r, 8).permute(0, 2, 1, 3).to(self.args.gpu)
            x = torch.cat((conv_e1, conv_e2, conv_e3, conv_e4, conv_e5, conv_e6, conv_e7, conv_e8, conv_e9), dim=1)
            x = self.pool1d(x)

            x = x.view(e1.shape[0], -1)
            # x = self.bn4(x)
            # x = torch.tanh(x)
            x = self.nonlinear(x)

            x = self.dropout(x)

            x = self.fc_9(x)

            return x

    def hyconve_scorer(self, test_batch):
        test_batch = torch.cat([b.unsqueeze(0) for b in test_batch], dim=0)
        r = torch.mean(test_batch[1::2,:], dim=0)
        ents = torch.transpose(test_batch[0::2,:], 0, 1)
        
        # r = self.rel_embeddings(torch.mean(batch[1::2,:], dim=0))
        # ents = self.ent_embeddings(batch[0::2,:])
        # seq_len = batch.shape[0]//2+2 - 1
        # pos = torch.arange(seq_len, dtype=torch.long).to(self.dataset.device)
        # pos = pos.unsqueeze(0).expand_as(batch[:-1])  # [seq_len] -> [batch_size, seq_len]
        # ents = self.ent_embeddings(batch[1:]) + self.pos_embeddings(pos)
        e1 = ents[:, 0]
        e2 = ents[:, 1]
        
        
        # r = self.rel_embeddings(test_batch[0])
        # ents = self.ent_embeddings(test_batch[1:])
        # # seq_len = test_batch.shape[0]//2+2 -1
        # # pos = torch.arange(seq_len, dtype=torch.long).to(self.dataset.device)
        # # pos = pos.unsqueeze(0).expand_as(test_batch[:-1])  # [seq_len] -> [batch_size, seq_len]
        # # ents = self.ent_embeddings(test_batch[1:]) + self.pos_embeddings(pos)
        # e1 = ents[:, 0]
        # e2 = ents[:, 1]
        if test_batch.shape[0]//2+2 == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[0]//2+2 == 4:
            e3 = ents[:, 2]
            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[0]//2+2 == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]

            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[0]//2+2 == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)
        if test_batch.shape[0]//2+2 == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[0]//2+2 == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[0]//2+2 == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        if test_batch.shape[0]//2+2 == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]
            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            score = x.view(-1)

        return score
    
    def neuinfer_loss(self, score, label):
        loss = -torch.mean(label * torch.log(torch.clamp(score, 1e-10, 1.0)) + (1 - label) * torch.log(torch.clamp(1 - score, 1e-10, 1.0)))
        return loss
    
    def hyconve_loss(self, batch, labels):
        batch = torch.cat([b.unsqueeze(0) for b in batch], dim=0)
        
        r = torch.mean(batch[1::2,:], dim=0)
        ents = torch.transpose(batch[0::2,:], 0, 1)
        
        # r = self.rel_embeddings(torch.mean(batch[1::2,:], dim=0))
        # ents = self.ent_embeddings(batch[0::2,:])
        
        # seq_len = batch.shape[0]//2+2 - 1
        # pos = torch.arange(seq_len, dtype=torch.long).to(self.dataset.device)
        # pos = pos.unsqueeze(0).expand_as(batch[:-1])  # [seq_len] -> [batch_size, seq_len]
        # ents = self.ent_embeddings(batch[1:]) + self.pos_embeddings(pos)
        e1 = ents[:, 0]
        e2 = ents[:, 1]
        if batch.shape[0]//2+2 == 3:
            x1 = self.conv3d_process((r, e1, e2))
            x2 = self.conv2d_process((r, e1, e2))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)

            batch_score = -x.view(-1)
            # l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2)
            for p in self.conv_layer_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_2.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)


            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[0]//2+2 == 4:
            e3 = ents[:, 2]

            x1 = self.conv3d_process((r, e1, e2, e3))
            x2 = self.conv2d_process((r, e1, e2, e3))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2)

            for p in self.conv_layer_3.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_3.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[0]//2+2 == 5:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            x1 = self.conv3d_process((r, e1, e2, e3, e4))
            x2 = self.conv2d_process((r, e1, e2, e3, e4))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)

            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2)

            for p in self.conv_layer_4.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_4.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[0]//2+2 == 6:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2) + torch.mean(re5 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2)

            for p in self.conv_layer_5.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_5.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular
        if batch.shape[0]//2+2 == 7:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]


            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2) + torch.mean(re5 ** 2) + torch.mean(re6 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2)

            for p in self.conv_layer_6.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_6.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[0]//2+2 == 8:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2) + torch.mean(re5 ** 2) + torch.mean(re6 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2)

            for p in self.conv_layer_7.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_7.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular


        if batch.shape[0]//2+2 == 9:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)
            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2) + torch.mean(re5 ** 2) + torch.mean(re6 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2) + torch.mean(e8 ** 2)

            for p in self.conv_layer_8.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_8.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        if batch.shape[0]//2+2 == 10:
            e3 = ents[:, 2]
            e4 = ents[:, 3]
            e5 = ents[:, 4]
            e6 = ents[:, 5]
            e7 = ents[:, 6]
            e8 = ents[:, 7]
            e9 = ents[:, 8]

            x1 = self.conv3d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x2 = self.conv2d_process((r, e1, e2, e3, e4, e5, e6, e7, e8, e9))
            x = x1 + x2
            # x = x2
            x = self.nonlinear(x)
            x = self.bn4(x)

            x = self.dropout(x)
            x = self.fc_layer(x)
            batch_score = -x.view(-1)
            # l2_regular = torch.mean(re1 ** 2) + torch.mean(re2 ** 2) + torch.mean(re3 ** 2) + torch.mean(re4 ** 2) + torch.mean(re5 ** 2) + torch.mean(re6 ** 2)
            l2_regular = torch.mean(r ** 2) + torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) + torch.mean(e5 ** 2) + torch.mean(e6 ** 2) + torch.mean(e7 ** 2) + torch.mean(e8 ** 2) + torch.mean(e9 ** 2)

            for p in self.conv_layer_9.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_layer.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_rel_2.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_pos.parameters():
                l2_regular += p.norm(2)
            for p in self.fc_9.parameters():
                l2_regular += p.norm(2)
            for p in self.pool.parameters():
                l2_regular += p.norm(2)
            for p in self.pool1d.parameters():
                l2_regular += p.norm(2)

            mean = torch.mean(self.criterion(labels * batch_score))
            regular = self.lmbda * l2_regular

        return mean + regular

    def hinge_scorer(self, embeddings):
        hrt_concat = torch.cat((embeddings[0].unsqueeze(1), embeddings[1].unsqueeze(1), embeddings[2].unsqueeze(1)), 1).unsqueeze(1)
        hrt_vectors1 = self.conv1(hrt_concat)
        hrt_vectors1 = self.batchNorm1(hrt_vectors1)
        hrt_vectors1 = F.relu(hrt_vectors1).squeeze(3)
        hrt_vectors1 = hrt_vectors1.view(hrt_vectors1.size(0), -1).unsqueeze(2)
        
        if len(embeddings) > 3:
            hrt_concat = torch.cat((embeddings[0].unsqueeze(1), embeddings[1].unsqueeze(1), embeddings[2].unsqueeze(1)), 1)
            for _, ele in enumerate(range(len(embeddings[3::2]))):
                hrt_qv_concat = torch.cat((hrt_concat, embeddings[3+_*2].unsqueeze(1), embeddings[3+_*2+1].unsqueeze(1)), 1).unsqueeze(1)
                hrt_qv_vector2 = self.conv2(hrt_qv_concat)
                hrt_qv_vector2 = self.batchNorm2(hrt_qv_vector2)
                hrt_qv_vector2 = F.relu(hrt_qv_vector2)
                hrt_qv_vector2 = hrt_qv_vector2.view(hrt_qv_vector2.size(0), -1).unsqueeze(2)
                hrt_vectors1 = torch.cat((hrt_vectors1, hrt_qv_vector2), 2)
        min_val, _ = torch.min(hrt_vectors1, 2)
        score = self.i_FCN_net(min_val)
        return score
    
    def shrinke_scorer(self, embeddings):
        # def forward(self, sub, rel, quals):
        sub_emb = embeddings[0]
        trans_emb = embeddings[1]
        diag_emb = self.diag_w(embeddings[1])
        offset_emb = self.offset_w(embeddings[1])
        obj_emb = embeddings[2]

        sub_trans_emb = self.rot_trans(sub_emb, diag_emb, trans_emb)
        query_boxes = Box(sub_trans_emb - F.softplus(offset_emb), sub_trans_emb + F.softplus(offset_emb))

        # 用qualifier转移
        if len(embeddings) != 3:
            
            qual_obj_emb = embeddings[4::2]
            qual_rel_trans_emb = embeddings[3::2]
            qual_rel_diag_emb = embeddings[3::2]
            qual_rel_offset_emb = embeddings[3::2]
            
            qual_obj_emb = torch.cat([b.unsqueeze(1) for b in qual_obj_emb], dim=1)
            qual_rel_trans_emb = torch.cat([b.unsqueeze(1) for b in qual_rel_trans_emb], dim=1)
            qual_rel_diag_emb = torch.cat([self.diag_w(b).unsqueeze(1) for b in qual_rel_diag_emb], dim=1)
            qual_rel_offset_emb = torch.cat([self.offset_w(b).unsqueeze(1) for b in qual_rel_offset_emb], dim=1)

            query_boxes = self.shrinking(query_boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb)
            
        neg_dist = - self.point2box_distance(obj_emb, query_boxes) # bsz*num_ent
        return neg_dist
        bh = torch.index_select(self.bh, 0, sub) # bsz*1
        bt = self.bt.t() # b=num_ent*1
        return torch.add(torch.add(neg_dist, bh), bt)
    
    def neuinfer_scorer(self, embeddings):
        hrt_concat = torch.cat((embeddings[0], embeddings[1], embeddings[2]), 1)
        
        if self.hrtFCNs_layers == 1:
            hrtFCNs_res1 = self.t_1_linear(hrt_concat)
            hrtFCNs_res1 = F.relu(hrtFCNs_res1)
        else:  # hrtFCNs_layers = 2
            hrtFCNs_res1 = self.t_2_linear(hrt_concat)
            hrtFCNs_res1 = F.relu(hrtFCNs_res1)
            hrtFCNs_res1 = self.t_3_linear(hrtFCNs_res1)
            hrtFCNs_res1 = F.relu(hrtFCNs_res1)
        validity = self.hrt_score_linear(hrtFCNs_res1)
        validity = torch.sigmoid(validity).squeeze(1)

        if len(embeddings) > 3:
            qv_concat = torch.cat((embeddings[3], embeddings[4]), 1)
            o_hrtaivi_list = self.hrtavFCNs(hrt_concat, qv_concat)
            for _, ele in enumerate(range(len(embeddings[5::2]))):
                qv_concat = torch.cat((embeddings[5+_*2], embeddings[5+_*2+1]), 1)
                o_hrtaivi_list_ = self.hrtavFCNs(hrt_concat, qv_concat)
                o_hrtaivi_list = torch.cat([o_hrtaivi_list, o_hrtaivi_list_], dim=0)
            o_hrtav = torch.min(o_hrtaivi_list, dim=0)[0]
            compatibility = self.Tkv_score_linear(o_hrtav)
            compatibility = torch.sigmoid(compatibility).squeeze(1)
            return self.weight * validity + (1-self.weight) * compatibility
        else:
            return validity

    def transformer_scorer(self, embeddings, mask_position, rel_embeddings, ent_embeddings):
        if self.args.scorer_func == 'i_stare':
            # embeddings = [self.input_dropout(self.input_norm(ele)) for ele in embeddings]
            embeddings[mask_position] = self.mask_emb.repeat(embeddings[0].shape[0],1)
            x = torch.stack(embeddings, 0)
            positions = torch.arange(x.shape[0], dtype=torch.long, device=self.args.gpu).repeat(x.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            x += pos_embeddings
            x = self.TransformerEncoder(x)
            x = x.transpose(1,0)
        elif self.args.scorer_func in ['i_hytransformer']:
            embeddings = [self.input_dropout(self.input_norm(ele)) for ele in embeddings]
            embeddings[mask_position] = self.mask_emb.repeat(embeddings[0].shape[0],1)
            x = torch.stack(embeddings, 0)
            positions = torch.arange(x.shape[0], dtype=torch.long, device=self.args.gpu).repeat(x.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            x += pos_embeddings
            x = self.TransformerEncoder(x)
            x = x.transpose(1,0)
        elif self.args.scorer_func == 'i_hahe':
            embeddings = [self.input_dropout(self.input_norm(ele)) for ele in embeddings]
            embeddings[mask_position] = self.mask_emb.repeat(embeddings[0].shape[0],1)
            edge_labels = self.edge_labels[:len(embeddings),:len(embeddings)]
            # L, B, H
            edge_query = self.edge_query_embedding(edge_labels).unsqueeze(0).repeat(embeddings[0].shape[0], 1, 1, 1)
            edge_key = self.edge_key_embedding(edge_labels).unsqueeze(0).repeat(embeddings[0].shape[0], 1, 1, 1)
            edge_value = self.edge_value_embedding(edge_labels).unsqueeze(0).repeat(embeddings[0].shape[0], 1, 1, 1)
            x = torch.stack(embeddings, 0).transpose(1,0)
            for layer in self.layers:
                x = layer(x, edge_key, edge_value, edge_query)
        elif self.args.scorer_func in ['i_gran']:
            embeddings = [self.input_dropout(self.input_norm(ele)) for ele in embeddings]
            embeddings[mask_position] = self.mask_emb.repeat(embeddings[0].shape[0],1)
            edge_labels = self.edge_labels[:len(embeddings),:len(embeddings)]
            edge_key = self.edge_key_embedding(edge_labels).unsqueeze(0).repeat(embeddings[0].shape[0], 1, 1, 1)
            edge_value = self.edge_value_embedding(edge_labels).unsqueeze(0).repeat(embeddings[0].shape[0], 1, 1, 1)
            # x = torch.stack(embeddings, 0).to(self.args.gpu)
            # x = self.GraphTransformerEncoder(x.transpose(1,0), edge_value, edge_key).transpose(1,0)
            x = torch.stack(embeddings, 0).transpose(1,0)
            for layer in self.layers:
                x = layer(x, edge_key, edge_value, 'edge_query')
            
        x = x[:, mask_position]
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        
        if self.args.scorer_func == 'ReDA':
            if mask_position % 2 == 1:
                y = torch.mm(x, ent_embeddings.transpose(0, 1))
            else:
                y = torch.mm(x, rel_embeddings.transpose(0, 1))
        else:
            if mask_position % 2 == 0:
                y = torch.mm(x, ent_embeddings.transpose(0, 1))
            else:
                y = torch.mm(x, rel_embeddings.transpose(0, 1))
        return y

    def split_pn_score(self, score, label):
        '''
        Get the scores of positive and negative facts
        :param score: scores of all facts
        :param label: positive facts: 1, negative facts: -1
        :return:
        '''
        p_score = score[torch.where(label>0)]
        n_score = (score[torch.where(label<0)]).reshape(-1, self.args.neg_ratio).mean(dim=1)
        return p_score, n_score

    def transe_fun(self, s, r, o):
        s = nn.functional.normalize(s, 2, -1)
        r = nn.functional.normalize(r, 2, -1)
        o = nn.functional.normalize(o, 2, -1)
        return torch.norm(s + r - o, 2, -1)

    def distmult_fun(self, s, r, o):
        output = (s * r * o).sum(dim = -1)
        return output

    def rotate_fun(self, head, relation, tail):
        re_head, im_head = torch.chunk(head, 2, dim=-1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=-1)

        pi = 3.14159265358979323846
        # Make phases of relations uniformly distributed in [-pi, pi]
        relation = self.rel_reduce(relation)
        phase_relation = relation.squeeze(0) / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return score.sum(dim=-1)

    def rot_trans(self, sub_emb, diag_emb, trans_emb):
        return givens_rotations(diag_emb, sub_emb) + trans_emb

    def shrinking(self, boxes, trans_emb, diag_emb, offset_emb, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb):
        trans_embedded = trans_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        diag_embedded = diag_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        offset_embedded = offset_emb.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)

        rel_key_value_emb = torch.cat((trans_embedded, diag_embedded, offset_embedded, qual_rel_trans_emb, qual_rel_diag_emb, qual_rel_offset_emb, qual_obj_emb), -1) 

        box_mins = boxes.min_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_maxs = boxes.max_embed.unsqueeze(1).repeat(1,qual_obj_emb.shape[1],1)
        box_widths = box_maxs - box_mins

        shrinking_min = F.relu( self.min_fc(rel_key_value_emb)*box_widths)
        shrinking_max = F.relu( self.max_fc(rel_key_value_emb)*box_widths)

        box_mins = box_mins + shrinking_min
        box_maxs = box_maxs - shrinking_max
        box_offset = F.softplus(box_maxs - box_mins)/2
        centers = (box_mins + box_maxs)/2
        box_mins = centers - box_offset
        box_maxs = centers + box_offset
        boxes = Box(torch.max(box_mins,1)[0], torch.min(box_maxs,1)[0]) 
        return boxes

    def point2box_distance(self, points, boxes):  
        centres = 0.5 * (boxes.min_embed + boxes.max_embed)
        boxes_min = boxes.min_embed
        boxes_max = boxes.max_embed

        dist_c = torch.norm(centres - points, p=1, dim=-1).unsqueeze(1)
        dist_m = torch.norm(boxes_min - points, p=1, dim=-1).unsqueeze(1)
        dist_M = torch.norm(boxes_max - points, p=1, dim=-1).unsqueeze(1)
        # dist_c = torch.cdist(centres, points, p=1)
        # dist_m = torch.cdist(boxes_min, points, p=1)
        # dist_M = torch.cdist(boxes_max, points, p=1)
        dist_mM = torch.norm(boxes_max - boxes_min,p=1, dim=-1, keepdim=True)

        dist_inside = dist_c/dist_mM
        dist_outside = F.relu(dist_m + dist_M - dist_mM)**2
        dist = dist_inside + dist_outside
        return dist 

    def hrtavFCNs(self, hrtFCNs_res, av_embed):
        """
        hrtavFCNs: Obtain the interaction vectors of hrt and all the atribute-value pairs via g-FCN
        """
        hrt_av_compability = self.g_theta(hrtFCNs_res, av_embed)
        hrt_av_compability = hrt_av_compability.view(1, hrt_av_compability.shape[0], hrt_av_compability.shape[1])
        return hrt_av_compability
        
    def g_theta(self, o_i, o_j):
        """
        g_theta: Obtain the interaction vector of the o_i (hrt) and o_j (aivi)
        """
        if self.hrtavFCNs_layers == 1:
            g_1 = self.g_1_linear(torch.cat([o_i, o_j], dim=1))
            # g_1 = self.layer_1_norm(g_1)
            g_1 = F.relu(g_1)
        else:  # hrtavFCNs_layers = 2
            g_1 = self.g_2_linear(torch.cat([o_i, o_j], dim=1))
            # g_1 = self.layer_2_norm(g_1)
            g_1 = F.relu(g_1)
            g_1 = self.g_3_linear(g_1)
            # g_1 = self.layer_3_norm(g_1)
            g_1 = F.relu(g_1)
        return g_1


class TransformerLayer_gran(nn.Module):
    def __init__(self, args, emb_dim, hid_dim, heads, dropout_prob, model='none') -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(emb_dim).to(args.gpu)
        self.attention = MultiHeadAttention_four(args, emb_dim, heads, dropout_prob, model).to(args.gpu)
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_ffn = nn.LayerNorm(emb_dim).to(args.gpu)
        self.ffn = FeedForward_gran(emb_dim, hid_dim).to(args.gpu)

    def forward(self, x: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        attn = self.attention(query=x, key=x, value=x, edge_key=edge_key, edge_value=edge_value, edge_query=edge_query)
        x = self.norm_attention(x + self.dropout(attn))
        ff = self.ffn(x)
        x = self.norm_ffn(x + self.dropout(ff))
        return x
    
class MultiHeadAttention_four(nn.Module):
    def __init__(self, args, hidden_dim, heads, dropout_prob, model) -> None:
        super().__init__()
        assert hidden_dim % heads == 0
        self.dim = hidden_dim // heads
        self.heads = heads
        self.model = model
    
        self.query = PrepareForMultiHeadAttention_gran(args, hidden_dim, heads)
        self.key = PrepareForMultiHeadAttention_gran(args, hidden_dim, heads)
        self.value = PrepareForMultiHeadAttention_gran(args, hidden_dim, heads)
        self.pos = PrepareForMultiHeadAttention_gran(args, hidden_dim, heads)
        
        self.softmax = nn.Softmax(dim=-1)
        self.output = nn.Linear(hidden_dim, hidden_dim).to(args.gpu)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.scale = 1 / math.sqrt(self.dim)
        # trasformer-xl
        self.r_w_bias = nn.Parameter(torch.Tensor(heads, self.dim)).to(args.gpu) # u
        self.r_r_bias = nn.Parameter(torch.Tensor(heads, self.dim)).to(args.gpu) # v

    def forward(self, *, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, edge_key: torch.Tensor, edge_value: torch.Tensor, edge_query: torch.Tensor):
        # query/key/value: (batch, seq_len, hidden_dim)
        # graph: (batch, kinds, query, key)
        shape = query.shape[:-1]
        query = self.query(query)   # (batch, seq_len, head, hidden)
        key = self.key(key)         # (batch, seq_len, head, hidden)
        value = self.value(value)   # (batch, seq_len, head, hidden)
        # edge_*: (batch, kinds, head, hidden)
        # k 也表示 seq_len
        if self.model == 'i_gran':
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) + torch.einsum("bqhd,bqkd->bhqk", query, edge_key) + torch.einsum("bkqd,bkhd->bhqk", edge_query, key) + torch.einsum("bkqd,bqkd->bqk", edge_query, edge_key).unsqueeze(1)
        scores = scores * self.scale
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum("bhqk,bkhd->bqhd", attn, value) + torch.einsum("bhqk,bqkd->bqhd", attn, edge_value)
        x = x.reshape(*shape, -1)

        return self.output(x)  # (batch, query, hidden_dim)
    
class PrepareForMultiHeadAttention_gran(nn.Module):
    def __init__(self, args, hidden_dim, heads, bias=False, use_node=False) -> None:
        super().__init__()
        self.heads = heads
        self.use_node = use_node      
        self.args = args 

        if self.use_node is True:
            self.layer_s=nn.Linear(hidden_dim,hidden_dim).to(self.args.gpu)
            self.layer_r=nn.Linear(hidden_dim,hidden_dim).to(self.args.gpu)
            self.layer_o=nn.Linear(hidden_dim,hidden_dim).to(self.args.gpu)
            self.layer_a=nn.Linear(hidden_dim,hidden_dim).to(self.args.gpu)
            self.layer_v=nn.Linear(hidden_dim,hidden_dim).to(self.args.gpu)
        else:
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=bias).to(self.args.gpu)


    def forward(self, x : torch.Tensor):
        shape = x.shape[:-1]

        if self.use_node is False:
            x = self.linear(x)
        else:
            device=x.device
            max_seq_len=x.size(1)
            mask_s = torch.tensor([1]+[0]*(max_seq_len-1)).to(device)
            mask_r = torch.tensor([0,1]+[0]*(max_seq_len-2)).to(device)
            mask_o = torch.tensor([0,0,1]+[0]*(max_seq_len-3)).to(device)
            mask_a = torch.tensor([0,0,0]+[1,0]*int(((max_seq_len-3)/2))).to(device)
            mask_v = torch.tensor([0,0,0]+[0,1]*int(((max_seq_len-3)/2))).to(device)

            x_s=self.layer_s(torch.mul(x,mask_s[:,None].expand(-1,x.size(-1))))
            x_r=self.layer_r(torch.mul(x,mask_r[:,None].expand(-1,x.size(-1))))
            x_o=self.layer_o(torch.mul(x,mask_o[:,None].expand(-1,x.size(-1))))
            x_a=self.layer_a(torch.mul(x,mask_a[:,None].expand(-1,x.size(-1))))
            x_v=self.layer_v(torch.mul(x,mask_v[:,None].expand(-1,x.size(-1))))
                            
            x=(x_s+x_r+x_o+x_a+x_v) 
      
        return x.reshape(*shape, self.heads, -1)

class FeedForward_gran(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        activation='elu'
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == 'elu':
            act = nn.ELU()
        elif activation == 'tanh':
            act = nn.Tanh()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), act, nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.layer(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale, edge_key, edge_value):
        # edge_key L,L,H
        # q BN,L,H
        # BN,L,L
        attn_score = torch.bmm(q, k.transpose(1, 2)) + torch.bmm(edge_key, q.permute(1,2,0)).transpose(0,2)
        attn_score = attn_score * scale
        attn_score = self.softmax(attn_score)
        attn_score = self.dropout(attn_score)
        # q M, B*N, H
        # edge_bias M, B*N, M
        # attn_score B,M,M
        # output B,M,H
        output = torch.bmm(attn_score, v) + torch.bmm(attn_score.transpose(0, 1), edge_value).transpose(0,1)

        # attn B, M, M->M,B,M
        # edge_value M,M,H
        # edge_bias M,B,H->B,M,H
        return output


class MultiHeadAttention_thire(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.linear_final = nn.Linear(embed_dim, embed_dim)
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key, value, edge_value, edge_key):
        max_seq_length, batch_size, embed_dim = key.size()
        residual = query
        head_dim = embed_dim // self.num_heads
        key = self.linear_k(key)
        query = self.linear_q(query)
        value = self.linear_v(value)

        # M, B*N, H
        key = key.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)
        query = query.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)
        value = value.contiguous().view(max_seq_length, batch_size*self.num_heads, head_dim).transpose(0,1)

        scale = float(head_dim) ** -0.5
        context = self.dot_product_attention(query, key, value, scale, edge_key, edge_value)
        context = context.transpose(0,1).contiguous().view(max_seq_length, batch_size, embed_dim)

        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(output+residual)

        return output


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, context):
        context = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(context))))
        context = context + self.dropout(context)
        context = self.layer_norm(context)
        return context


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention_thire(embed_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(embed_dim, hidden_dim)
    
    def forward(self, inputs, edge_value, edge_key):
        output = self.attention(inputs, inputs, inputs, edge_value, edge_key)
        output = self.feed_forward(output)
        return output


class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_heads=4, num_transformer_layers=12, embed_dim=100, hidden_dim=100, dropout=0.2):
        super().__init__()
        num_layers = num_transformer_layers
        self.num_heads = num_heads
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(embed_dim, hidden_dim, self.num_heads, dropout) for _ in range(num_layers)]
        )
    def forward(self, facts, edge_value, edge_key):
        facts = facts.transpose(0, 1)
        # L, B, H
        for encoder in self.encoder_layers:
            facts = encoder(facts, edge_value, edge_key)
        return facts
    


class Box:
    def __init__(self, min_embed, max_embed):
        self.min_embed = min_embed
        self.max_embed = max_embed
        self.delta_embed = max_embed - min_embed

    def volumes(self, dim=-1):
        return F.softplus(self.delta_embed).prod(dim, keepdim=True).clamp(1e-5,1e5)



class InGramEntityLayer(nn.Module):
    def __init__(self, dim_in_ent, dim_out_ent, dim_rel, bias = True, num_head = 8):
        super(InGramEntityLayer, self).__init__()

        self.dim_out_ent = dim_out_ent
        self.dim_hid_ent = dim_out_ent // num_head
        assert dim_out_ent == self.dim_hid_ent * num_head
        self.num_head = num_head

        self.attn_proj = nn.Linear(2 * dim_in_ent + dim_rel, dim_out_ent, bias = bias)
        self.attn_vec = nn.Parameter(torch.zeros((1, num_head, self.dim_hid_ent)))
        self.aggr_proj = nn.Linear(dim_in_ent + dim_rel, dim_out_ent, bias = bias)

        self.dim_rel = dim_rel
        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.bias = bias
        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_ent, emb_rel, triplets): 
        num_ent = len(emb_ent)
        num_rel = len(emb_rel)
        head_idxs = triplets[..., 0]
        rel_idxs = triplets[..., 1]
        tail_idxs = triplets[..., 2]

        ent_freq = torch.zeros((num_ent, )).cuda().index_add(dim = 0, index = tail_idxs, \
                                                             source = torch.ones_like(tail_idxs, dtype = torch.float).cuda()).unsqueeze(dim = 1)

        self_rel = torch.zeros((num_ent, self.dim_rel)).cuda().index_add(dim=0, index = tail_idxs, source = emb_rel[rel_idxs])/ent_freq

        # add self-loops
        emb_rels = torch.cat([emb_rel[rel_idxs], self_rel], dim = 0)
        head_idxs = torch.cat([head_idxs, torch.arange(num_ent).cuda()], dim = 0)
        tail_idxs = torch.cat([tail_idxs, torch.arange(num_ent).cuda()], dim = 0)

        concat_mat_att = torch.cat([emb_ent[tail_idxs], emb_ent[head_idxs], \
                                    emb_rels], dim = -1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat_att).view(-1, self.num_head, self.dim_hid_ent)) * 
                       self.attn_vec).sum(dim = -1, keepdim = True)

        scatter_idx = tail_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)

        attn_val_max = torch.zeros((num_ent, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[tail_idxs])
        
        attn_sums = torch.zeros((num_ent, self.num_head, 1)).cuda().index_add(dim = 0, index = tail_idxs, source = attn_val)

        beta = attn_val / (attn_sums[tail_idxs]+1e-16)

        concat_mat = torch.cat([emb_ent[head_idxs], emb_rels], dim = -1)

        aggr_val = beta * self.aggr_proj(concat_mat).view(-1, self.num_head, self.dim_hid_ent)
        
        output = torch.zeros((num_ent, self.num_head, self.dim_hid_ent)).cuda().index_add(dim = 0, index = tail_idxs, source = aggr_val)

        return output.flatten(1,-1)

class InGramRelationLayer(nn.Module):
    def __init__(self, dim_in_rel, dim_out_rel, num_bin, bias = True, num_head = 8):
        super(InGramRelationLayer, self).__init__()

        self.dim_out_rel = dim_out_rel
        self.dim_hid_rel = dim_out_rel // num_head
        assert dim_out_rel == self.dim_hid_rel * num_head

        self.attn_proj = nn.Linear(2*dim_in_rel, dim_out_rel, bias = bias)
        self.attn_bin = nn.Parameter(torch.zeros(num_bin, num_head, 1))
        self.attn_vec = nn.Parameter(torch.zeros(1, num_head, self.dim_hid_rel))
        self.aggr_proj = nn.Linear(dim_in_rel, dim_out_rel, bias = bias)
        self.num_head = num_head

        self.act = nn.LeakyReLU(negative_slope = 0.2)
        self.num_bin = num_bin
        self.bias = bias

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.attn_proj.weight, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.attn_vec, gain = nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.aggr_proj.weight, gain = nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.zeros_(self.attn_proj.bias)
            nn.init.zeros_(self.aggr_proj.bias)
    
    def forward(self, emb_rel, relation_triplets):
        num_rel = len(emb_rel)
        
        head_idxs = relation_triplets[..., 0]
        tail_idxs = relation_triplets[..., 1]
        concat_mat = torch.cat([emb_rel[head_idxs], emb_rel[tail_idxs]], dim = -1)

        attn_val_raw = (self.act(self.attn_proj(concat_mat).view(-1, self.num_head, self.dim_hid_rel)) * \
                        self.attn_vec).sum(dim = -1, keepdim = True) + self.attn_bin[relation_triplets[...,2]]

        scatter_idx = head_idxs.unsqueeze(dim = -1).repeat(1, self.num_head).unsqueeze(dim = -1)

        attn_val_max = torch.zeros((num_rel, self.num_head, 1)).cuda().scatter_reduce(dim = 0, \
                                                                    index = scatter_idx, \
                                                                    src = attn_val_raw, reduce = 'amax', \
                                                                    include_self = False)
        attn_val = torch.exp(attn_val_raw - attn_val_max[head_idxs])

        attn_sums = torch.zeros((num_rel, self.num_head, 1)).cuda().index_add(dim = 0, index = head_idxs, source = attn_val)

        beta = attn_val / (attn_sums[head_idxs]+1e-16)
        
        output = torch.zeros((num_rel, self.num_head, self.dim_hid_rel)).cuda().index_add(dim = 0, \
                                                                                            index = head_idxs, 
                                                                                            source = beta * self.aggr_proj(emb_rel[tail_idxs]).view(-1, self.num_head, self.dim_hid_rel))

        return output.flatten(1,-1)

