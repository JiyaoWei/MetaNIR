import torch.nn as nn
import torch
import dgl
from ext_gnn import ExtGNN

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.dim = args.dim

        # 初始化的默认角色、实体嵌入
        self.rel_feat = nn.Parameter(torch.Tensor(args.num_rel, args.rel_dim)).to(self.args.gpu)
        nn.init.xavier_uniform_(self.rel_feat)
        # nn.init.xavier_uniform_(self.rel_feat, gain=nn.init.calculate_gain('relu'))
        self.ent_feat = nn.Parameter(torch.Tensor(args.num_ent, self.args.ent_dim)).to(self.args.gpu)
        nn.init.xavier_uniform_(self.ent_feat)
        # nn.init.xavier_uniform_(self.ent_feat, gain=nn.init.calculate_gain('relu'))
        # 元关系的嵌入
        self.pattern_rel_feat = nn.Parameter(torch.Tensor(16, args.num_rel_bases)).to(self.args.gpu)
        nn.init.xavier_uniform_(self.pattern_rel_feat)
        # nn.init.xavier_uniform_(self.pattern_rel_feat, gain=nn.init.calculate_gain('relu'))
        # 元关系转化为关系的投影矩阵
        self.base2rel_feat = nn.Parameter(torch.Tensor(args.num_rel_bases, self.args.rel_dim)).to(self.args.gpu)
        nn.init.xavier_uniform_(self.base2rel_feat)
        # nn.init.xavier_uniform_(self.base2rel_feat, gain=nn.init.calculate_gain('relu'))
        
        # 关系转化为实体的投影矩阵
        self.rel2ent_feat = nn.Parameter(torch.Tensor(args.rel_dim, self.args.ent_dim)).to(self.args.gpu)
        nn.init.xavier_uniform_(self.rel2ent_feat)
        # nn.init.xavier_uniform_(self.rel2ent_feat, gain=nn.init.calculate_gain('relu'))

        self.ext_gnn = ExtGNN(args)
        
        self.act = nn.GELU()
    

        # if args.scorer_func in ['maker', 'ingram', 'pmpi']:
        #     self.rel_matric = nn.Parameter(torch.Tensor(args.rel_dim, args.rel_dim)).to(self.args.gpu)
        #     self.ent_matric = nn.Parameter(torch.Tensor(args.ent_dim, args.ent_dim)).to(self.args.gpu)
        #     nn.init.xavier_uniform_(self.rel_matric)
        #     nn.init.xavier_uniform_(self.ent_matric)
            
        #     data = pickle.load(open(args.data_path, 'rb'))
        #     self.fact2ents = data['fact2ents']
        #     self.fact2rels = data['fact2rels']
        #     for fact_id, ents in self.fact2ents.items():
        #         self.ent_feat[fact_id] = torch.mean(self.ent_feat[ents]) + torch.mean(self.ent_feat[rels])
        
        
    # relation feature representation
    def init_rel(self, pattern_g, train_rels):
        with pattern_g.local_scope():
            etypes = pattern_g.edata['type']
            # 16 * 4, 4 * 128
            # pattern_rel_feat = torch.matmul(self.pattern_rel_feat, self.base2rel_feat) # self.base2rel_feat[self.pattern_rel_feat]
            pattern_g.edata['edge_h'] = self.pattern_rel_feat[etypes]

            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            pattern_g.update_all(message_func, reduce_func)
            pattern_g.edata.pop('edge_h')
            
            rel_coef = pattern_g.ndata['h']
            rel_coef[train_rels] = self.rel_feat[train_rels]
        return self.act(rel_coef)

    # entity feature representation
    def init_ent(self, g, init_rel_feat, train_ents):
        with g.local_scope():
            etypes = g.edata['type']
            # n*128, 128*128
            init_ent_feat = torch.matmul(init_rel_feat, self.rel2ent_feat)
            g.edata['edge_h'] = init_ent_feat[etypes]
            
            reverse_g = dgl.reverse(g, copy_edata=True)
            message_func = dgl.function.copy_e('edge_h', 'msg')
            reduce_func = dgl.function.mean('msg', 'h')
            # 在反转图上执行消息传递，将边信息聚合到头实体
            reverse_g.update_all(message_func, reduce_func)            
            reverse_g.edata.pop('edge_h')
            ent_feat = reverse_g.ndata['h'][:self.args.num_ent]
            
            ent_feat[train_ents] = self.ent_feat[train_ents]
        return self.act(ent_feat)

    def forward(self, ent_g, rel_g, train_ents, train_rels):
        if self.args.initrel == "True":
            init_rel_feat = self.init_rel(rel_g, train_rels)
        else:
            init_rel_feat = self.rel_feat
        if self.args.initent == "True":
            init_ent_feat = self.init_ent(ent_g, init_rel_feat, train_ents)
        else:
            init_ent_feat = self.ent_feat
        
        ent_emb, rel_emb = self.ext_gnn(ent_g, rel_g, self.base2rel_feat, ent_feat=init_ent_feat, rel_feat=init_rel_feat, mrel_feat=self.pattern_rel_feat)
        return ent_emb, rel_emb