import torch
import pickle
import numpy as np
from collections import defaultdict as ddict
import lmdb
from utils import deserialize
from subgraph import get_rel_metagraph
import random
from copy import deepcopy as dcopy
import time


class Data(object):
    def __init__(self, args, data):
        self.args = args

        self.entity_dict = data['ent2id']
        self.relation_dict = data['rel2id']

        self.num_ent = len(self.entity_dict)
        self.num_rel = len(self.relation_dict)

    def get_train_g(self, sup_tri, num_ent):
        
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
            
        num_fact = len(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_pattern_g(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_ques2ans(self, facts):
        # pad_facts = [fact[:16] + (16-len(fact))*[0] for fact in facts]
        ques2ans = ddict(list)
        for fact in facts:
            for i in range(len(fact)):
            # for i in range(16):
                tmp_fact = dcopy(fact)
                tmp_fact[i] = -1
                cur_questr = tuple(tmp_fact)
                ques2ans[cur_questr].append(fact[i])
        return ques2ans


class ValidData(Data):
    def __init__(self, args, data):
        super(ValidData, self).__init__(args, data)
        self.triples = data['train']['triples']
        self.sup_facts = data['valid']['support']
        self.que_facts = data['valid']['query']
        self.que_uent_facts = data['valid']['query_uent']
        self.que_urel_facts = data['valid']['query_urel']
        self.que_uboth_facts = data['valid']['query_uboth']

        self.train_ents = torch.LongTensor(list(data['train_ents']))
        self.train_rels = torch.LongTensor(list(data['train_rels']))
        
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        if args.dataset == 'FJF17K':
            self.rel2candidates = data['rel2candidates']

        self.ques2ans = self.get_ques2ans(self.sup_facts + self.que_facts)

        # g and pattern g
        self.g = self.get_train_g(self.sup_facts, len(data['ent2id'])).to(self.args.gpu)

        self.pattern_g = self.get_pattern_g(self.sup_facts, len(data['rel2id'])).to(self.args.gpu)


class TestData(Data):
    def __init__(self, args, data):
        super(TestData, self).__init__(args, data)
        self.triples = data['train']['triples']
        self.sup_facts = data['test']['support']
        self.que_facts = data['test']['query']
        self.que_uent_facts = data['test']['query_uent']
        self.que_urel_facts = data['test']['query_urel']
        self.que_uboth_facts = data['test']['query_uboth']

        self.train_ents = torch.LongTensor(list(data['train_ents']))
        self.train_rels = torch.LongTensor(list(data['train_rels']))
        
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        
        if args.dataset == 'FJF17K':
            self.rel2candidates = data['rel2candidates']
        
        self.ques2ans = self.get_ques2ans(self.sup_facts + self.que_facts)

        # g and pattern g
        self.g = self.get_train_g(self.sup_facts, len(data['ent2id'])).to(self.args.gpu)

        self.pattern_g = self.get_pattern_g(self.sup_facts, len(data['rel2id'])).to(self.args.gpu)
            
# metalearning
class TrainSubgraphDataset(Data):
    def __init__(self, args):
        self.args = args
        data = pickle.load(open(args.data_path, 'rb'))
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        self.train_ents = data['train_ents']
        self.train_rels = data['train_rels']
        self.ents_num = len(data['ent2id'])
        self.rels_num = len(data['rel2id'])
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=1, lock=False)
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())
        self.query2answer = None

        edge_labels = []
        max_aux = 8 - 2
        edge_labels.append([0, 1, 2, 3] + [4, 5] * max_aux )
        edge_labels.append([1, 0, 6, 7] + [8, 9] * max_aux )
        edge_labels.append([2, 6, 0, 10] + [11, 12] * max_aux )
        edge_labels.append([3, 7, 10, 0] + [13, 14] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([4,8,11,13] + [15,16] * idx + [0,17] + [15,16] * (max_aux - idx - 1))
            edge_labels.append([5,9,12,14] + [16,18] * idx + [17,0] + [16,18] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")

        self.train_data = data['train']['triples']
        if self.args.train_background == 'all':
            # g and pattern g
            self.g = self.get_train_g_fact(self.train_data, len(data['ent2id']))
            self.pattern_g = self.get_pattern_g_fact(self.train_data, len(data['rel2id']))
        self.j = 0
        self.i = 0
        
        self.train_data = data['train']['triples']
        self.ques2ans = self.get_fact2ans(self.train_data)
        
    def get_fact2ans(self, facts):
        pad_facts = [fact[:16] + (16-len(fact))*[0] for fact in facts]
        ques2ans = ddict(list)
        for fact in pad_facts:
            for i in range(len(fact)):
            # for i in range(16):
                tmp_fact = dcopy(fact)
                tmp_fact[i] = 1
                cur_questr = tuple(tmp_fact)
                ques2ans[cur_questr].append(fact[i])
        return ques2ans
        
    def reset(self):
        # self.tmp_sub_range = random.sample(range(0, self.args.num_train_subgraph), self.args.train_bs)
        return self

    def __len__(self):
        return self.args.num_train_subgraph

    @staticmethod
    def collate_fn(data):
        return data

    def get_train_g(self, edge_index, edge_type, num_ent, num_fact):
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g
    
    def get_pattern_g(self, edge_index, edge_type, num_rel):
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g
    
    def get_train_g_fact(self, sup_tri, num_ent):
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
        num_fact = len(sup_tri)
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g


    def get_pattern_g_fact(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g

    def __iter__(self):
        return self

    def __next__(self):
        if self.j % self.args.train_bs == 0 and self.j != 0:
            self.i = self.j + self.i
            self.j = 0
            raise StopIteration
        if self.i + self.j >= self.args.num_train_subgraph:
            self.i = 0
            self.j = 0

        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(self.i + self.j).encode('ascii')
            sup_facts, que_facts, edge_index_e, edge_type_e, edge_index_r, edge_type_r, ques2ans = deserialize(txn.get(str_id))

        # 随机mask一部分，用于模拟unseen entity/relation
        ent_mask = np.random.choice(np.arange(self.ents_num), int(self.ents_num * random.randint(3, 8) * self.args.task_mask_rate), replace=False)
        rel_mask = np.random.choice(np.arange(self.rels_num), int(self.rels_num * random.randint(3, 8) * self.args.task_mask_rate), replace=False)

        tmp_seen_ents = torch.LongTensor(list(set(self.train_ents) - set(ent_mask)))
        tmp_seen_rels = torch.LongTensor(list(set(self.train_rels) - set(rel_mask)))
        
        if self.args.train_background == 'all':
            g = self.g
            pattern_g = self.pattern_g
        else:
            g = self.get_train_g(edge_index_e, edge_type_e, self.ents_num, len(sup_facts))
            pattern_g = self.get_pattern_g(edge_index_r, edge_type_r, self.rels_num)

        pred_facts = []
        pred_indexs = []
        mask_outputs = []
        mask_inputs = []
        query_types = []
        mask_labels = []

        for que_fact in que_facts:
            que_fact = que_fact[:16]
            fact_length = len(que_fact)
            rand_indx = int(random.randint(0, fact_length-1))
            pred_fact = []
            for _, ele in enumerate(que_fact):
                if _ != rand_indx:
                    pred_fact.append(ele)
                else:
                    # mask 1；padding 0
                    pred_fact.append(1)
                    # mask_labels不需要padding吗？
                    if rand_indx % 2 == 0:
                        mask_labels.append(ele)
                    else:
                        mask_labels.append(ele+self.rels_num)
            mask_input = [1] * len(que_fact) + [0] * (16-len(que_fact))
            pred_fact += [0] * (16-len(que_fact))
        
            mask_output = np.zeros(self.rels_num+self.ents_num).astype("bool")
            if rand_indx % 2 == 0:
                query_type = -1
                mask_output[:self.rels_num] = True
            else:
                query_type = 1
                mask_output[self.rels_num:] = True
            
            pred_facts.append(pred_fact)
            pred_indexs.append(rand_indx)
            mask_outputs.append(mask_output)
            mask_input = np.array(mask_input).astype("int64")
            mask_input = np.outer(mask_input, mask_input).astype("bool")
            mask_inputs.append(mask_input)
            query_types.append(query_type)
                
        pred_facts = torch.LongTensor(pred_facts)
        pred_indexs = torch.LongTensor(pred_indexs)
        mask_outputs = torch.LongTensor(mask_outputs)
        mask_inputs = torch.LongTensor(mask_inputs)
        query_types = torch.LongTensor(query_types)
        mask_labels = torch.LongTensor(mask_labels)
        
        self.j += 1
        edge_labels = torch.LongTensor(self.edge_labels)
        
        if self.args.adjustment == 'True':
            
            if self.args.adjustment_reduce == 'True':
                que_ents, que_rels = set(), set()
                for fact in que_facts:
                    que_ents = que_ents | set(fact[1::2])
                    que_rels = que_rels | set(fact[::2])
                
                adjust_facts = []
                for fact in sup_facts:
                    unseen_ent = set(fact[1::2]) & set(ent_mask) & que_ents
                    unseen_rel = set(fact[0::2]) & set(rel_mask) & que_rels
                    if len(unseen_ent) !=0 or len(unseen_rel) != 0:
                        adjust_facts.append(fact)
                        que_rels = que_rels - unseen_rel
                        que_ents = que_ents - unseen_ent
            else:
                adjust_facts = sup_facts
            
            sup_pred_facts = []
            sup_pred_indexs = []
            sup_mask_outputs = []
            sup_mask_inputs = []
            sup_query_types = []
            sup_mask_labels = []

            for sup_fact in adjust_facts:
                sup_fact = sup_fact[:16]
                mask_input = [1] * len(sup_fact) + [0] * (16-len(sup_fact))
                fact_length = len(sup_fact)
                rand_indx = int(random.randint(0, fact_length-1))
                pred_fact = []
                for _, ele in enumerate(sup_fact):
                    if _ != rand_indx:
                        pred_fact.append(ele)
                    else:
                        # mask 1；padding 0
                        pred_fact.append(1)
                pred_fact += [0] * (16-len(sup_fact))
                
                if self.args.pred_truth == 'True':
                    ans = self.ques2ans[tuple(pred_fact)]
                    for ele in ans:
                        if rand_indx % 2 == 0:
                            sup_mask_labels.append(ele)
                        else:
                            sup_mask_labels.append(ele+self.rels_num)
                else:
                    if rand_indx % 2 == 0:
                        sup_mask_labels.append(sup_fact[rand_indx])
                    else:
                        sup_mask_labels.append(sup_fact[rand_indx]+self.rels_num)
                    
            
                mask_output = np.zeros(self.rels_num+self.ents_num).astype("bool")
                if rand_indx % 2 == 0:
                    query_type = -1
                    mask_output[:self.rels_num] = True
                else:
                    query_type = 1
                    mask_output[self.rels_num:] = True
                
                sup_pred_facts.append(pred_fact)
                sup_pred_indexs.append(rand_indx)
                sup_mask_outputs.append(mask_output)
                sup_mask_input = np.array(mask_input).astype("int64")
                sup_mask_input = np.outer(mask_input, mask_input).astype("bool")
                sup_mask_inputs.append(sup_mask_input)
                sup_query_types.append(query_type)
                    
            sup_pred_facts = torch.LongTensor(sup_pred_facts)
            sup_pred_indexs = torch.LongTensor(sup_pred_indexs)
            sup_mask_outputs = torch.LongTensor(sup_mask_outputs)
            sup_mask_inputs = torch.LongTensor(sup_mask_inputs)
            sup_query_types = torch.LongTensor(sup_query_types)
            if self.args.pred_truth != 'True':
                sup_mask_labels = torch.LongTensor(sup_mask_labels)
            
            return g, pattern_g, pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels, tmp_seen_ents, tmp_seen_rels, sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels
        else:
            return g, pattern_g, pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels, tmp_seen_ents, tmp_seen_rels

# metalearning
class TrainBinarySubgraphDataset(Data):
    def __init__(self, args):
        self.args = args
        data = pickle.load(open(args.data_path, 'rb'))
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        self.train_ents = data['train_ents']
        self.train_rels = data['train_rels']
        self.ents_num = len(data['ent2id'])
        self.rels_num = len(data['rel2id'])
        self.env = lmdb.open(args.db_path, readonly=True, max_dbs=1, lock=False)
        self.subgraphs_db = self.env.open_db("train_subgraphs".encode())
        self.query2answer = None

        self.train_data = data['train']['triples']

        self.train_data = data['train']['triples']
        self.ques2ans = self.get_fact2ans(self.train_data)
        self.i, self.j, self.p = 0, 0, 0
        
        
    def get_fact2ans(self, facts):
        pad_facts = [fact[:16] + (16-len(fact))*[0] for fact in facts]
        ques2ans = ddict(list)
        for fact in pad_facts:
            for i in range(len(fact)):
            # for i in range(16):
                tmp_fact = dcopy(fact)
                tmp_fact[i] = 1
                cur_questr = tuple(tmp_fact)
                ques2ans[cur_questr].append(fact[i])
        return ques2ans
        
    def reset(self):
        self.i, self.j, self.p, self.c = 0, 0, 0, 0
        with self.env.begin(db=self.subgraphs_db) as txn:
            str_id = '{:08}'.format(self.p+self.c).encode('ascii')
            self.sup_facts, self.que_facts, edge_index_e, edge_type_e, edge_index_r, edge_type_r, ques2ans = deserialize(txn.get(str_id))
            self.facts, self.facts_ary = self.build_facts(self.que_facts)
        self.ary = self.facts_ary[self.i]    
    
        self.g = self.get_train_g(edge_index_e, edge_type_e, self.ents_num, len(self.sup_facts))
        self.pattern_g = self.get_pattern_g(edge_index_r, edge_type_r, self.rels_num)
        return self

    def __len__(self):
        return self.args.num_train_subgraph

    @staticmethod
    def collate_fn(data):
        return data

    def get_train_g(self, edge_index, edge_type, num_ent, num_fact):
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g
    
    def get_pattern_g(self, edge_index, edge_type, num_rel):
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        return g
    
    def __iter__(self):
        return self

    def build_facts(self, que_facts):
        '''
        build training data for each snapshots
        :return: training data
        '''
        fact_ary = []
        fact_list = []
        que_facts_ = ddict(list)
        for fact_ in que_facts:
            que_facts_[len(fact_)].append(fact_)
        for length, facts_ in que_facts_.items():
            fact_list.append(facts_)
        for length in range(len(fact_list)):
            fact_ary.append(length)
        return fact_list, fact_ary

    def __next__(self):
        # self.j 第几个事件；self.i 第几个ary
        if self.j >= len(self.facts[self.ary]):
            if self.i < len(self.facts_ary) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.facts_ary[self.i]
            elif self.i == len(self.facts_ary) - 1:
                if self.p + 1 >= self.args.train_bs:#self.args.num_train_subgraph:
                    self.c = self.p + 1
                    self.p = 0
                    raise StopIteration
                else:
                    self.p += 1
                    with self.env.begin(db=self.subgraphs_db) as txn:
                        str_id = '{:08}'.format(self.p + self.c).encode('ascii')
                        self.sup_facts, self.que_facts, edge_index_e, edge_type_e, edge_index_r, edge_type_r, ques2ans = deserialize(txn.get(str_id))
                        self.facts, self.facts_ary = self.build_facts(self.que_facts)
                    self.i, self.j = 0, 0
                    self.ary = self.facts_ary[self.i]
            
                    self.g = self.get_train_g(edge_index_e, edge_type_e, self.ents_num, len(self.sup_facts))
                    self.pattern_g = self.get_pattern_g(edge_index_r, edge_type_r, self.rels_num)

        # 随机mask一部分，用于模拟unseen entity/relation
        ent_mask = np.random.choice(np.arange(self.ents_num), int(self.ents_num * random.randint(3, 8) * self.args.task_mask_rate), replace=False)
        rel_mask = np.random.choice(np.arange(self.rels_num), int(self.rels_num * random.randint(3, 8) * self.args.task_mask_rate), replace=False)

        tmp_seen_ents = torch.LongTensor(list(set(self.train_ents) - set(ent_mask)))
        tmp_seen_rels = torch.LongTensor(list(set(self.train_rels) - set(rel_mask)))

        que_facts_ = ddict(list)
        for cur_fact in self.que_facts:
            que_facts_[len(cur_fact)].append(cur_fact)

        _fact = torch.LongTensor([])
        _label = torch.LongTensor([])
        for cur_fact in self.facts[self.ary][self.j: self.j+self.args.train_bs]:
            fact, label = self.corrupt(cur_fact)
            _fact = torch.cat([_fact, fact], dim=0)
            _label = torch.cat([_label, label], dim=0)

        self.j = min(self.j+self.args.train_bs, len(self.facts[self.ary]))
        return self.g, self.pattern_g, tmp_seen_ents, tmp_seen_rels, [_fact[:,_] for _ in range(len(_fact[0]))], _label

    def corrupt(self, fact):
        '''
        :param fact: positive facts
        :return: positive facts & negative facts ; pos/neg labels.
        '''
        facts = np.array([fact]).repeat(self.args.neg_ratio+1, axis=0)
        # todu
        if self.args.scorer_func in ['maker']:
            label = [1] + [-1 for _ in range(self.args.neg_ratio)]
        rand_prob = np.random.randint(0, len(fact), self.args.neg_ratio)
        neg_e = np.random.randint(0, len(self.train_ents) - 1, self.args.neg_ratio)
        neg_r = np.random.randint(0, len(self.train_rels) - 1, self.args.neg_ratio)
        for _ in range(self.args.neg_ratio):
            if self.args.scorer_func in ['maker']:
                if rand_prob[_]%2 == 1:
                    facts[_+1][rand_prob[_]] = neg_e[_]
                else:
                    facts[_+1][rand_prob[_]] = neg_r[_]

        facts = torch.LongTensor(facts)
        label = torch.LongTensor(label)
        return facts, label


class FewTestDataset():
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    # total, urel, uent, uboth
    # total, ent, rel
    def __init__(self, args, data, type = 'total', pred_type='ent'):
        self.args = args
        if type == 'total':
            self.facts = data.que_facts
        elif type == 'urel':
            self.facts = data.que_urel_facts
        elif type == 'uent':
            self.facts = data.que_uent_facts
        elif type == 'uboth':
            self.facts = data.que_uboth_facts

        self.num_ent = len(data.ent2id)
        self.num_rel = len(data.rel2id)
        self.ques2ans = data.ques2ans
        self.pred_type = pred_type

        self.train_bs = args.eval_bs
        self.valid, self.valid_ary = self.build_facts()

        edge_labels = []
        max_aux = 8 - 2
        edge_labels.append([0, 1, 2, 3] + [4, 5] * max_aux )
        edge_labels.append([1, 0, 6, 7] + [8, 9] * max_aux )
        edge_labels.append([2, 6, 0, 10] + [11, 12] * max_aux )
        edge_labels.append([3, 7, 10, 0] + [13, 14] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([4,8,11,13] + [15,16] * idx + [0,17] + [15,16] * (max_aux - idx - 1))
            edge_labels.append([5,9,12,14] + [16,18] * idx + [17,0] + [16,18] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = ddict(list)
        for fact in self.facts:
            fact = fact[:16]
            valid[len(fact)].append(fact)
        valid_ary = list(valid.keys())
        return valid, valid_ary

    def reset(self):
        # i-th ary; j-th fact; k-th index
        self.eval_data = self.valid
        self.eval_ary = self.valid_ary
            
        self.i = 0
        self.ary = self.eval_ary[self.i] if len(self.eval_ary) != 0 else None
        self.j = 0
        self.k = 3
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.eval_data) == 0:
            raise StopIteration
        # i: ary, j: fact, k: index
        if self.j >= len(self.eval_data[self.ary]):
            if self.i < len(self.eval_data) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.eval_ary[self.i]
            elif self.i == len(self.eval_data) - 1:
                raise StopIteration
        _fact = self.eval_data[self.ary][self.j: self.j+self.train_bs]

        _label = torch.LongTensor([])
        mask_outputs, mask_inputs = [], []
        _fact_pred = []
        for cur_fact in _fact:
            cur_fact = cur_fact[:16]
            _fact_pred.append(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:] + [1] * (16-len(cur_fact)))
            
            mask_input = [1] * len(cur_fact) + [0] * (16-len(cur_fact))
            mask_input = np.array(mask_input).astype("int64")
            mask_input = np.outer(mask_input, mask_input).astype("bool")
            mask_inputs.append(mask_input)
            
            mask_output = np.zeros(self.num_rel+self.num_ent).astype("bool")
            if self.k % 2 == 0:
                mask_output[:self.num_rel] = True
            else:
                mask_output[self.num_rel:] = True
                
            label = self.ques2ans[tuple(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:])]
            _label = torch.cat((_label, self.get_label(label).unsqueeze(0)), 0)
            
            mask_outputs.append(mask_output)
            
        _fact_pred = np.array(_fact_pred)
        _fact = np.array(_fact)
        
        edge_labels = torch.LongTensor(self.edge_labels)
        mask_inputs = torch.LongTensor(mask_inputs)
        mask_outputs = torch.LongTensor(mask_outputs)

        self.j = min(self.j+self.train_bs, len(self.eval_data[self.ary]))
        return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], len(_fact[0])-1, _label, _fact[:,3], mask_inputs, mask_outputs, edge_labels
    
    def get_label(self, label):
        if self.k%2==1:
            y = np.zeros([self.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.num_rel], dtype=np.float32)

        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)


class InTestDataset():
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    # total, urel, uent, uboth
    # total, ent, rel
    def __init__(self, args, data, type = 'total', pred_type = 'ent'):
        self.args = args
        if type == 'total':
            self.facts = data.que_facts
            self.facts = data.triples[:2000]
        elif type == 'urel':
            self.facts = data.que_urel_facts
        elif type == 'uent':
            self.facts = data.que_uent_facts
        elif type == 'uboth':
            self.facts = data.que_uboth_facts

        self.num_ent = len(data.ent2id)
        self.num_rel = len(data.rel2id)
        self.ques2ans = data.ques2ans
        self.pred_type = pred_type

        self.eval_bs = args.eval_bs
        self.valid, self.valid_ary = self.build_facts()

        edge_labels = []
        max_aux = 8 - 2
        edge_labels.append([0, 1, 2, 3] + [4, 5] * max_aux )
        edge_labels.append([1, 0, 6, 7] + [8, 9] * max_aux )
        edge_labels.append([2, 6, 0, 10] + [11, 12] * max_aux )
        edge_labels.append([3, 7, 10, 0] + [13, 14] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([4,8,11,13] + [15,16] * idx + [0,17] + [15,16] * (max_aux - idx - 1))
            edge_labels.append([5,9,12,14] + [16,18] * idx + [17,0] + [16,18] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")


    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = ddict(list)
        for fact in self.facts:
            fact = fact[:16]
            valid[len(fact)].append(fact)
        valid_ary = list(valid.keys())
        return valid, valid_ary

    def reset(self):
        # i-th ary; j-th fact; k-th index
        self.eval_data = self.valid
        self.eval_ary = self.valid_ary
            
        self.i = 0
        self.ary = self.eval_ary[self.i] if len(self.eval_ary) != 0 else None
        self.j = 0
        self.k = 1
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.eval_data) == 0:
            raise StopIteration
        # i: ary, j: fact, k: index
        if self.j >= len(self.eval_data[self.ary]):
            if self.i < len(self.eval_data) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.eval_ary[self.i]
                self.k = 1
            elif self.i == len(self.eval_data) - 1:
                raise StopIteration
        _fact = self.eval_data[self.ary][self.j: self.j+self.eval_bs]
        
        _label = torch.LongTensor([])
        mask_outputs, mask_inputs = [], []
        _fact_pred = []
        for cur_fact in _fact:
            cur_fact = cur_fact[:16]
            _fact_pred.append(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:] + [1] * (16-len(cur_fact)))
            
            mask_input = [1] * len(cur_fact) + [0] * (16-len(cur_fact))
            mask_input = np.array(mask_input).astype("int64")
            mask_input = np.outer(mask_input, mask_input).astype("bool")
            mask_inputs.append(mask_input)
            
            mask_output = np.zeros(self.num_rel+self.num_ent).astype("bool")
            if self.k % 2 == 0:
                mask_output[:self.num_rel] = True
            else:
                mask_output[self.num_rel:] = True
                
            label = self.ques2ans[tuple(list(cur_fact)[:self.k]+[-1]+list(cur_fact)[self.k+1:])]
            _label = torch.cat((_label, self.get_label(label).unsqueeze(0)), 0)
            
            mask_outputs.append(mask_output)
            
        _fact_pred = np.array(_fact_pred)
        _fact = np.array(_fact)
        
        edge_labels = torch.LongTensor(self.edge_labels)
        mask_inputs = torch.LongTensor(mask_inputs)
        mask_outputs = torch.LongTensor(mask_outputs)

        if self.k == 1:
            self.k = 3
            return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], len(_fact[0])-1, _label, _fact[:,1], mask_inputs, mask_outputs, edge_labels
        else:
            self.k = 1
            self.j = min(self.j+self.eval_bs, len(self.eval_data[self.ary]))
            return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], len(_fact[0])-1, _label, _fact[:,3], mask_inputs, mask_outputs, edge_labels
    
    def get_label(self, label):
        if self.k%2==1:
            y = np.zeros([self.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.num_rel], dtype=np.float32)

        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TestDataset():
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    # total, urel, uent, uboth
    # total, ent, rel
    def __init__(self, args, data, type = 'total', pred_type = 'ent'):
        self.args = args
        if type == 'total':
            self.facts = data.que_facts
            self.sup_facts = data.sup_facts
        elif type == 'urel':
            self.facts = data.que_urel_facts
            self.sup_facts = data.sup_facts
        elif type == 'uent':
            self.facts = data.que_uent_facts
            self.sup_facts = data.sup_facts
        elif type == 'uboth':
            self.facts = data.que_uboth_facts
            self.sup_facts = data.sup_facts

        self.train_ents = data.train_ents
        self.train_rels = data.train_rels
        self.num_ent = len(data.ent2id)
        self.num_rel = len(data.rel2id)
        self.ques2ans = data.ques2ans
        self.pred_type = pred_type

        self.train_bs = args.eval_bs
        self.valid, self.valid_ary = self.build_facts()

        edge_labels = []
        max_aux = 8 - 2
        edge_labels.append([0, 1, 2, 3] + [4, 5] * max_aux )
        edge_labels.append([1, 0, 6, 7] + [8, 9] * max_aux )
        edge_labels.append([2, 6, 0, 10] + [11, 12] * max_aux )
        edge_labels.append([3, 7, 10, 0] + [13, 14] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([4,8,11,13] + [15,16] * idx + [0,17] + [15,16] * (max_aux - idx - 1))
            edge_labels.append([5,9,12,14] + [16,18] * idx + [17,0] + [16,18] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid = ddict(list)
        for fact in self.facts:
            fact = fact[:16]
            valid[len(fact)].append(fact)
        valid_ary = list(valid.keys())
        return valid, valid_ary

    def reset(self):
        # i-th ary; j-th fact; k-th index
        self.eval_data = self.valid
        self.eval_ary = self.valid_ary
            
        self.i = 0
        self.ary = self.eval_ary[self.i] if len(self.eval_ary) != 0 else None
        self.j = 0
        self.k = 0
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.eval_data) == 0:
            raise StopIteration
        # i: ary, j: fact, k: index
        if self.j >= len(self.eval_data[self.ary]):
            if self.i < len(self.eval_data) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.eval_ary[self.i]
                self.k = 0
            elif self.i == len(self.eval_data) - 1:
                raise StopIteration
        _fact = self.eval_data[self.ary][self.j: self.j+self.train_bs]

        _label = torch.LongTensor([])
        mask_outputs, mask_inputs = [], []
        _fact_pred = []
        for cur_fact in _fact:
            cur_fact = cur_fact[:16]
            _fact_pred.append(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:] + [1] * (16-len(cur_fact)))
            
            mask_input = [1] * len(cur_fact) + [0] * (16-len(cur_fact))
            mask_input = np.array(mask_input).astype("int64")
            mask_input = np.outer(mask_input, mask_input).astype("bool")
            mask_inputs.append(mask_input)
            
            mask_output = np.zeros(self.num_rel+self.num_ent).astype("bool")
            if self.k % 2 == 0:
                mask_output[:self.num_rel] = True
            else:
                mask_output[self.num_rel:] = True
                
            label = self.ques2ans[tuple(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:])]
            _label = torch.cat((_label, self.get_label(label).unsqueeze(0)), 0)
            
            mask_outputs.append(mask_output)
            
        _fact_pred = np.array(_fact_pred)
        _fact = np.array(_fact)
        
        edge_labels = torch.LongTensor(self.edge_labels)
        mask_inputs = torch.LongTensor(mask_inputs)
        mask_outputs = torch.LongTensor(mask_outputs)
        
        # sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels = [], [], [], [], [], []

        if self.args.adjustment == 'True':
            if self.args.adjustment_reduce == 'True':
                que_ents, que_rels = set(), set()
                for fact in _fact:
                    que_ents = que_ents | set(fact[1::2])
                    que_rels = que_rels | set(fact[::2])
                
                adjust_facts = []
                for fact in self.sup_facts:
                    unseen_ent = set(fact[1::2]) & que_ents
                    unseen_rel = set(fact[0::2]) & que_rels
                    if len(unseen_ent) !=0 or len(unseen_rel) != 0:
                        adjust_facts.append(fact)
                        que_rels = que_rels - unseen_rel
                        que_ents = que_ents - unseen_ent
            else:
                adjust_facts = self.sup_facts
            
            
            sup_pred_facts = []
            sup_pred_indexs = []
            sup_mask_outputs = []
            sup_mask_inputs = []
            sup_query_types = []
            sup_mask_labels = []

            for sup_fact in adjust_facts:
                sup_fact = sup_fact[:16]
                mask_input = [1] * len(sup_fact) + [0] * (16-len(sup_fact))
                fact_length = len(sup_fact)
                rand_indx = int(random.randint(0, fact_length-1))
                pred_fact = []
                for _, ele in enumerate(sup_fact):
                    if _ != rand_indx:
                        pred_fact.append(ele)
                    else:
                        # mask 1；padding 0
                        pred_fact.append(1)
                pred_fact += [0] * (16-len(sup_fact))
                
                if self.args.pred_truth == 'True':
                    ans = self.ques2ans[tuple(pred_fact)]
                    for ele in ans:
                        if rand_indx % 2 == 0:
                            sup_mask_labels.append(ele)
                        else:
                            sup_mask_labels.append(ele+self.num_rel)
                else:
                    if rand_indx % 2 == 0:
                        sup_mask_labels.append(sup_fact[rand_indx])
                    else:
                        sup_mask_labels.append(sup_fact[rand_indx]+self.num_rel)
                    
            
                mask_output = np.zeros(self.num_rel+self.num_ent).astype("bool")
                if rand_indx % 2 == 0:
                    query_type = -1
                    mask_output[:self.num_rel] = True
                else:
                    query_type = 1
                    mask_output[self.num_rel:] = True
                
                sup_pred_facts.append(pred_fact)
                sup_pred_indexs.append(rand_indx)
                sup_mask_outputs.append(mask_output)
                sup_mask_input = np.array(mask_input).astype("int64")
                sup_mask_input = np.outer(mask_input, mask_input).astype("bool")
                sup_mask_inputs.append(sup_mask_input)
                sup_query_types.append(query_type)
                    
            sup_pred_facts = torch.LongTensor(sup_pred_facts)
            sup_pred_indexs = torch.LongTensor(sup_pred_indexs)
            sup_mask_outputs = torch.LongTensor(sup_mask_outputs)
            sup_mask_inputs = torch.LongTensor(sup_mask_inputs)
            sup_query_types = torch.LongTensor(sup_query_types)

            if self.args.pred_truth != 'True':
                sup_mask_labels = torch.LongTensor(sup_mask_labels)
            if self.k < len(_fact[0])-1:
                self.k += 1
                return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], self.k-1, _label, _fact[:,self.k-1], mask_inputs, mask_outputs, edge_labels, sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels
            else:
                self.k = 0
                self.j = min(self.j+self.train_bs, len(self.eval_data[self.ary]))
                return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], len(_fact[0])-1, _label, _fact[:,-1], mask_inputs, mask_outputs, edge_labels, sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels
        else:
            if self.k < len(_fact[0])-1:
                self.k += 1
                return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], self.k-1, _label, _fact[:,self.k-1], mask_inputs, mask_outputs, edge_labels
            else:
                self.k = 0
                self.j = min(self.j+self.train_bs, len(self.eval_data[self.ary]))
                return [_fact_pred[:,_] for _ in range(len(_fact_pred[0]))], len(_fact[0])-1, _label, _fact[:,-1], mask_inputs, mask_outputs, edge_labels


    def get_label(self, label):
        if self.k%2==1:
            y = np.zeros([self.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.num_rel], dtype=np.float32)

        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)


class TrainDataset(Data):
    def __init__(self, args):
        self.args = args
        data = pickle.load(open(args.data_path, 'rb'))
        self.train_data = data['train']['triples']
        self.rels_num = len(data['rel2id'])
        self.ents_num = len(data['ent2id'])
        self.train_ents = data['train_ents']
        self.train_rels = data['train_rels']
        # g and pattern g
        self.g = self.get_train_g(self.train_data, len(data['ent2id']))
        self.pattern_g = self.get_pattern_g(self.train_data, len(data['rel2id']))

        self.tmp_seen_ents = torch.LongTensor(list(set(self.train_ents)))
        self.tmp_seen_rels = torch.LongTensor(list(set(self.train_rels)))

        edge_labels = []
        max_aux = 8 - 2
        edge_labels.append([0, 1, 2, 3] + [4, 5] * max_aux )
        edge_labels.append([1, 0, 6, 7] + [8, 9] * max_aux )
        edge_labels.append([2, 6, 0, 10] + [11, 12] * max_aux )
        edge_labels.append([3, 7, 10, 0] + [13, 14] * max_aux )
        for idx in range(max_aux):
            edge_labels.append([4,8,11,13] + [15,16] * idx + [0,17] + [15,16] * (max_aux - idx - 1))
            edge_labels.append([5,9,12,14] + [16,18] * idx + [17,0] + [16,18] * (max_aux - idx - 1))
        self.edge_labels = np.asarray(edge_labels).astype("int64")
        self.i = 0


    def reset(self):
        self.i = 0
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= len(self.train_data):
            raise StopIteration
        _facts = self.train_data[self.i: self.i+self.args.train_bs]
        
        pred_facts = []
        pred_indexs = []
        mask_outputs = []
        mask_inputs = []
        query_types = []
        mask_labels = []

        for que_fact in _facts:
            que_fact = que_fact[:16]
            fact_length = len(que_fact)
            rand_indx = int(random.randint(0, fact_length-1))
            pred_fact = []
            for _, ele in enumerate(que_fact):
                if _ != rand_indx:
                    pred_fact.append(ele)
                else:
                    # mask 1；padding 0
                    pred_fact.append(1)
                    if rand_indx % 2 == 0:
                        mask_labels.append(ele)
                    else:
                        mask_labels.append(ele+self.rels_num)
            mask_input = [1] * len(que_fact) + [0] * (16-len(que_fact))
            pred_fact += [0] * (16-len(que_fact))
        
            mask_output = np.zeros(self.rels_num+self.ents_num).astype("bool")
            if rand_indx % 2 == 0:
                query_type = -1
                mask_output[:self.rels_num] = True
            else:
                query_type = 1
                mask_output[self.rels_num:] = True
            
            pred_facts.append(pred_fact)
            pred_indexs.append(rand_indx)
            mask_outputs.append(mask_output)
            mask_input = np.array(mask_input).astype("int64")
            mask_input = np.outer(mask_input, mask_input).astype("bool")
            mask_inputs.append(mask_input)
            query_types.append(query_type)
                
        pred_facts = torch.LongTensor(pred_facts)
        pred_indexs = torch.LongTensor(pred_indexs)
        mask_outputs = torch.LongTensor(mask_outputs)
        mask_inputs = torch.LongTensor(mask_inputs)
        query_types = torch.LongTensor(query_types)
        mask_labels = torch.LongTensor(mask_labels)
        
        edge_labels = torch.LongTensor(self.edge_labels)
        
        self.i = min(self.i+self.args.train_bs, len(self.train_data))
        return pred_facts, pred_indexs, mask_outputs, mask_inputs, edge_labels, query_types, mask_labels


class TrainDatasetMarginLoss:
    def __init__(self, args):
        self.args = args
        
        data = pickle.load(open(args.data_path, 'rb'))
        self.train_data = data['train']['triples']
        # g and pattern g
        self.g = self.get_train_g(self.train_data, len(data['ent2id']))
        self.pattern_g = self.get_pattern_g(self.train_data, len(data['rel2id']))
        self.train_data = [fact[1:16] for fact in self.train_data]
        
        self.rels_num = len(data['rel2id'])
        self.ents_num = len(data['ent2id'])
        self.train_ents = data['train_ents']
        self.train_rels = data['train_rels']

        self.tmp_seen_ents = torch.LongTensor(list(set(self.train_ents)))
        self.tmp_seen_rels = torch.LongTensor(list(set(self.train_rels)))
        
        
        self.facts, self.facts_ary = self.build_facts()

    def get_train_g(self, sup_tri, num_ent):
        
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
            
        num_fact = len(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_pattern_g(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        
        return g

    def reset(self):
        # i-th ary; j-th fact
        self.i = 0
        self.j = 0
        self.ary = self.facts_ary[self.i]
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.j >= len(self.facts[self.ary]):
            if self.i < len(self.facts_ary) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.facts_ary[self.i]
            elif self.i == len(self.facts_ary) - 1:
                raise StopIteration
        _fact = torch.LongTensor([])
        _label = torch.LongTensor([])
        for cur_fact in self.facts[self.ary][self.j: self.j+self.args.train_bs]:
            fact, label = self.corrupt(cur_fact)
            _fact = torch.cat([_fact, fact], dim=0)
            _label = torch.cat([_label, label], dim=0)

        self.j = min(self.j+self.args.train_bs, len(self.facts[self.ary]))
        return [_fact[:,_] for _ in range(len(_fact[0]))], _label


    def build_facts(self):
        facts_new = ddict(list)
        '''for LKGE and other baselines'''
        for fact in self.train_data:
            facts_new[len(fact)].append([fact[0],fact[1],fact[2]]+list(fact)[3:])
        ary_list = []
        for length, facts in facts_new.items():
            ary_list.append(length)
        
        return facts_new, ary_list

    def corrupt(self, fact):
        '''
        :param fact: positive facts
        :return: positive facts & negative facts ; pos/neg labels.
        '''
        facts = np.array([fact]).repeat(self.args.neg_ratio+1, axis=0)
        # todu
        if self.args.scorer_func in ['i_neuinfer', 'i_shrinke']:
            label = [1] + [0 for _ in range(self.args.neg_ratio)]
        else:
            label = [1] + [-1 for _ in range(self.args.neg_ratio)]
        rand_prob = np.random.randint(0, len(fact), self.args.neg_ratio)
        neg_e = np.random.randint(0, self.ents_num - 1, self.args.neg_ratio)
        neg_r = np.random.randint(0, self.rels_num - 1, self.args.neg_ratio)
        for _ in range(self.args.neg_ratio):
            if self.args.scorer_func in ['maker', 'cvt_dicgrl']:
                if rand_prob[_]%2 == 1:
                    facts[_+1][rand_prob[_]] = neg_e[_]
                else:
                    facts[_+1][rand_prob[_]] = neg_r[_]
            else:
                if rand_prob[_]%2 == 0:
                    facts[_+1][rand_prob[_]] = neg_e[_]
                else:
                    facts[_+1][rand_prob[_]] = neg_r[_]

        facts = torch.LongTensor(facts)
        label = torch.LongTensor(label)
        return facts, label


class MaskTrainDatasetMarginLoss:
    def __init__(self, args):
        self.args = args
        
        data = pickle.load(open(args.data_path, 'rb'))
        self.train_data = data['train']['triples']
        # g and pattern g
        self.g = self.get_train_g(self.train_data, len(data['ent2id']))
        self.pattern_g = self.get_pattern_g(self.train_data, len(data['rel2id']))
        self.train_data = [fact[1:16] for fact in self.train_data]
        
        self.rels_num = len(data['rel2id'])
        self.ents_num = len(data['ent2id'])
        self.train_ents = data['train_ents']
        self.train_rels = data['train_rels']

        self.tmp_seen_ents = torch.LongTensor(list(set(self.train_ents)))
        self.tmp_seen_rels = torch.LongTensor(list(set(self.train_rels)))
        
        self.facts, self.facts_ary, self.facts_list = self.build_facts()

        self.num_training_fact = len(self.train_data)
        
        edge_index_eH, edge_index_hrt, edge_type_hrt, edge_index_rv, edge_type_rv  = self.get_kg(self.train_data)
        self.edge_index_stare, self.edge_type_stare, self.quals_stare = self.build_graph(edge_index_hrt, edge_type_hrt, quals_index=edge_index_rv, quals_type=edge_type_rv)
        self.edge_index_eH = torch.LongTensor(edge_index_eH).to(self.args.gpu)#rv


    def get_kg(self, facts_list):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        edge_index_eH, edge_index_rv = [], []
        edge_type_eH, edge_type_rv = [], []
        edge_index_hrt, edge_type_hrt = [], []
        fact_id = 0
        if self.args.scorer_func == 'i_stare':
            for fact in facts_list:
                # 正事件
                edge_index_hrt.append([fact[0], fact[2]])
                edge_type_hrt.append(fact[1])
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                    if _ > 2 and _%2 == 1:
                        edge_index_rv.append([fact[_], fact[_+1]])
                        edge_type_rv.append(fact_id)
                fact_id += 1
                # 逆事件
                edge_index_hrt.append([fact[2], fact[0]])
                edge_type_hrt.append(fact[1]-1)
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                    if _ > 2 and _%2 == 1:
                        edge_index_rv.append([fact[_], fact[_+1]])
                        edge_type_rv.append(fact_id)
                fact_id += 1
        else:
            for fact in facts_list:
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                fact_id += 1

        return edge_index_eH, edge_index_hrt, edge_type_hrt, edge_index_rv, edge_type_rv


    def build_graph(self, edge_index, edge_type, quals_index=None, quals_type=None):
        if self.args.scorer_func != 'i_stare':
            return None, None, None
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu).transpose(1,0)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        edge_index_inv = torch.cat((edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)), dim=0)
        edge_index = torch.cat((edge_index, edge_index_inv), dim=1)
        edge_type_inv = edge_type-1
        edge_type = torch.cat((edge_type, edge_type_inv))
        
        quals_index = torch.LongTensor(quals_index).to(self.args.gpu)#rv
        quals_type = torch.LongTensor(quals_type).to(self.args.gpu)#fact id
        quals = torch.cat((quals_index[:, 0].unsqueeze(0), quals_index[:, 1].unsqueeze(0), quals_type.unsqueeze(0)), dim=0)
        quals_inv = torch.cat((quals_index[:, 0].unsqueeze(0), quals_index[:, 1].unsqueeze(0), quals_type.unsqueeze(0)), dim=0)
        quals = torch.cat((quals, quals_inv), dim=1)
        return edge_index, edge_type, quals


    def get_train_g(self, sup_tri, num_ent):
        
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
            
        num_fact = len(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_pattern_g(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        
        return g

    def reset(self):
        # i-th ary; j-th fact
        self.k = 0
        self.i = 0
        self.j = 0
        self.d = 0
        self.train_data = self.facts
        self.train_ary = self.facts_ary
        self.ary = self.train_ary[self.i]
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.j >= len(self.facts[self.ary]):
            if self.i < len(self.facts_ary) - 1:
                self.i += 1
                self.ary = self.train_ary[self.i]
                self.j = 0
                self.k = 0
            elif self.i == len(self.facts_ary) - 1:
                raise StopIteration
        _fact = torch.LongTensor(self.facts[self.ary][self.j: self.j+self.args.train_bs])
        
        if self.k%2 == 0:
            mask_output = torch.zeros(_fact.shape[0], self.ents_num) + (0.8/self.ents_num)
            mask_output[torch.arange(_fact.shape[0]), _fact[:,self.k]] = 1 - 0.8
        else:
            mask_output = torch.zeros(_fact.shape[0], self.rels_num) + (0.8/self.rels_num)
            mask_output[torch.arange(_fact.shape[0]), _fact[:,self.k]] = 1 - 0.8
        
        if self.k < len(_fact[0])-1:
            self.k += 1
            return [_fact[:,_] for _ in range(len(_fact[0]))], mask_output, self.k-1, -1
        else:
            self.k = 0
            self.j = min(self.j+self.args.train_bs, len(self.facts[self.ary]))
            return [_fact[:,_] for _ in range(len(_fact[0]))], mask_output, len(_fact[0])-1, -1


    def build_facts(self):
        '''
        build training data for each snapshots
        :return: training data
        '''
        facts_list = list()
        facts_new = ddict(list)
        '''for LKGE and other baselines'''
        for fact in self.train_data:
            facts_new[len(fact)].append([fact[0],fact[1],fact[2]]+list(fact)[3:])
            facts_list.append([fact[0],fact[1],fact[2]]+list(fact)[3:])
        
        ary_list = []
        for length, facts in facts_new.items():
            ary_list.append(length)
        return facts_new, ary_list, facts_list


class BaselineTestDataset():
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args):
        self.args = args
        self.train_bs = self.args.eval_batch_size
        
        data = pickle.load(open(args.data_path, 'rb'))
        self.num_ent = len(data['ent2id'])
        self.num_rel = len(data['rel2id'])
        
        # if istest:
        self.test_sup_facts = data['test']['support']
        self.test_que_facts = data['test']['query']
        self.uent_test_que_facts = data['test']['query_uent']
        self.urel_test_que_facts = data['test']['query_urel']
        self.uboth_test_que_facts = data['test']['query_uboth']
        # else:
        self.valid_sup_facts = data['valid']['support']
        self.valid_que_facts = data['valid']['query']
        
        self.uent_valid_que_facts = data['valid']['query_uent']
        self.urel_valid_que_facts = data['valid']['query_urel']
        self.uboth_valid_que_facts = data['valid']['query_uboth']
        # g and pattern g
        self.test_g = self.get_train_g(self.test_sup_facts, len(data['ent2id'])).to(self.args.gpu)
        self.valid_g = self.get_train_g(self.valid_sup_facts, len(data['ent2id'])).to(self.args.gpu)

        self.test_pattern_g = self.get_pattern_g(self.test_sup_facts, len(data['rel2id'])).to(self.args.gpu)
        self.valid_pattern_g = self.get_pattern_g(self.valid_sup_facts, len(data['rel2id'])).to(self.args.gpu)
        
        self.test_sup_facts = [fact[1:16] for fact in self.test_sup_facts]
        self.test_que_facts = [fact[1:16] for fact in self.test_que_facts]
        self.urel_test_que_facts = [fact[1:16] for fact in self.urel_test_que_facts]
        self.uent_test_que_facts = [fact[1:16] for fact in self.uent_test_que_facts]
        self.uboth_test_que_facts = [fact[1:16] for fact in self.uboth_test_que_facts]
        self.valid_sup_facts = [fact[1:16] for fact in self.valid_sup_facts]
        self.valid_que_facts = [fact[1:16] for fact in self.valid_que_facts]
        self.urel_valid_que_facts = [fact[1:16] for fact in self.urel_valid_que_facts]
        self.uent_valid_que_facts = [fact[1:16] for fact in self.uent_valid_que_facts]
        self.uboth_valid_que_facts = [fact[1:16] for fact in self.uboth_valid_que_facts]

        self.train_ents = torch.LongTensor(list(data['train_ents']))
        self.train_rels = torch.LongTensor(list(data['train_rels']))
        
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        
        self.ques2ans = self.get_ques2ans(self.valid_sup_facts + self.valid_que_facts + self.test_sup_facts + self.test_que_facts)


        '''prepare data for validation and testing'''
        self.eval_num_t, self.eval_num_v, self.eval_num_d = 0, 0, 0
        self.urel_eval_num_t, self.urel_eval_num_v, self.urel_eval_num_d = 0, 0, 0
        self.uent_eval_num_t, self.uent_eval_num_v, self.uent_eval_num_d = 0, 0, 0
        self.uboth_eval_num_t, self.uboth_eval_num_v, self.uboth_eval_num_d = 0, 0, 0
        self.valid, self.test, self.valid_ary, self.test_ary, self.urel_valid, self.urel_test, self.urel_valid_ary, self.urel_test_ary, self.uent_valid, self.uent_test, self.uent_valid_ary, self.uent_test_ary, self.uboth_valid, self.uboth_test, self.uboth_valid_ary, self.uboth_test_ary = self.build_facts()

        self.valid_num_training_fact = len(self.valid_sup_facts)
        valid_edge_index_eH, edge_index_hrt, edge_type_hrt, edge_index_rv, edge_type_rv  = self.get_kg(self.valid_sup_facts)
        self.valid_edge_index_stare, self.valid_edge_type_stare, self.valid_quals_stare = self.build_graph(edge_index_hrt, edge_type_hrt, quals_index=edge_index_rv, quals_type=edge_type_rv)
        self.valid_edge_index_eH = torch.LongTensor(valid_edge_index_eH).to(self.args.gpu)#rv

        self.test_num_training_fact = len(self.test_sup_facts)
        test_edge_index_eH, edge_index_hrt, edge_type_hrt, edge_index_rv, edge_type_rv  = self.get_kg(self.test_sup_facts)
        self.test_edge_index_stare, self.test_edge_type_stare, self.test_quals_stare = self.build_graph(edge_index_hrt, edge_type_hrt, quals_index=edge_index_rv, quals_type=edge_type_rv)
        self.test_edge_index_eH = torch.LongTensor(test_edge_index_eH).to(self.args.gpu)#rv


    def get_kg(self, facts_list):
        '''expand edge_index, edge_type (for GCN) and sr2o (to filter golden facts)'''
        edge_index_eH, edge_index_rv = [], []
        edge_type_eH, edge_type_rv = [], []
        edge_index_hrt, edge_type_hrt = [], []
        fact_id = 0
        if self.args.scorer_func == 'i_stare':
            for fact in facts_list:
                # 正事件
                edge_index_hrt.append([fact[0], fact[2]])
                edge_type_hrt.append(fact[1])
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                    if _ > 2 and _%2 == 1:
                        edge_index_rv.append([fact[_], fact[_+1]])
                        edge_type_rv.append(fact_id)
                fact_id += 1
                # 逆事件
                edge_index_hrt.append([fact[2], fact[0]])
                edge_type_hrt.append(fact[1]-1)
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                    if _ > 2 and _%2 == 1:
                        edge_index_rv.append([fact[_], fact[_+1]])
                        edge_type_rv.append(fact_id)
                fact_id += 1
        else:
            for fact in facts_list:
                for _, ele in enumerate(fact):
                    if _%2 == 0:
                        edge_index_eH.append([fact[_], fact_id])
                        if _ == 0:
                            edge_type_eH.append(fact[_+1])
                        else:
                            edge_type_eH.append(fact[_-1])
                fact_id += 1

        return edge_index_eH, edge_index_hrt, edge_type_hrt, edge_index_rv, edge_type_rv


    def build_graph(self, edge_index, edge_type, quals_index=None, quals_type=None):
        if self.args.scorer_func != 'i_stare':
            return None, None, None
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu).transpose(1,0)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        edge_index_inv = torch.cat((edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)), dim=0)
        edge_index = torch.cat((edge_index, edge_index_inv), dim=1)
        edge_type_inv = edge_type-1
        edge_type = torch.cat((edge_type, edge_type_inv))
        
        quals_index = torch.LongTensor(quals_index).to(self.args.gpu)#rv
        quals_type = torch.LongTensor(quals_type).to(self.args.gpu)#fact id
        quals = torch.cat((quals_index[:, 0].unsqueeze(0), quals_index[:, 1].unsqueeze(0), quals_type.unsqueeze(0)), dim=0)
        quals_inv = torch.cat((quals_index[:, 0].unsqueeze(0), quals_index[:, 1].unsqueeze(0), quals_type.unsqueeze(0)), dim=0)
        quals = torch.cat((quals, quals_inv), dim=1)
        return edge_index, edge_type, quals


    def get_train_g(self, sup_tri, num_ent):
        
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
            
        num_fact = len(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_pattern_g(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_ques2ans(self, facts):
        # pad_facts = [fact[:16] + (16-len(fact))*[0] for fact in facts]
        ques2ans = ddict(list)
        for fact in facts:
            # for i in range(16):
            for i in range(len(fact)):
                tmp_fact = dcopy(fact)
                tmp_fact[i] = 1
                cur_questr = tuple(tmp_fact)
                ques2ans[cur_questr].append(fact[i])
        return ques2ans


    def reset(self):
        # i-th ary; j-th fact; k-th index
        if self.args.valid:
            if self.args.utype == 'urel':
                self.eval_data = self.urel_valid
                self.eval_num = self.urel_eval_num_v
                self.eval_ary = self.urel_valid_ary
            elif self.args.utype == 'uent':
                self.eval_data = self.uent_valid
                self.eval_num = self.uent_eval_num_v
                self.eval_ary = self.uent_valid_ary
            elif self.args.utype == 'uboth':
                self.eval_data = self.uboth_valid
                self.eval_num = self.uboth_eval_num_v
                self.eval_ary = self.uboth_valid_ary

            
            self.g = self.valid_g
            self.pattern_g = self.valid_pattern_g
            
            self.edge_index_stare = self.valid_edge_index_stare
            self.edge_type_stare = self.valid_edge_type_stare
            self.quals_stare = self.valid_quals_stare
            self.edge_index_eH = self.valid_edge_index_eH
            self.num_training_fact = self.valid_num_training_fact
        
        else:
            if self.args.utype == 'urel':
                self.eval_data = self.urel_test
                self.eval_num = self.urel_eval_num_t
                self.eval_ary = self.urel_test_ary
            elif self.args.utype == 'uent':
                self.eval_data = self.uent_test
                self.eval_num = self.uent_eval_num_t
                self.eval_ary = self.uent_test_ary
            elif self.args.utype == 'uboth':
                self.eval_data = self.uboth_test
                self.eval_num = self.uboth_eval_num_t
                self.eval_ary = self.uboth_test_ary
            
            
            self.g = self.test_g
            self.pattern_g = self.test_pattern_g
            
            self.edge_index_stare = self.test_edge_index_stare
            self.edge_type_stare = self.test_edge_type_stare
            self.quals_stare = self.test_quals_stare
            self.edge_index_eH = self.test_edge_index_eH
            self.num_training_fact = self.test_num_training_fact
        
            
        self.i = 0
        self.ary = self.eval_ary[self.i] if len(self.eval_ary) != 0 else None
        self.j = 0
        self.k = 0
        self.b_ind = 0
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.eval_data) == 0:
            raise StopIteration
        # i: ary, j: fact, k: index
        if self.j >= len(self.eval_data[self.ary]):
            if self.i < len(self.eval_data) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.eval_ary[self.i]
                self.k = 0
            elif self.i == len(self.eval_data) - 1:
                raise StopIteration
        _fact = self.eval_data[self.ary][self.j: self.j+self.train_bs]

        _label = torch.LongTensor([])
        for cur_fact in _fact:
            # if self.args.valid:
            label = self.ques2ans[tuple(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:])]
            _label = torch.cat((_label, self.get_label(label).unsqueeze(0)), 0)
        
        if self.args.scorer_func not in ['i_hinge', 'i_hyconve', 'i_shrinke', 'i_neuinfer']:
            _fact = torch.LongTensor(_fact)
        else:
            _fact = np.array(_fact)

        self.b_ind += _label.shape[0]
        if self.k < len(_fact[0])-1:
            self.k += 1
            return [_fact[:,_] for _ in range(len(_fact[0]))], self.k-1, _label, _fact[:,self.k-1]
        else:
            self.k = 0
            self.j = min(self.j+self.train_bs, len(self.eval_data[self.ary]))
            return [_fact[:,_] for _ in range(len(_fact[0]))], len(_fact[0])-1, _label, _fact[:,-1]
        

    def get_label(self, label):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if self.args.scorer_func in ['maker', 'ReDA' ,'cvt_dicgrl']:
            if self.k%2==1:
                y = np.zeros([self.num_ent], dtype=np.float32)
            else:
                y = np.zeros([self.num_rel], dtype=np.float32)
        else:
            if self.k%2==0:
                y = np.zeros([self.num_ent], dtype=np.float32)
            else:
                y = np.zeros([self.num_rel], dtype=np.float32)

        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid, test = ddict(list), ddict(list)
        valid_ary, test_ary = [], []
        urel_valid, urel_test = ddict(list), ddict(list)
        urel_valid_ary, urel_test_ary = [], []
        uent_valid, uent_test = ddict(list), ddict(list)
        uent_valid_ary, uent_test_ary = [], []
        uboth_valid, uboth_test = ddict(list), ddict(list)
        uboth_valid_ary, uboth_test_ary = [], []
        '''for LKGE and other baselines'''

        for fact in self.test_que_facts:
            valid[len(fact)].append(fact)
            self.eval_num_v += len(fact)
        for fact in self.valid_que_facts:
            test[len(fact)].append(fact)
            self.eval_num_t += len(fact)
        
        for fact in self.urel_test_que_facts:
            urel_valid[len(fact)].append(fact)
            self.urel_eval_num_v += len(fact)
        for fact in self.urel_valid_que_facts:
            urel_test[len(fact)].append(fact)
            self.urel_eval_num_t += len(fact)
        
        for fact in self.uent_test_que_facts:
            uent_valid[len(fact)].append(fact)
            self.uent_eval_num_v += len(fact)
        for fact in self.uent_valid_que_facts:
            uent_test[len(fact)].append(fact)
            self.uent_eval_num_t += len(fact)
        
        for fact in self.uboth_test_que_facts:
            uboth_valid[len(fact)].append(fact)
            self.uent_eval_num_v += len(fact)
        for fact in self.uboth_valid_que_facts:
            uboth_test[len(fact)].append(fact)
            self.uboth_eval_num_t += len(fact)


        valid_ary = list(valid.keys())
        test_ary = list(test.keys())
        
        urel_valid_ary = list(urel_valid.keys())
        urel_test_ary = list(urel_test.keys())
        
        uent_valid_ary = list(uent_valid.keys())
        uent_test_ary = list(uent_test.keys())
        
        uboth_valid_ary = list(uboth_valid.keys())
        uboth_test_ary = list(uboth_test.keys())
        return valid, test, valid_ary, test_ary, urel_valid, urel_test, urel_valid_ary, urel_test_ary, uent_valid, uent_test, uent_valid_ary, uent_test_ary, uboth_valid, uboth_test, uboth_valid_ary, uboth_test_ary


class BaselineBinaryTestDataset():
    '''
    Dataloader for evaluation. For each snapshot, load the valid & test facts and filter the golden facts.
    '''
    def __init__(self, args):
        self.args = args
        self.train_bs = self.args.eval_batch_size
        
        data = pickle.load(open(args.data_path, 'rb'))
        self.num_ent = len(data['ent2id'])
        self.num_rel = len(data['rel2id'])
        
        # if istest:
        self.test_sup_facts = data['test']['support']
        self.test_que_facts = data['test']['query']
        self.uent_test_que_facts = data['test']['query_uent']
        self.urel_test_que_facts = data['test']['query_urel']
        self.uboth_test_que_facts = data['test']['query_uboth']
        # else:
        self.valid_sup_facts = data['valid']['support']
        self.valid_que_facts = data['valid']['query']
        self.uent_valid_que_facts = data['valid']['query_uent']
        self.urel_valid_que_facts = data['valid']['query_urel']
        self.uboth_valid_que_facts = data['valid']['query_uboth']
        # g and pattern g
        self.test_g = self.get_train_g(self.test_sup_facts, len(data['ent2id'])).to(self.args.gpu)
        self.valid_g = self.get_train_g(self.valid_sup_facts, len(data['ent2id'])).to(self.args.gpu)

        self.test_pattern_g = self.get_pattern_g(self.test_sup_facts, len(data['rel2id'])).to(self.args.gpu)
        self.valid_pattern_g = self.get_pattern_g(self.valid_sup_facts, len(data['rel2id'])).to(self.args.gpu)
        
        self.train_ents = torch.LongTensor(list(data['train_ents']))
        self.train_rels = torch.LongTensor(list(data['train_rels']))
        
        self.ent2id = data['ent2id']
        self.rel2id = data['rel2id']
        
        self.ques2ans = self.get_ques2ans(self.valid_sup_facts + self.valid_que_facts + self.test_sup_facts + self.test_que_facts)


        '''prepare data for validation and testing'''
        self.eval_num_t, self.eval_num_v, self.eval_num_d = 0, 0, 0
        self.urel_eval_num_t, self.urel_eval_num_v, self.urel_eval_num_d = 0, 0, 0
        self.uent_eval_num_t, self.uent_eval_num_v, self.uent_eval_num_d = 0, 0, 0
        self.uboth_eval_num_t, self.uboth_eval_num_v, self.uboth_eval_num_d = 0, 0, 0
        self.valid, self.test, self.valid_ary, self.test_ary, self.urel_valid, self.urel_test, self.urel_valid_ary, self.urel_test_ary, self.uent_valid, self.uent_test, self.uent_valid_ary, self.uent_test_ary, self.uboth_valid, self.uboth_test, self.uboth_valid_ary, self.uboth_test_ary = self.build_facts()
        
        self.edge_type_stare, self.edge_index_stare, self.quals_stare, self.edge_index_eH, self.num_training_fact = [], [], [], [], 0

    def get_train_g(self, sup_tri, num_ent):
        
        edge_index, edge_type = [], []
        fact_id = 0
        entity_set = set()
        for fact in sup_tri:
            for _, ele in enumerate(fact):
                if _%2 == 1:
                    entity_set.add(ele)
                    edge_index.append([fact[_], fact_id])
                    edge_type.append(fact[_-1])
            fact_id += 1
            
        num_fact = len(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]+num_ent), num_nodes=num_ent+num_fact).to(self.args.gpu)
        
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_pattern_g(self, sup_tri, num_rel):
        # adjacency matrix for rel and ent
        edge_index, edge_type = get_rel_metagraph(sup_tri)
        
        import dgl
        edge_index = torch.LongTensor(edge_index).to(self.args.gpu)
        edge_type = torch.LongTensor(edge_type).to(self.args.gpu)
        g = dgl.graph((edge_index[:, 0], edge_index[:, 1]), num_nodes=num_rel).to(self.args.gpu)
        g.edata['type'] = torch.tensor(edge_type)
        
        return g


    def get_ques2ans(self, facts):
        # pad_facts = [fact[:16] + (16-len(fact))*[0] for fact in facts]
        ques2ans = ddict(list)
        for fact in facts:
            # for i in range(16):
            for i in range(len(fact)):
                tmp_fact = dcopy(fact)
                tmp_fact[i] = 1
                cur_ques = tuple(tmp_fact)
                ques2ans[cur_ques].append(fact[i])
        return ques2ans


    def reset(self):
        # i-th ary; j-th fact; k-th index
        if self.args.valid:
            if self.args.utype == 'urel':
                self.eval_data = self.urel_valid
                self.eval_num = self.urel_eval_num_v
                self.eval_ary = self.urel_valid_ary
            elif self.args.utype == 'uent':
                self.eval_data = self.uent_valid
                self.eval_num = self.uent_eval_num_v
                self.eval_ary = self.uent_valid_ary
            elif self.args.utype == 'uboth':
                self.eval_data = self.uboth_valid
                self.eval_num = self.uboth_eval_num_v
                self.eval_ary = self.uboth_valid_ary

            self.g = self.valid_g
            self.pattern_g = self.valid_pattern_g
        
        else:
            if self.args.utype == 'urel':
                self.eval_data = self.urel_test
                self.eval_num = self.urel_eval_num_t
                self.eval_ary = self.urel_test_ary
            elif self.args.utype == 'uent':
                self.eval_data = self.uent_test
                self.eval_num = self.uent_eval_num_t
                self.eval_ary = self.uent_test_ary
            elif self.args.utype == 'uboth':
                self.eval_data = self.uboth_test
                self.eval_num = self.uboth_eval_num_t
                self.eval_ary = self.uboth_test_ary
            
            self.g = self.test_g
            self.pattern_g = self.test_pattern_g
        
            
        self.i = 0
        self.ary = self.eval_ary[self.i] if len(self.eval_ary) != 0 else None
        self.j = 0
        self.k = 0
        self.b_ind = 0
        return self
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self.eval_data) == 0:
            raise StopIteration
        # i: ary, j: fact, k: index
        if self.j >= len(self.eval_data[self.ary]):
            if self.i < len(self.eval_data) - 1:
                self.j = 0
                self.i += 1
                self.ary = self.eval_ary[self.i]
                self.k = 0
            elif self.i == len(self.eval_data) - 1:
                raise StopIteration
        _fact = self.eval_data[self.ary][self.j: self.j+self.train_bs]

        _label = torch.LongTensor([])
        for cur_fact in _fact:
            label = self.ques2ans[tuple(list(cur_fact)[:self.k]+[1]+list(cur_fact)[self.k+1:])]
            _label = torch.cat((_label, self.get_label(label).unsqueeze(0)), 0)
        
        _fact = np.array(_fact)

        self.b_ind += _label.shape[0]
        if self.k < len(_fact[0])-1:
            self.k += 1
            return [_fact[:,_] for _ in range(len(_fact[0]))], self.k-1, _label, _fact[:,self.k-1]
        else:
            self.k = 0
            self.j = min(self.j+self.train_bs, len(self.eval_data[self.ary]))
            return [_fact[:,_] for _ in range(len(_fact[0]))], len(_fact[0])-1, _label, _fact[:,-1]
        

    def get_label(self, label):
        '''
        Filter the golden facts. The label 1.0 denote that the entity is the golden answer.
        :param label:
        :return: dim = test factnum * all seen entities
        '''
        if self.k%2==1:
            y = np.zeros([self.num_ent], dtype=np.float32)
        else:
            y = np.zeros([self.num_rel], dtype=np.float32)

        for e2 in label: 
            y[e2] = 1.0
        return torch.FloatTensor(y)

    def build_facts(self):
        '''
        build validation and test set using the valid & test data for each snapshots
        :return: validation set and test set
        '''
        valid, test = ddict(list), ddict(list)
        valid_ary, test_ary = [], []
        urel_valid, urel_test = ddict(list), ddict(list)
        urel_valid_ary, urel_test_ary = [], []
        uent_valid, uent_test = ddict(list), ddict(list)
        uent_valid_ary, uent_test_ary = [], []
        uboth_valid, uboth_test = ddict(list), ddict(list)
        uboth_valid_ary, uboth_test_ary = [], []
        '''for LKGE and other baselines'''

        for fact in self.test_que_facts:
            valid[len(fact)].append(fact)
            self.eval_num_v += len(fact)
        for fact in self.valid_que_facts:
            test[len(fact)].append(fact)
            self.eval_num_t += len(fact)
        
        for fact in self.urel_test_que_facts:
            urel_valid[len(fact)].append(fact)
            self.urel_eval_num_v += len(fact)
        for fact in self.urel_valid_que_facts:
            urel_test[len(fact)].append(fact)
            self.urel_eval_num_t += len(fact)
        
        for fact in self.uent_test_que_facts:
            uent_valid[len(fact)].append(fact)
            self.uent_eval_num_v += len(fact)
        for fact in self.uent_valid_que_facts:
            uent_test[len(fact)].append(fact)
            self.uent_eval_num_t += len(fact)
        
        for fact in self.uboth_test_que_facts:
            uboth_valid[len(fact)].append(fact)
            self.uent_eval_num_v += len(fact)
        for fact in self.uboth_valid_que_facts:
            uboth_test[len(fact)].append(fact)
            self.uboth_eval_num_t += len(fact)


        valid_ary = list(valid.keys())
        test_ary = list(test.keys())
        
        urel_valid_ary = list(urel_valid.keys())
        urel_test_ary = list(urel_test.keys())
        
        uent_valid_ary = list(uent_valid.keys())
        uent_test_ary = list(uent_test.keys())
        
        uboth_valid_ary = list(uboth_valid.keys())
        uboth_test_ary = list(uboth_test.keys())
        return valid, test, valid_ary, test_ary, urel_valid, urel_test, urel_valid_ary, urel_test_ary, uent_valid, uent_test, uent_valid_ary, uent_test_ary, uboth_valid, uboth_test, uboth_valid_ary, uboth_test_ary

