import pickle
import numpy as np
from utils import serialize
from copy import deepcopy as dcopy
import lmdb
from collections import defaultdict as ddict
from tqdm import tqdm
import random
import multiprocessing as mp


def gen_subgraph_datasets(args):
    print('----------generate tasks (sub-KGs) for meta-training----------')
    data = pickle.load(open(args.data_path, 'rb'))
    bg_train_g = data['train']['triples']
    
    # num_sample_for_estimate_size 10
    BYTES_PER_DATUM = get_average_subgraph_size(args, args.num_sample_for_estimate_size, bg_train_g) * 2
    map_size = (args.num_train_subgraph) * BYTES_PER_DATUM
    env = lmdb.open(args.db_path, map_size=map_size, max_dbs=1)
    train_subgraphs_db = env.open_db("train_subgraphs".encode())

    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        idx_ = range(args.num_train_subgraph)
        for (str_id, datum) in tqdm(p.imap(sample_one_subgraph, idx_), total=args.num_train_subgraph):
            with env.begin(write=True, db=train_subgraphs_db) as txn:
                txn.put(str_id, serialize(datum))


def intialize_worker(args, bg_train_g):
    global args_, bg_train_g_
    args_, bg_train_g_ = args, bg_train_g

def gen_ent2ent(sts):
    ent2ent = ddict(set)
    for st in sts:
        for id_1, ent in enumerate(st[1::2]):
            for id_2, ent2 in enumerate(st[1::2]):
                if id_1 != id_2:
                    ent2ent[ent].add(ent2)
                    ent2ent[ent2].add(ent)
    return ent2ent

def gen_ele2facts(sts):
    ent2facts = ddict(list)
    rel2facts = ddict(list)
    for st in sts:
        for id_1, ent in enumerate(st[1::2]):
            ent2facts[ent].append(st)
        for id_1, ent in enumerate(st[0::2]):
            rel2facts[ent].append(st)
    return ent2facts, rel2facts

def sample_one_subgraph(idx_):
    args = args_
    bg_train_g = [fact[:16] for fact in bg_train_g_]
    
    if args.subgraph_type == 'v2':
        bg_train_ent2facts, bg_train_rel2facts = gen_ele2facts(bg_train_g)
        
        
        que_facts_index = random.sample(range(len(bg_train_g)), 128)
        que_facts = [bg_train_g[index] for index in que_facts_index]
            
        que_ents, que_rels = set(), set()
        que_fact_tupples = set()
        for fact in que_facts:
            que_fact_tupples.add(tuple(fact))
            for ele in fact[1::2]:
                que_ents.add(ele)
            for ele in fact[0::2]:
                que_rels.add(ele)
        
        sup_facts = []
        for ele in que_ents:
            cur_neighbor = []
            for fact in bg_train_ent2facts[ele]:
                if tuple(fact) not in que_fact_tupples:
                    cur_neighbor.append(fact)
                if len(cur_neighbor) >= args.ent_neighbor:
                    break
            sup_facts += cur_neighbor
        
        for ele in que_rels:
            cur_neighbor = []
            # 避免了重复
            for fact in bg_train_rel2facts[ele]:
                if tuple(fact) not in que_fact_tupples:
                    cur_neighbor.append(fact)
                if len(cur_neighbor) >= args.rel_neighbor:
                    break
            sup_facts += cur_neighbor
    elif args.subgraph_type == 'v1':    
    # # induce sub-graph by sampled nodes
        bg_train_ent2ent = gen_ent2ent(bg_train_g)
        while True:
            while True:
                # 从不同的结点出发，游走rw_0次 # 每次游走时，在同一个结点重复rw_1次 # 每次游走rw_2步
                sel_nodes = []
                for i in range(args.rw_0):
                    if i == 0:
                        total_cand = len(bg_train_ent2ent)
                        cand_nodes = np.arange(total_cand)
                        start_node = int(np.random.choice(cand_nodes, 1, replace=False)[0])
                        while len(list(bg_train_ent2ent[start_node])) < 2:
                            start_node = int(np.random.choice(cand_nodes, 1, replace=False)[0])
                    else:
                        # if len(sel_nodes) == 0:
                        #     break
                        start_node = int(np.random.choice(sel_nodes, 1, replace=False)[0])
                    for time_ in range(args.rw_1):
                        state_node = int(start_node)
                        for len_ in range(args.rw_2):
                            if len(bg_train_ent2ent[state_node]) > 0:
                                state_node = int(np.random.choice(np.array(list(bg_train_ent2ent[state_node])), 1, replace=False))
                                if state_node not in sel_nodes:
                                    sel_nodes.append(state_node)
                            else:
                                break
                # 根据采样的种子得到子图sub_train_g
                sub_train_g, sub_train_ents = [], set()
                for st in bg_train_g:
                    sign = 1
                    for ele in st[1::2]:
                        if ele not in sel_nodes:
                            sign = 0
                            break
                    if sign:
                        sub_train_g.append(st)
                        sub_train_ents = sub_train_ents | set(st[1::2])

                if len(sub_train_ents) >= args.sub_train_ent_num:
                    break

            random.shuffle(sub_train_g)

            rel_freq, ent_freq = ddict(int), ddict(int)
            for tri in sub_train_g:
                for _, ele in enumerate(tri):
                    if _ % 2 == 0:
                        rel_freq[ele] += 1
                    else:
                        ent_freq[ele] += 1

            # randomly get query triples，如果元素的频率小于等于2，则其不能作为query，support中样例太少了随机性强
            que_facts, sup_facts = [], []
            for idx, tri in enumerate(sub_train_g):
                sign = 1
                for _, ele in enumerate(tri):
                    if _ % 2 == 0:
                        if rel_freq[ele] <= 2:
                            sign = 0
                    else:
                        if ent_freq[ele] <= 2:
                            sign = 0
                if sign == 0:
                    if tri not in sup_facts:
                        sup_facts.append(tri)
                else:
                    if tri not in que_facts:
                        que_facts.append(tri)
                    for _, ele in enumerate(tri):
                        if _ % 2 == 0:
                            rel_freq[ele] -= 1
                        else:
                            ent_freq[ele] -= 1
                if len(que_facts) >= int(len(sub_train_g)*args.train_task_query_rate):
                    break

            sup_facts.extend(sub_train_g[idx+1:])

            if len(que_facts) >= int(len(sub_train_g)*args.train_task_query_rate):
                break

    # edge_index_r_f, edge_type_r_f = get_rel_metagraph(sup_facts, u_redu = "False")
    edge_index_r, edge_type_r = get_rel_metagraph(sup_facts, u_redu = args.u_redu)
    edge_index_e, edge_type_e = get_ent_graph(sup_facts)

    str_id = '{:08}'.format(idx_).encode('ascii')

    # que_facts_padding = [fact[:16] + (16-len(fact[:16]))*[0] for fact in que_facts]
    # sup_facts_padding = [fact[:16] + (16-len(fact[:16]))*[0] for fact in sup_facts]
    # for fact in que_facts_padding + sup_facts_padding + sup_facts + que_facts:
    ques2ans = ddict(list)
    for fact in sup_facts + que_facts:
        for i in range(len(fact)):
            tmp_fact = dcopy(fact)
            tmp_fact[i] = -1
            cur_questr = tuple(tmp_fact)
            ques2ans[cur_questr].append(fact[i])
    return str_id, (sup_facts, que_facts, edge_index_e, edge_type_e, edge_index_r, edge_type_r, ques2ans)

def get_ent_graph(facts_list):
    edge_index_e, edge_type_e = [], []
    fact_id = 0
    for fact in facts_list:
        for _, ele in enumerate(fact):
            if _%2 == 1:
                edge_index_e.append([fact[_], fact_id])
                edge_type_e.append(fact[_-1])
        fact_id += 1
    return edge_index_e, edge_type_e

def get_rel_metagraph(facts_list, u_redu = "False"):
    # 相同位置，相同相邻结点，相邻结点角色相同，比如都为r，都有e作为头实体
    # 最相关实体之间存在联系，q的e，r的h/t
    # 相同位置，相同相邻结点，但相邻结点角色不同，比如都为r，e是r1的头实体，是r2的尾实体
    # 处在一个事件内，共存或者主辅关系
    def index2fea(ind, type):
        if type == 'extra':
            # 不同事件内，以相关联的实体类型为特征
            if ind == 1:
                return 'h'
            elif ind == 3:
                return 't'
            else:
                return 'e'
        elif type == 'inter':
            # 同一个事件内，以关系位置作为特征
            if ind == 0:
                return 'r_h'
            elif ind == 2:
                return 'r_t'
            else:
                return 'q'
    def fea2type(fea):
        if fea[0] == 'r_h' and fea[1] == 'r_t':
            return 0
        elif fea[0] == 'r_t' and fea[1] == 'r_h':
            return 1
        elif fea[0] == 'r_h' and fea[1] == 'q':
            return 2
        elif fea[0] == 'q' and fea[1] == 'r_h':
            return 3
        elif fea[0] == 'r_t' and fea[1] == 'q':
            return 4
        elif fea[0] == 'q' and fea[1] == 'r_t':
            return 5
        elif fea[0] == 'q' and fea[1] == 'q':
            return 6
        
        elif fea[0] == 'h' and fea[1] == 'h':
            return 7
        elif fea[0] == 't' and fea[1] == 't':
            return 8
        elif fea[0] == 'e' and fea[1] == 'e':
            return 9
        elif fea[0] == 'h' and fea[1] == 't':
            return 10
        elif fea[0] == 'h' and fea[1] == 'e':
            return 11
        elif fea[0] == 't' and fea[1] == 'h':
            return 12
        elif fea[0] == 't' and fea[1] == 'e':
            return 13
        elif fea[0] == 'e' and fea[1] == 'h':
            return 14
        elif fea[0] == 'e' and fea[1] == 't':
            return 15
        else:
            print('Error in fea2type')
            assert 1==2
            
    cur_rels, cur_facts = set(), facts_list
    for fact in facts_list:
        for rel in fact[0::2]:
            cur_rels.add(rel)
    
    inter_meta_relation = dict()
    extra_meta_relation = dict()
    for rel in list(cur_rels):
        for first_fact in cur_facts:
            if rel in first_fact[0::2]:
                rel_index = first_fact[0::2].index(rel)
                rel_index = rel_index*2
                # 二元事件中的头角色和尾角色同样存在模式特征
                for ind, ele in enumerate(first_fact):
                    if ind != rel_index and ind%2==0:
                        if tuple([rel, ele]) not in inter_meta_relation:
                            # 返回边和实体的模式特征
                            inter_meta_relation[tuple([rel, ele])] = [(index2fea(rel_index, 'inter'), index2fea(ind, 'inter'))]
                        else:
                            if u_redu == "True":
                                inter_meta_relation[tuple([rel, ele])].append((index2fea(rel_index, 'inter'), index2fea(ind, 'inter')))
                            else:
                                if (index2fea(rel_index, 'inter'), index2fea(ind, 'inter')) not in inter_meta_relation[tuple([rel, ele])]:
                                    inter_meta_relation[tuple([rel, ele])].append((index2fea(rel_index, 'inter'), index2fea(ind, 'inter')))
                                

                related_ents_set = set([first_fact[rel_index+1]])
                for second_fact in cur_facts:
                    if second_fact == first_fact:   continue
                    com_ents = set(second_fact[1::2]) & related_ents_set
                    if len(com_ents) > 0:
                        for ent in com_ents:
                            second_ent_ind = second_fact[1::2].index(ent)
                            second_ent_ind = second_ent_ind*2+1
                            first_ent_ind = first_fact[1::2].index(ent)
                            first_ent_ind = first_ent_ind*2+1
                            related_rel = second_fact[second_ent_ind-1]
                            if tuple([rel, related_rel]) not in extra_meta_relation:
                                extra_meta_relation[tuple([rel, related_rel])] = [(index2fea(first_ent_ind, 'extra'), index2fea(second_ent_ind, 'extra'))]
                            else:
                                if u_redu == "True":
                                    extra_meta_relation[tuple([rel, related_rel])].append((index2fea(first_ent_ind, 'extra'), index2fea(second_ent_ind, 'extra')))
                                else:
                                    if (index2fea(first_ent_ind, 'extra'), index2fea(second_ent_ind, 'extra')) not in extra_meta_relation[tuple([rel, related_rel])]:
                                        extra_meta_relation[tuple([rel, related_rel])].append((index2fea(first_ent_ind, 'extra'), index2fea(second_ent_ind, 'extra')))
    edge_index_rm, edge_type_rm = [], []
    meta_relation = dcopy(inter_meta_relation)
    for key, values in extra_meta_relation.items():
        if key not in meta_relation:
            meta_relation[key] = values
        else:
            meta_relation[key] += values

    for key, values in meta_relation.items():
        for value in values:
            edge_index_rm.append([key[0], key[1]])
            edge_type_rm.append(fea2type(value))

    return edge_index_rm, edge_type_rm


def get_average_subgraph_size(args, sample_size, bg_train_g):
    total_size = 0
    
    with mp.Pool(processes=10, initializer=intialize_worker, initargs=(args, bg_train_g)) as p:
        idx_ = range(sample_size)
        # intialize_worker(args, bg_train_g)
        # sample_one_subgraph(1)
        for (str_id, datum) in p.imap(sample_one_subgraph, idx_):
            total_size += len(serialize(datum))

    return total_size / sample_size

