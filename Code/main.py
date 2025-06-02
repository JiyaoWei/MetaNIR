import os
import argparse
from utils import init_dir
from meta_trainer import MetaTrainer
from subgraph import gen_subgraph_datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='JF17K')
    parser.add_argument('--state_dir', default='./state')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--tb_log_dir', default='./tb_log')
    parser.add_argument('--data_num', default='_0')

    parser.add_argument('--train_bs', default=64, type=int)
    parser.add_argument('--eval_bs', default=64, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_step', default=100000, type=int)
    parser.add_argument('--pre_train_epoch', default=10, type=int)
    parser.add_argument('--log_per_step', default=1, type=int)
    parser.add_argument('--check_per_step', default=10, type=int)
    parser.add_argument('--early_stop_patience', default=3, type=int)
    parser.add_argument('--eval_times', default=5, type=int)
    parser.add_argument('--num_sample_cand', default=50, type=int)
    parser.add_argument('--neg_ratio', default=10, type=int)

    parser.add_argument('--dim', default=128, type=int)
    parser.add_argument('--num_rel_bases', default=128, type=int)

    parser.add_argument('--metatrain_num_neg', default=32)
    parser.add_argument('--adv_temp', default=1, type=float)
    parser.add_argument('--gamma', default=10, type=float)

    parser.add_argument('--cpu_num', default=10, type=float)
    parser.add_argument('--gpu', default='cuda:0', type=str)
    parser.add_argument("--entity_soft", type=float, default=0.8)
    parser.add_argument("--relation_soft", type=float, default=0.9)
    # subgraph
    parser.add_argument('--db_path', default=None)
    parser.add_argument('--num_train_subgraph', default=10000, type=int)
    parser.add_argument('--num_sample_for_estimate_size', default=10, type=int)
    parser.add_argument('--rw_0', default=10, type=int)
    parser.add_argument('--rw_1', default=10, type=int)
    parser.add_argument('--rw_2', default=5, type=int)
    parser.add_argument('--ent_neighbor', default=5, type=int)
    parser.add_argument('--rel_neighbor', default=5, type=int)
    parser.add_argument('--sub_train_ent_num', default=50, type=int)
    parser.add_argument('--train_task_query_rate', default=0.1, type=float)
    parser.add_argument('--task_mask_rate', default=0.1, type=float)
    parser.add_argument('--train_background', default='part', type=str)
    parser.add_argument('--subgraph_type', default='v2', type=str)
    parser.add_argument('--use_seed', default='True', type=str)
    parser.add_argument('--save_emb', default='False', type=str)
    
    
    parser.add_argument('--adjustment', default='False', type=str)
    parser.add_argument('--pretraining', default='False', type=str)
    parser.add_argument('--metalearning', default='False', type=str)
    parser.add_argument('--initent', default='False', type=str)
    parser.add_argument('--initrel', default='False', type=str)
    parser.add_argument('--gnnent', default='False', type=str)
    parser.add_argument('--gnnrel', default='False', type=str)
    parser.add_argument('--ent_beta', default=5, type=float)
    parser.add_argument('--rel_beta', default=5, type=float)
    parser.add_argument('--u_redu', default='False', type=str)
    parser.add_argument('--pred_truth', default='False', type=str)
    parser.add_argument('--adjustment_type', default='part', type=str)
    parser.add_argument('--adjustment_reduce', default='True', type=str)
    
    parser.add_argument('--scorer_func', default='MNKGE', type=str)
    parser.add_argument('--rel_transfer_method', default='rgraph', type=str)
    parser.add_argument('--ent_transfer_method', default='egraph', type=str)
    parser.add_argument('--egraph_act', dest='egraph_act', default='tanh')
    parser.add_argument('--egraph_gcn_dim', dest='egraph_gcn_dim', default=128, type=int)
    parser.add_argument('--egraph_dropout', dest='egraph_dropout', default=0.1, type=float)
    parser.add_argument('--egraph_opn', dest='egraph_opn', default='sub', type=str)
    parser.add_argument('--egraph_bias', dest='egraph_bias', default="False")
    parser.add_argument('--egraph_bn', dest='egraph_bn', default="False")
    parser.add_argument('--egraph_num_layers', dest='egraph_num_layers', default=2, type=int)
    parser.add_argument('--rgraph_act', dest='rgraph_act', default='tanh')
    parser.add_argument('--rgraph_gcn_dim', dest='rgraph_gcn_dim', default=4, type=int)
    parser.add_argument('--rgraph_dropout', dest='rgraph_dropout', default=0.1, type=float)
    parser.add_argument('--rgraph_bias', dest='rgraph_bias', default="False")
    parser.add_argument('--rgraph_bn', dest='rgraph_bn', default="False")
    parser.add_argument('--rgraph_attn', dest='rgraph_attn', default="none")# ful, semantic, statistic, none
    parser.add_argument('--rgraph_num_layers', dest='rgraph_num_layers', default=2, type=int)
    parser.add_argument('--Trans_layers', dest='Trans_layers', default=12, type=int, help='The number of local layers')
    parser.add_argument('--Trans_heads', dest='Trans_heads', default=4, type=int, help='The head number of local layers')
    parser.add_argument('--Trans_drop', dest='Trans_drop', default=0.1, type=int, help='The dropout rate of local layers')
    parser.add_argument('--Trans_hid_dim', dest='Trans_hid_dim', default=128, type=int)
    

    args = parser.parse_args()

    args.task_name = args.dataset + '_ext'
    args.data_path = './data/' + args.dataset + '/test_data' + args.data_num + '.pkl'
    
    if args.dataset == 'FI_WD20K100_v1':
        args.num_sample_cand = 0
    
    if args.scorer_func == 'maker':
        args.train_bs = 64
        args.eval_batch_size = 1
        # args.early_stop_patience = 3
        # args.check_per_step = 30
    # elif args.scorer_func == 'cvt_maker':
    #     args.train_bs = 256
    #     args.eval_batch_size = 1
    #     args.early_stop_patience = 3
        # args.check_per_step = 30
    elif args.scorer_func in ['i_hinge', 'i_neuinfer', 'i_hyconve', 'i_shrinke']:
        # args.train_bs = 256
        args.eval_batch_size = 1
        # args.lr = 1e-4
    elif args.scorer_func in ['i_hytransformer', 'i_hahe', 'i_gran', 'i_stare']:
        # args.train_bs = 128
        args.eval_batch_size = 128
        # args.lr = 1e-4
    

    args.egraph_gcn_dim = args.dim
    # if args.scorer_func == 'maker':
    #     args.ent_dim = args.dim * 2
    #     args.rel_dim = args.dim
    # else:
    args.ent_dim = args.dim
    args.rel_dim = args.dim
    args.Trans_hid_dim = args.dim
    args.num_rel_bases = args.dim
    args.rgraph_gcn_dim = args.num_rel_bases
    

    args.db_path = args.data_path[:-4] + '_subgraph' + '_' + args.subgraph_type + '_' + str(args.sub_train_ent_num) + '_' + str(args.train_task_query_rate) + '_' + str(args.num_train_subgraph) + '_' + str(args.rw_0) + '_' + str(args.rw_1) + '_' + str(args.rw_2) + '_' + str(args.ent_neighbor) + '_' + str(args.rel_neighbor) + '_' + args.u_redu

    if not os.path.exists(args.db_path) and args.metalearning == 'True' and args.scorer_func == 'MNKGE':
        gen_subgraph_datasets(args)
    if args.scorer_func == 'maker':
        gen_subgraph_datasets(args)

    init_dir(args)
    if args.scorer_func == 'MNKGE':
        args.exp_name = args.data_path.split('/')[2] + args.data_num + '_' + args.scorer_func + '_' + args.metalearning[0] + '_' + args.adjustment[0] + '_'  + args.pretraining[0] + '_' + args.initrel[0] + '_' + args.initent[0] + '_' + args.gnnrel[0] + '_' + args.gnnent[0] + '_' + args.u_redu[0] + '_' + args.pred_truth[0] + f'_lr_{args.lr}' + f'_{args.dim}' + f'_{args.num_rel_bases}' + f'_{args.pre_train_epoch}' + '_' + str(args.sub_train_ent_num) + '_' + str(args.train_task_query_rate) + '_' + str(args.num_train_subgraph) + '_' + str(args.rw_0) + '_' + str(args.rw_1) + '_' + str(args.rw_2) + '_' + str(args.task_mask_rate) + '_' + str(args.ent_beta) + '_' + str(args.rel_beta) + '_' + args.train_background + '_' + args.subgraph_type + '_' + str(args.eval_times) + '_' + str(args.ent_neighbor) + '_' + str(args.rel_neighbor) + '_' + args.adjustment_type + '_' + args.adjustment_reduce[0] + '_' + args.use_seed + '_' + str(args.egraph_num_layers)
    else:
        args.exp_name = args.data_path.split('/')[2] + args.data_num + '_' + args.scorer_func + '_' + str(args.eval_times) + '_' + str(args.lr) + '_' + str(args.dim)


    trainer = MetaTrainer(args)
    if args.scorer_func in ['MNKGE']:
        trainer.train()
    elif args.scorer_func in ['maker']:
        trainer.binary_baseline_train()
    else:
        trainer.baseline_train()