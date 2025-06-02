from torch.utils.tensorboard import SummaryWriter
from data import ValidData, TestData, TestDataset, InTestDataset, FewTestDataset
from collections import defaultdict as ddict
from utils import Log
import numpy as np
import random
import pickle
import torch
import json
import csv
import os


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # writer and logger
        self.name = args.exp_name
        self.writer = SummaryWriter(os.path.join(args.tb_log_dir, self.name))
        self.logger = Log(args.log_dir, self.name).get_logger()
        self.logger.info(json.dumps(vars(args)))

        # state dir
        self.state_path = os.path.join(args.state_dir, self.name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # load data
        self.data = pickle.load(open(args.data_path, 'rb'))
        args.num_ent = len(self.data['ent2id'])
        args.num_rel = len(self.data['rel2id'])

        # dataset for validation and testing
        self.valid_data = ValidData(args, self.data)
        self.test_data = TestData(args, self.data)
        
        if self.args.use_seed == 'True':
            OMP_NUM_THREADS = 8
            torch.manual_seed(0)
            random.seed(0)
            np.random.seed(0)
            torch.autograd.set_detect_anomaly(True)
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(8)
            torch.cuda.empty_cache()

    def write_training_loss(self, loss, step):
        self.writer.add_scalar("training/loss", loss, step)


    def write_evaluation_result(self, results, e):
        self.writer.add_scalar("evaluation/mrr", results['mrr'], e)
        self.writer.add_scalar("evaluation/hits10", results['hits10'], e)
        self.writer.add_scalar("evaluation/hits5", results['hits5'], e)
        self.writer.add_scalar("evaluation/hits1", results['hits1'], e)


    def write_rst_csv(self, suffix_dict, query_part):
        for suf, rst in suffix_dict.items():
            with open(os.path.join(self.args.log_dir, f"{self.args.data_path.split('/')[2]}_{suf}_{query_part}.csv"), "a") as rstfile:
                rst_writer = csv.writer(rstfile)
                rst_writer.writerow([self.name, round(rst["mrr"], 4), round(rst["hits1"], 4),
                                     round(rst["hits5"], 4), round(rst["hits10"], 4)])


    def save_checkpoint(self, e, state):
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.name, self.name + '.' + str(e) + '.ckpt'))


    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.name + '.' + str(best_step) + '.ckpt'), os.path.join(self.state_path, self.name + '.best'))


    def baseline_train(self):
        best_step = 0
        self.logger.info('start training')
        self.model.to(self.args.gpu)
        '''Training iteration'''
        self.best_valid = 0.0
        self.stop_epoch = 0
        self.args.valid_metrics = 'mrr'
        for i in range(1, self.args.num_step + 1):
            loss = self.process_epoch_train()
 
            if i % self.log_per_step == 0:
                self.logger.info('Training step: {} | loss: {:.4f}'.format(i, loss))
                self.write_training_loss(loss, i)

            if i % self.check_per_step == 0 and i != 1:
                detail_results = self.process_epoch_eval()
                
                self.args.valid = False
                
                if self.best_valid < detail_results['total'][self.args.valid_metrics]:
                    self.best_valid = detail_results['total'][self.args.valid_metrics]
                    self.stop_epoch = 0
                    self.logger.info('Training best model | mrr {:.4f}'.format(self.best_valid))
                    self.save_checkpoint(i, self.get_curr_state())
                    best_step = i
                else:
                    self.stop_epoch += 1
                    self.logger.info('Training best model | mrr {:.4f} at {}'.format(self.best_valid, best_step))
                    
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Total',
                    round(detail_results['total']['mrr'] * 100, 2), round(detail_results['total']['hits1'] * 100, 2),
                    round(detail_results['total']['hits5'] * 100, 2), round(detail_results['total']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent',
                    round(detail_results['entity']['mrr'] * 100, 2), round(detail_results['entity']['hits1'] * 100, 2),
                    round(detail_results['entity']['hits5'] * 100, 2), round(detail_results['entity']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_uent',
                    round(detail_results['uent_ent']['mrr'] * 100, 2), round(detail_results['uent_ent']['hits1'] * 100, 2),
                    round(detail_results['uent_ent']['hits5'] * 100, 2), round(detail_results['uent_ent']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_urel',
                    round(detail_results['urel_ent']['mrr'] * 100, 2), round(detail_results['urel_ent']['hits1'] * 100, 2),
                    round(detail_results['urel_ent']['hits5'] * 100, 2), round(detail_results['urel_ent']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_uboth',
                    round(detail_results['uboth_ent']['mrr'] * 100, 2), round(detail_results['uboth_ent']['hits1'] * 100, 2),
                    round(detail_results['uboth_ent']['hits5'] * 100, 2), round(detail_results['uboth_ent']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel',
                    round(detail_results['relation']['mrr'] * 100, 2), round(detail_results['relation']['hits1'] * 100, 2),
                    round(detail_results['relation']['hits5'] * 100, 2), round(detail_results['relation']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_uent',
                    round(detail_results['uent_rel']['mrr'] * 100, 2), round(detail_results['uent_rel']['hits1'] * 100, 2),
                    round(detail_results['uent_rel']['hits5'] * 100, 2), round(detail_results['uent_rel']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_urel',
                    round(detail_results['urel_rel']['mrr'] * 100, 2), round(detail_results['urel_rel']['hits1'] * 100, 2),
                    round(detail_results['urel_rel']['hits5'] * 100, 2), round(detail_results['urel_rel']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_uboth',
                    round(detail_results['uboth_rel']['mrr'] * 100, 2), round(detail_results['uboth_rel']['hits1'] * 100, 2),
                    round(detail_results['uboth_rel']['hits5'] * 100, 2), round(detail_results['uboth_rel']['hits10'] * 100, 2)))
                
            if self.stop_epoch >= self.args.early_stop_patience:
                self.logger.info('Early Stopping! Epoch: {} Best Results: {}'.format(i, round(self.best_valid*100, 3)))
                break


        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_step)


        self.before_test_load()
        detail_results = self.process_epoch_eval(istest=True)

        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Total',
            round(detail_results['total']['mrr'] * 100, 2), round(detail_results['total']['hits1'] * 100, 2),
            round(detail_results['total']['hits5'] * 100, 2), round(detail_results['total']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent',
            round(detail_results['entity']['mrr'] * 100, 2), round(detail_results['entity']['hits1'] * 100, 2),
            round(detail_results['entity']['hits5'] * 100, 2), round(detail_results['entity']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_uent',
            round(detail_results['uent_ent']['mrr'] * 100, 2), round(detail_results['uent_ent']['hits1'] * 100, 2),
            round(detail_results['uent_ent']['hits5'] * 100, 2), round(detail_results['uent_ent']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_urel',
            round(detail_results['urel_ent']['mrr'] * 100, 2), round(detail_results['urel_ent']['hits1'] * 100, 2),
            round(detail_results['urel_ent']['hits5'] * 100, 2), round(detail_results['urel_ent']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_uboth',
            round(detail_results['uboth_ent']['mrr'] * 100, 2), round(detail_results['uboth_ent']['hits1'] * 100, 2),
            round(detail_results['uboth_ent']['hits5'] * 100, 2), round(detail_results['uboth_ent']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel',
            round(detail_results['relation']['mrr'] * 100, 2), round(detail_results['relation']['hits1'] * 100, 2),
            round(detail_results['relation']['hits5'] * 100, 2), round(detail_results['relation']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_uent',
            round(detail_results['uent_rel']['mrr'] * 100, 2), round(detail_results['uent_rel']['hits1'] * 100, 2),
            round(detail_results['uent_rel']['hits5'] * 100, 2), round(detail_results['uent_rel']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_urel',
            round(detail_results['urel_rel']['mrr'] * 100, 2), round(detail_results['urel_rel']['hits1'] * 100, 2),
            round(detail_results['urel_rel']['hits5'] * 100, 2), round(detail_results['urel_rel']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_uboth',
            round(detail_results['uboth_rel']['mrr'] * 100, 2), round(detail_results['uboth_rel']['hits1'] * 100, 2),
            round(detail_results['uboth_rel']['hits5'] * 100, 2), round(detail_results['uboth_rel']['hits10'] * 100, 2)))

        self.write_rst_csv({'all': detail_results['total']}, 'all_query')
        self.write_rst_csv({'Ent': detail_results['entity']}, 'all_query')
        self.write_rst_csv({'Rel': detail_results['relation']}, 'all_query')

        for k, v in detail_results.items():
            self.write_rst_csv({'all': v}, k)
    
    
    def binary_baseline_train(self):
        best_step = 0
        self.logger.info('start training') 
        self.stop_epoch = 0
        self.best_valid = 0
        self.logger.info('Training')
        self.args.valid_metrics = 'mrr'
        for i in range(1, self.args.num_step + 1):
            if self.args.scorer_func in ['maker', 'pmpi']:
                loss, ent_emb, rel_emb = self.binary_train_one_step()
            else:
                loss, ent_emb, rel_emb = self.train_one_epoch()
 
            if i % self.log_per_step == 0:
                self.logger.info('Training step: {} | loss: {:.4f}'.format(i, loss))
                self.write_training_loss(loss, i)

            if i % self.check_per_step == 0 and i != 1:
                detail_results = self.process_epoch_eval()
                
                self.args.valid = False
                
                if self.best_valid < detail_results['total'][self.args.valid_metrics]:
                    self.best_valid = detail_results['total'][self.args.valid_metrics]
                    self.stop_epoch = 0
                    self.logger.info('Training best model | mrr {:.4f}'.format(self.best_valid))
                    self.save_checkpoint(i, self.get_curr_state())
                    best_step = i
                else:
                    self.stop_epoch += 1
                    self.logger.info('Training best model | mrr {:.4f} at {}'.format(self.best_valid, best_step))
                    
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Total',
                    round(detail_results['total']['mrr'] * 100, 2), round(detail_results['total']['hits1'] * 100, 2),
                    round(detail_results['total']['hits5'] * 100, 2), round(detail_results['total']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent',
                    round(detail_results['entity']['mrr'] * 100, 2), round(detail_results['entity']['hits1'] * 100, 2),
                    round(detail_results['entity']['hits5'] * 100, 2), round(detail_results['entity']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_uent',
                    round(detail_results['uent_ent']['mrr'] * 100, 2), round(detail_results['uent_ent']['hits1'] * 100, 2),
                    round(detail_results['uent_ent']['hits5'] * 100, 2), round(detail_results['uent_ent']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_urel',
                    round(detail_results['urel_ent']['mrr'] * 100, 2), round(detail_results['urel_ent']['hits1'] * 100, 2),
                    round(detail_results['urel_ent']['hits5'] * 100, 2), round(detail_results['urel_ent']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Ent_uboth',
                    round(detail_results['uboth_ent']['mrr'] * 100, 2), round(detail_results['uboth_ent']['hits1'] * 100, 2),
                    round(detail_results['uboth_ent']['hits5'] * 100, 2), round(detail_results['uboth_ent']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel',
                    round(detail_results['relation']['mrr'] * 100, 2), round(detail_results['relation']['hits1'] * 100, 2),
                    round(detail_results['relation']['hits5'] * 100, 2), round(detail_results['relation']['hits10'] * 100, 2)))
                
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_uent',
                    round(detail_results['uent_rel']['mrr'] * 100, 2), round(detail_results['uent_rel']['hits1'] * 100, 2),
                    round(detail_results['uent_rel']['hits5'] * 100, 2), round(detail_results['uent_rel']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_urel',
                    round(detail_results['urel_rel']['mrr'] * 100, 2), round(detail_results['urel_rel']['hits1'] * 100, 2),
                    round(detail_results['urel_rel']['hits5'] * 100, 2), round(detail_results['urel_rel']['hits10'] * 100, 2)))
                self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                    'Rel_uboth',
                    round(detail_results['uboth_rel']['mrr'] * 100, 2), round(detail_results['uboth_rel']['hits1'] * 100, 2),
                    round(detail_results['uboth_rel']['hits5'] * 100, 2), round(detail_results['uboth_rel']['hits10'] * 100, 2)))
                
            if self.stop_epoch >= self.args.early_stop_patience:
                self.logger.info('Early Stopping! Epoch: {} Best Results: {}'.format(i, round(self.best_valid*100, 3)))
                break

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.before_test_load()
        detail_results = self.process_epoch_eval(istest=True)

        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Total',
            round(detail_results['total']['mrr'] * 100, 2), round(detail_results['total']['hits1'] * 100, 2),
            round(detail_results['total']['hits5'] * 100, 2), round(detail_results['total']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent',
            round(detail_results['entity']['mrr'] * 100, 2), round(detail_results['entity']['hits1'] * 100, 2),
            round(detail_results['entity']['hits5'] * 100, 2), round(detail_results['entity']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_uent',
            round(detail_results['uent_ent']['mrr'] * 100, 2), round(detail_results['uent_ent']['hits1'] * 100, 2),
            round(detail_results['uent_ent']['hits5'] * 100, 2), round(detail_results['uent_ent']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_urel',
            round(detail_results['urel_ent']['mrr'] * 100, 2), round(detail_results['urel_ent']['hits1'] * 100, 2),
            round(detail_results['urel_ent']['hits5'] * 100, 2), round(detail_results['urel_ent']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Ent_uboth',
            round(detail_results['uboth_ent']['mrr'] * 100, 2), round(detail_results['uboth_ent']['hits1'] * 100, 2),
            round(detail_results['uboth_ent']['hits5'] * 100, 2), round(detail_results['uboth_ent']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel',
            round(detail_results['relation']['mrr'] * 100, 2), round(detail_results['relation']['hits1'] * 100, 2),
            round(detail_results['relation']['hits5'] * 100, 2), round(detail_results['relation']['hits10'] * 100, 2)))
        
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_uent',
            round(detail_results['uent_rel']['mrr'] * 100, 2), round(detail_results['uent_rel']['hits1'] * 100, 2),
            round(detail_results['uent_rel']['hits5'] * 100, 2), round(detail_results['uent_rel']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_urel',
            round(detail_results['urel_rel']['mrr'] * 100, 2), round(detail_results['urel_rel']['hits1'] * 100, 2),
            round(detail_results['urel_rel']['hits5'] * 100, 2), round(detail_results['urel_rel']['hits10'] * 100, 2)))
        self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
            'Rel_uboth',
            round(detail_results['uboth_rel']['mrr'] * 100, 2), round(detail_results['uboth_rel']['hits1'] * 100, 2),
            round(detail_results['uboth_rel']['hits5'] * 100, 2), round(detail_results['uboth_rel']['hits10'] * 100, 2)))

        self.write_rst_csv({'all': detail_results['total']}, 'all_query')
        self.write_rst_csv({'Ent': detail_results['entity']}, 'all_query')
        self.write_rst_csv({'Rel': detail_results['relation']}, 'all_query')

        for k, v in detail_results.items():
            self.write_rst_csv({'all': v}, k)


    def train(self):
        best_step = 0
        best_eval_rst = {'mrr': 0, 'hits1': 0, 'hits5': 0, 'hits10': 0}
        bad_count = 0
        self.logger.info('start training')
        if self.args.pretraining == 'True':
            if os.path.exists('./data/' + self.args.dataset + '/ent_feat_'+str(self.args.ent_dim)+'.pth'):
                ent_feat = torch.load('./data/' + self.args.dataset + '/ent_feat_'+str(self.args.ent_dim)+'.pth')
                rel_feat = torch.load('./data/' + self.args.dataset + '/rel_feat_'+str(self.args.rel_dim)+'.pth')
                pattern_rel_feat = torch.load('./data/' + self.args.dataset + '/pattern_rel_feat_'+str(self.args.num_rel_bases)+'.pth')
                base2rel_feat = torch.load('./data/' + self.args.dataset + '/base2rel_feat_'+str(self.args.num_rel_bases)+'.pth')
                self.model.ent_feat.data.copy_(ent_feat)
                self.model.rel_feat.data.copy_(rel_feat)
                self.model.pattern_rel_feat.data.copy_(pattern_rel_feat)
                self.model.base2rel_feat.data.copy_(base2rel_feat)
            else:
                self.logger.info('Pretraining')
                for i in range(1, self.args.pre_train_epoch + 1):
                    loss  = self.pretrain_one_epoch()
                    if i % self.log_per_step == 0:
                        self.logger.info('Pretraining step: {} | loss: {:.4f}'.format(i, loss.item()))
                        self.write_training_loss(loss.item(), i)

                    if i % self.check_per_step == 0 and i != 1:
                        eval_rst, ent_emb, rel_emb = self.evaluate()
                        self.write_evaluation_result(eval_rst, i)

                        if eval_rst['mrr'] > best_eval_rst['mrr']:
                            best_eval_rst = eval_rst
                            best_step = i
                            self.logger.info('Pretraining best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                            self.logger.info('Start saveing')
                            self.save_checkpoint(i, self.get_curr_state())
                            self.logger.info('Finsh saveing')
                            bad_count = 0
                        else:
                            bad_count += 1
                            self.logger.info('Pretraining best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                                best_step, best_eval_rst['mrr'], bad_count))

                    if bad_count >= self.early_stop_patience:
                        self.logger.info('Pretraining early stop at step {}'.format(i))
                        break
                torch.save(self.model.ent_feat, './data/' + self.args.dataset + '/ent_feat_'+str(self.args.ent_dim)+'.pth')
                torch.save(self.model.rel_feat, './data/' + self.args.dataset + '/rel_feat_'+str(self.args.rel_dim)+'.pth')
                torch.save(self.model.pattern_rel_feat, './data/' + self.args.dataset + '/pattern_rel_feat_'+str(self.args.num_rel_bases)+'.pth')
                torch.save(self.model.base2rel_feat, './data/' + self.args.dataset + '/base2rel_feat_'+str(self.args.num_rel_bases)+'.pth')
            
        best_eval_rst['mrr'] = 0
        self.logger.info('Training')
        for i in range(1, self.args.num_step + 1):
            if self.args.metalearning == 'True':
                loss, ent_emb, rel_emb = self.train_one_step()
            else:
                loss, ent_emb, rel_emb = self.train_one_epoch()

            if i % self.log_per_step == 0:
                self.logger.info('Training step: {} | loss: {:.4f}'.format(i, loss.item()))
                self.write_training_loss(loss.item(), i)

            if i % self.check_per_step == 0 and i != 1:
                eval_rst, ent_emb, rel_emb = self.evaluate()
                self.write_evaluation_result(eval_rst, i)

                if eval_rst['mrr'] > best_eval_rst['mrr']:
                    best_eval_rst = eval_rst
                    best_step = i
                    self.logger.info('Training best model | mrr {:.4f}'.format(best_eval_rst['mrr']))
                    self.save_checkpoint(i, self.get_curr_state())
                    bad_count = 0
                                
                    if self.args.save_emb == 'True':
                        torch.save(ent_emb, './data/' + self.args.dataset + '/ent_feat_dev_'+self.args.metalearning[0]+self.args.adjustment[0]+self.args.initent[0]+self.args.initrel[0]+self.args.gnnent[0]+'.pth')
                        torch.save(rel_emb, './data/' + self.args.dataset + '/rel_feat_dev_'+self.args.metalearning[0]+self.args.adjustment[0]+self.args.initent[0]+self.args.initrel[0]+self.args.gnnent[0]+'.pth')
                else:
                    bad_count += 1
                    self.logger.info('Training best model is at step {0}, mrr {1:.4f}, bad count {2}'.format(
                        best_step, best_eval_rst['mrr'], bad_count))

            if bad_count >= self.early_stop_patience:
                self.logger.info('Training early stop at step {}'.format(i))
                break

        self.logger.info('finish training')
        self.logger.info('save best model')
        self.save_model(best_step)

        self.logger.info('best validation | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(best_eval_rst['mrr'], best_eval_rst['hits1'], best_eval_rst['hits5'], best_eval_rst['hits10']))

        self.before_test_load()
        rst_all, rst_all_dict, ent_emb, rel_emb = self.evaluate(istest=True)
        if self.args.save_emb == 'True':
            torch.save(ent_emb, './data/' + self.args.dataset + '/ent_feat_'+self.args.metalearning[0]+self.args.adjustment[0]+self.args.initent[0]+self.args.initrel[0]+self.args.gnnent[0]+'.pth')
            torch.save(rel_emb, './data/' + self.args.dataset + '/rel_feat_'+self.args.metalearning[0]+self.args.adjustment[0]+self.args.initent[0]+self.args.initrel[0]+self.args.gnnent[0]+'.pth')
        self.write_rst_csv({'all': rst_all}, 'all_query')

        for k, v in rst_all_dict.items():
            self.write_rst_csv({'all': v}, k)


    def evaluate(self, istest=False):
        if not istest:
            ent_emb, rel_emb = self.model(self.valid_data.g, self.valid_data.pattern_g, self.valid_data.train_ents, self.valid_data.train_rels)
            
            if self.args.dataset in ['FJF17K']:
                ent_eval_dataloader = FewTestDataset(self.args, self.valid_data, pred_type='ent')
            elif self.args.dataset in ['FI_WD20K100_v1']:
                ent_eval_dataloader = TestDataset(self.args, self.valid_data, pred_type='ent')
                # ent_eval_dataloader = InTestDataset(self.args, self.valid_data, pred_type='ent')
            else:
                ent_eval_dataloader = TestDataset(self.args, self.valid_data, pred_type='ent')
                
            ent_eval_dataloader.reset()
            ent_results, ent_count, ent_emb_save, rel_emb_save = self.get_rank(ent_eval_dataloader, ent_emb, rel_emb)
            for k, v in ent_results.items():
                ent_results[k] = v / ent_count
            
            # 虽然执行了，但是在predict函数汇总会continue
            if self.args.dataset in ['FJF17K']:
                rel_eval_dataloader = FewTestDataset(self.args, self.valid_data, pred_type='rel')
            elif self.args.dataset in ['FI_WD20K100_v1']:
                rel_eval_dataloader = TestDataset(self.args, self.valid_data, pred_type='rel')
                # rel_eval_dataloader = InTestDataset(self.args, self.valid_data, pred_type='rel')
            else:
                rel_eval_dataloader = TestDataset(self.args, self.valid_data, pred_type='rel')
            
            rel_eval_dataloader.reset()
            rel_results, rel_count, ent_emb_save, rel_emb_save = self.get_rank(rel_eval_dataloader, ent_emb, rel_emb)
            for k, v in rel_results.items():
                rel_results[k] = v / rel_count

            total_results = ddict()
            for k in list(set(list(ent_results.keys())+list(rel_results.keys()))):
                total_results[k] = (ent_results[k]*ent_count + rel_results[k]*rel_count) / (ent_count + rel_count)

            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Ent',
                ent_results['mrr'], ent_results['hits1'],
                ent_results['hits5'], ent_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Rel',
                rel_results['mrr'], rel_results['hits1'],
                rel_results['hits5'], rel_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Total',
                total_results['mrr'], total_results['hits1'],
                total_results['hits5'], total_results['hits10']))

            return total_results, ent_emb_save, rel_emb_save
        else:
            ent_emb, rel_emb = self.model(self.test_data.g, self.test_data.pattern_g, self.test_data.train_ents, self.test_data.train_rels)

            if self.args.dataset in ['FJF17K']:
                ent_uent_dataloader = FewTestDataset(self.args, self.test_data, type='uent', pred_type='ent')
                ent_uent_dataloader.reset()
                ent_urel_dataloader = FewTestDataset(self.args, self.test_data, type='urel', pred_type='ent')
                ent_urel_dataloader.reset()
                ent_uboth_dataloader = FewTestDataset(self.args, self.test_data, type='uboth', pred_type='ent')
                ent_uboth_dataloader.reset()
            elif self.args.dataset in ['FI_WD20K100_v1']:
                ent_uent_dataloader = InTestDataset(self.args, self.test_data, type='uent', pred_type='ent')
                ent_uent_dataloader.reset()
                ent_urel_dataloader = InTestDataset(self.args, self.test_data, type='urel', pred_type='ent')
                ent_urel_dataloader.reset()
                ent_uboth_dataloader = InTestDataset(self.args, self.test_data, type='uboth', pred_type='ent')
                ent_uboth_dataloader.reset()
            else:
                ent_uent_dataloader = TestDataset(self.args, self.test_data, type='uent', pred_type='ent')
                ent_uent_dataloader.reset()
                ent_urel_dataloader = TestDataset(self.args, self.test_data, type='urel', pred_type='ent')
                ent_urel_dataloader.reset()
                ent_uboth_dataloader = TestDataset(self.args, self.test_data, type='uboth', pred_type='ent')
                ent_uboth_dataloader.reset()
            

            ent_uent_results, ent_uent_count, _, _ = self.get_rank(ent_uent_dataloader, ent_emb, rel_emb)
            ent_urel_results, ent_urel_count, _, _ = self.get_rank(ent_urel_dataloader, ent_emb, rel_emb)
            ent_uboth_results, ent_uboth_count, ent_emb_save, rel_emb_save = self.get_rank(ent_uboth_dataloader, ent_emb, rel_emb)

            ent_results = ddict()
            for k in list(set(list(ent_urel_results.keys())+list(ent_uent_results.keys())+list(ent_uboth_results.keys()))):
                ent_results[k] = (ent_uent_results[k] + ent_urel_results[k] + ent_uboth_results[k]) / (ent_uent_count + ent_urel_count + ent_uboth_count)

            for k, v in ent_uent_results.items():
                if ent_uent_count == 0:
                    ent_uent_results[k] = 0
                else:
                    ent_uent_results[k] = v / ent_uent_count

            for k, v in ent_urel_results.items():
                if ent_urel_count == 0:
                    ent_urel_results[k] = 0
                else:
                    ent_urel_results[k] = v / ent_urel_count

            for k, v in ent_uboth_results.items():
                if ent_uboth_count == 0:
                    ent_uboth_results[k] = 0
                else:
                    ent_uboth_results[k] = v / ent_uboth_count
                

            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Ent',
                ent_results['mrr'], ent_results['hits1'],
                ent_results['hits5'], ent_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Ent_uent',
                ent_uent_results['mrr'], ent_uent_results['hits1'],
                ent_uent_results['hits5'], ent_uent_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Ent_urel',
                ent_urel_results['mrr'], ent_urel_results['hits1'],
                ent_urel_results['hits5'], ent_urel_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Ent_uboth',
                ent_uboth_results['mrr'], ent_uboth_results['hits1'],
                ent_uboth_results['hits5'], ent_uboth_results['hits10']))

            if self.args.dataset in ['FJF17K']:
                rel_uent_dataloader = FewTestDataset(self.args, self.test_data, type='uent', pred_type='rel')
                rel_uent_dataloader.reset()
                rel_urel_dataloader = FewTestDataset(self.args, self.test_data, type='urel', pred_type='rel')
                rel_urel_dataloader.reset()
                rel_uboth_dataloader = FewTestDataset(self.args, self.test_data, type='uboth', pred_type='rel')
                rel_uboth_dataloader.reset()
            elif self.args.dataset in ['FI_WD20K100_v1']:
                rel_uent_dataloader = InTestDataset(self.args, self.test_data, type='uent', pred_type='rel')
                rel_uent_dataloader.reset()
                rel_urel_dataloader = InTestDataset(self.args, self.test_data, type='urel', pred_type='rel')
                rel_urel_dataloader.reset()
                rel_uboth_dataloader = InTestDataset(self.args, self.test_data, type='uboth', pred_type='rel')
                rel_uboth_dataloader.reset()
            else:
                rel_uent_dataloader = TestDataset(self.args, self.test_data, type='uent', pred_type='rel')
                rel_uent_dataloader.reset()
                rel_urel_dataloader = TestDataset(self.args, self.test_data, type='urel', pred_type='rel')
                rel_urel_dataloader.reset()
                rel_uboth_dataloader = TestDataset(self.args, self.test_data, type='uboth', pred_type='rel')
                rel_uboth_dataloader.reset()
                
            rel_uent_results, rel_uent_count, _, _ = self.get_rank(rel_uent_dataloader, ent_emb, rel_emb)
            rel_urel_results, rel_urel_count, _, _ = self.get_rank(rel_urel_dataloader, ent_emb, rel_emb)
            rel_uboth_results, rel_uboth_count, ent_emb_save, rel_emb_save = self.get_rank(rel_uboth_dataloader, ent_emb, rel_emb)

            rel_results = ddict()
            for k in list(set(list(rel_urel_results.keys())+list(rel_uent_results.keys())+list(rel_uboth_results.keys()))):
                rel_results[k] = (rel_uent_results[k] + rel_urel_results[k] + rel_uboth_results[k]) / (rel_uent_count + rel_urel_count + rel_uboth_count)

            for k, v in rel_uent_results.items():
                if rel_uent_count != 0:
                    rel_uent_results[k] = v / rel_uent_count
                else:
                    rel_uent_results[k] = 0

            for k, v in rel_urel_results.items():
                if rel_urel_count != 0:
                    rel_urel_results[k] = v / rel_urel_count
                else:
                    rel_urel_results[k] = 0

            for k, v in rel_uboth_results.items():
                if rel_uboth_count !=0:
                    rel_uboth_results[k] = v / rel_uboth_count
                else:
                    rel_uboth_results[k] = 0
                
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Rel',
                rel_results['mrr'], rel_results['hits1'],
                rel_results['hits5'], rel_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Rel_uent',
                rel_uent_results['mrr'], rel_uent_results['hits1'],
                rel_uent_results['hits5'], rel_uent_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Rel_urel',
                rel_urel_results['mrr'], rel_urel_results['hits1'],
                rel_urel_results['hits5'], rel_urel_results['hits10']))
            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Rel_uboth',
                rel_uboth_results['mrr'], rel_uboth_results['hits1'],
                rel_uboth_results['hits5'], rel_uboth_results['hits10']))

            total_results = ddict()
            for k in list(set(list(ent_urel_results.keys())+list(ent_uent_results.keys())+list(ent_uboth_results.keys()))):
                total_results[k] = (ent_results[k]*(ent_uent_count + ent_urel_count + ent_uboth_count) + rel_results[k]*(rel_uent_count + rel_urel_count + rel_uboth_count)) / ((ent_uent_count + ent_urel_count + ent_uboth_count)+(rel_uent_count + rel_urel_count + rel_uboth_count))

            self.logger.info('{} | mrr: {:.4f}, hits1: {:.4f}, hits5: {:.4f}, hits10: {:.4f}'.format(
                'Total',
                total_results['mrr'], total_results['hits1'],
                total_results['hits5'], total_results['hits10']))

            if self.args.dataset in ['FI_WD20K100_v1', 'FJF17K']:
                return ent_results, {'uent': ent_uent_results, 'urel': ent_urel_results, 'uboth': ent_uboth_results}, ent_emb_save, rel_emb_save
            else:
                return total_results, {'uent': ent_uent_results, 'urel': ent_urel_results, 'uboth': ent_uboth_results}, ent_emb_save, rel_emb_save


    def get_rank(self, eval_dataloader, ent_emb, rel_emb):
        results = ddict(float)
        count = 0
        for _ in range(self.args.eval_times):
            for batch in eval_dataloader:
                if self.args.adjustment == 'True':
                    fact, pred_indexs, label, truth, mask_inputs, mask_outputs, edge_labels, sup_pred_facts, sup_pred_indexs, sup_mask_outputs, sup_mask_inputs, sup_query_types, sup_mask_labels = batch
                    sup_pred_facts = sup_pred_facts.to(self.args.gpu).transpose(1,0)
                    sup_pred_indexs = sup_pred_indexs.to(self.args.gpu)
                    # 哪些candidates有用，包含 True、False
                    sup_mask_outputs = sup_mask_outputs.to(self.args.gpu)
                    sup_mask_inputs = sup_mask_inputs.to(self.args.gpu)
                    edge_labels = edge_labels.to(self.args.gpu)
                    
                    ent_emb.retain_grad()
                    rel_emb.retain_grad()
                    result = self.get_transformer_scorer(sup_pred_facts, sup_pred_indexs, sup_mask_inputs, sup_mask_outputs, edge_labels, ent_emb, rel_emb)
                    self.optimizer.zero_grad()
                    # query_type为1，表示预测的是实体
                    entities, relations = (sup_query_types == 1), (sup_query_types == -1)

                    # 负样例
                    sup_label_entity = sup_mask_outputs[entities] * (self.args.entity_soft / (eval_dataloader.num_ent - 1))
                    if self.args.pred_truth == 'True':
                        ent_sup_mask_labels = []
                        for _, ele in enumerate(sup_query_types):
                            if ele == 1:
                                ent_sup_mask_labels.append(sup_mask_labels[_])
                        for _ in range(sup_label_entity.shape[0]):
                            sup_label_entity[_, ent_sup_mask_labels[_]] = 1 - self.args.entity_soft
                    else:
                        sup_label_entity[torch.arange(sup_label_entity.shape[0]), sup_mask_labels[entities]] = 1 - self.args.entity_soft
                    
                    sup_label_relation = sup_mask_outputs[relations] * (self.args.relation_soft / (eval_dataloader.num_rel - 1))
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
                    ent_emb_q = ent_emb - self.args.ent_beta*ent_grad_meta
                    rel_emb_q = rel_emb - self.args.rel_beta*rel_grad_meta                
                    
                    if self.args.adjustment_type != 'all':
                        ent_emb_q[eval_dataloader.train_ents] = ent_emb[eval_dataloader.train_ents]
                        rel_emb_q[eval_dataloader.train_rels] = rel_emb[eval_dataloader.train_rels]
                else:
                    fact, pred_indexs, label, truth, mask_inputs, mask_outputs, edge_labels = batch
                    ent_emb_q = ent_emb
                    rel_emb_q = rel_emb
                
                if eval_dataloader.pred_type == 'ent' and pred_indexs % 2 == 0:
                    continue
                if eval_dataloader.pred_type == 'rel' and pred_indexs % 2 == 1:
                    continue
                if fact == 0:
                    continue
                eles = [torch.LongTensor(ele).to(self.args.gpu) for ele in fact]
                label = label.to(self.args.gpu)
                mask_inputs = mask_inputs.to(self.args.gpu)
                mask_outputs = mask_outputs.to(self.args.gpu)
                edge_labels = edge_labels.to(self.args.gpu)
                '''link prediction'''
                pred = self.get_transformer_scorer(eles, pred_indexs, mask_inputs, mask_outputs, edge_labels, ent_emb_q, rel_emb_q)
                
                if pred_indexs % 2 == 0:
                    pred = pred[:,:eval_dataloader.num_rel]
                    candidate_num = eval_dataloader.num_rel
                else:
                    pred = pred[:,eval_dataloader.num_rel:]
                    candidate_num = eval_dataloader.num_ent
                
                if self.args.num_sample_cand != 0:
                    if self.args.scorer_func in ['FJF17K', 'FWD50K', 'FWikiPeople']:
                        new_score = []
                        for _ in range(len(pred)):
                            # type限定的潜在candidate
                            type_candidates = eval_dataloader.rel2canidates[eles[2][_]]
                            # 只保留错误的candidate
                            type_candidates_ = [truth[_]]
                            for cand in type_candidates:
                                if label[_][cand] == 0:
                                    type_candidates_.append(cand)
                            
                            tmp_score = pred[_][eval_candidates]
                            new_score.append(tmp_score.unsqueeze(0))
                            # 真值就在第一个位置
                            truth[_] = 0
                        pred = torch.cat(new_score, dim=0)
                    else:
                        new_score = []
                        for _ in range(len(pred)):
                            total_candidates = range(candidate_num)
                            
                            indices = (label[_] == 0).nonzero(as_tuple=True)[0]
                            total_negatives = [total_candidates[i] for i in indices]
                            
                            eval_candidates = random.sample(total_negatives, self.args.num_sample_cand)
                            eval_candidates[0] = truth[_]
                            tmp_score = pred[_][eval_candidates]
                            new_score.append(tmp_score.unsqueeze(0))
                            truth[_] = 0
                        pred = torch.cat(new_score, dim=0)
                    b_range = torch.arange(pred.size()[0]).to(self.args.gpu)
                else:
                    b_range = torch.arange(pred.size()[0]).to(self.args.gpu)
                    target_pred = pred[b_range, truth]
                    pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                    pred[b_range, truth] = target_pred

                '''rank all candidate entities'''
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, truth]
                '''get results'''
                ranks = ranks.float()

                count += torch.numel(ranks)
                results['mr'] += torch.sum(ranks).item()
                results['mrr'] += torch.sum(1.0 / ranks).item()

                for k in [1, 5, 10]:
                    results['hits{}'.format(k)] += torch.numel(ranks[ranks <= k])
        if count == 0:
            return results, count, 0, 0
        else:
            return results, count, ent_emb_q, rel_emb_q
    
    # 先测试Transformer loss
    def get_transformer_scorer(self, pred_facts, pred_indexs, mask_inputs, mask_outputs, edge_labels, ent_emb, rel_emb):
        r_embedding_h = torch.index_select(rel_emb, 0, pred_facts[0])
        h_embedding = torch.index_select(ent_emb, 0, pred_facts[1])
        r_embedding = torch.index_select(rel_emb, 0, pred_facts[2])
        t_embedding = torch.index_select(ent_emb, 0, pred_facts[3])
        embeddings = [r_embedding_h, h_embedding, r_embedding, t_embedding]
        for _ in range(len(pred_facts)//2-2):
            embeddings.append(torch.index_select(rel_emb, 0, pred_facts[4+2*_]))
            embeddings.append(torch.index_select(ent_emb, 0, pred_facts[4+2*_+1]))

        embeddings = [self.input_dropout(self.input_norm(ele)) for ele in embeddings]
        
        x = torch.stack(embeddings, 0).transpose(1,0)
        edge_query = self.edge_query_embedding(edge_labels)
        edge_key = self.edge_key_embedding(edge_labels)
        edge_value = self.edge_value_embedding(edge_labels)
        
        for layer in self.layers:
            x = layer(x, mask_inputs, edge_key, edge_value, edge_query)
        x = x[torch.arange(x.shape[0]), pred_indexs]
        x = self.output_linear(x)  # x(batch_size, hiddem_dim)
        x = self.output_act(x)
        x = self.output_norm(x)
        embedding = torch.cat((rel_emb, ent_emb), 0)
        
        y = torch.mm(x, embedding.transpose(0, 1))# + self.output_bias
        # 选择entity或者relation
        y = y.masked_fill(mask_outputs == 0, -100000)
        return y