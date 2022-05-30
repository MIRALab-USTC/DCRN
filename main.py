import os
import sys
import ipdb
import time
import torch
import pickle
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data_loader import DataLoader, KvMemDataLoader, GraphMemDataLoader
from metaqa_data_loader import MetaQaGraphMemDataLoader
from model.graftnet import GraftNet
from model.kvmem import KvMem
from model.graphmem import GraphMem
from model.simple_graphmem import SimpleGraphMem
from util import use_cuda, get_config, load_dict, cal_accuracy, cal_accuracy_list, cal_rank, cal_hits, load_triplet, MetaQaAdjMat
from util import output_pred_dist, read_dict
import warnings
warnings.filterwarnings("ignore")

def train(cfg):
    save_model_path = os.path.dirname(cfg['save_model_file'])
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)

    # tensorboard
    writer = SummaryWriter("tensorboard/"+cfg['name'])

    print("training ...")
    # prepare data
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    dataset = cfg['dataset']
    data_dir = cfg['data_folder']
    cache_dir = os.path.join(data_dir, 'cache', cfg['model'])

    cache_train_dir = os.path.join(cache_dir, 'train')
    cache_valid_dir = os.path.join(cache_dir, 'valid')
    cache_test_dir = os.path.join(cache_dir, 'test')

    if dataset == 'webqsp' and os.path.exists(cache_dir) and os.path.isdir(cache_dir):
    # if False:
        print('Load data from cache')
        with open(os.path.join(cache_dir, 'train_loader.pkl'), 'rb') as f:
            train_data = pickle.load(f)

        with open(os.path.join(cache_dir, 'valid_loader.pkl'), 'rb') as f:
            valid_data = pickle.load(f)

        with open(os.path.join(cache_dir, 'test_loader.pkl'), 'rb') as f:
            test_data = pickle.load(f)
        # if 'metaqa' in dataset:
        #     train_data.load(cache_train_dir)
        #     valid_data.load(cache_valid_dir)
        #     test_data.load(cache_test_dir)

    else:
        model_type = cfg['model']

        if dataset == 'webqsp':
            if model_type == 'graftnet':
                data_loader = DataLoader
            elif model_type == 'kvmem':
                data_loader = KvMemDataLoader
            elif model_type == 'graphmem' or model_type == 'simple_graphmem':
                data_loader = GraphMemDataLoader
            else:
                raise ValueError("Invalid model type: {}".format(model_type))

            train_data = data_loader(cfg['data_folder'] + cfg['train_data'], word2id, relation2id, entity2id,
                                     cfg['max_query_word'], cfg['use_inverse_relation'])
            valid_data = data_loader(cfg['data_folder'] + cfg['dev_data'], word2id, relation2id, entity2id,
                                     cfg['max_query_word'], cfg['use_inverse_relation'])
            test_data = data_loader(cfg['data_folder'] + cfg['test_data'], word2id, relation2id, entity2id,
                                    cfg['max_query_word'], cfg['use_inverse_relation'])

            os.mkdir(cache_dir)


        elif 'metaqa' in dataset:
            if model_type == 'graphmem':
                triplets = load_triplet(cfg['data_folder'] + cfg['kb_data'])
                data_loader = MetaQaGraphMemDataLoader
            else:
                raise ValueError("Invalid model type: {}".format(model_type))
            if '1hop' in dataset:
                subgraph_file = os.path.join(cfg['data_folder'], '1_hop_subgraph.pkl')
            elif '2hop' in dataset:
                subgraph_file = os.path.join(cfg['data_folder'], '2_hop_subgraph.pkl')
            elif '3hop' in dataset:
                subgraph_file = os.path.join(cfg['data_folder'], '3_hop_subgraph.pkl')

            print("loading pickled adj data ...")
            start_time = time.time()
            # adj_data_file = os.path.join(cfg['data_folder'], 'data.pkl')
            adj_data_file = os.path.join(cfg['data_folder'], 'data_global.pkl')
            with open(adj_data_file, 'rb') as f:
                subgraph_file = pickle.load(f)
            end_time = time.time()
            print("load time: {}".format(end_time - start_time))

            train_batch_size = cfg['batch_size']
            valid_batch_size = cfg['test_batch_size']
            test_batch_size = cfg['test_batch_size']

            adj_data = MetaQaAdjMat(subgraph_file, train_batch_size, len(entity2id), 2 * len(relation2id))
            batch_adj_data = adj_data.batch_adj_data

            train_data = data_loader(cfg['data_folder'] + cfg['train_data'], subgraph_file, word2id, relation2id, entity2id,
                                     cfg['max_query_word'], cfg['use_inverse_relation'], train_batch_size, batch_adj_data)
            valid_data = data_loader(cfg['data_folder'] + cfg['dev_data'], subgraph_file, word2id, relation2id, entity2id,
                                     cfg['max_query_word'], cfg['use_inverse_relation'], valid_batch_size, batch_adj_data)
            test_data = data_loader(cfg['data_folder'] + cfg['test_data'], subgraph_file, word2id, relation2id, entity2id,
                                    cfg['max_query_word'], cfg['use_inverse_relation'], test_batch_size, batch_adj_data)

            # each dataloader build their own adj data
            # train_data = data_loader(cfg['data_folder'] + cfg['train_data'], subgraph_file, word2id, relation2id, entity2id,
            #                          cfg['max_query_word'], cfg['use_inverse_relation'], train_batch_size)
            # valid_data = data_loader(cfg['data_folder'] + cfg['dev_data'], subgraph_file, word2id, relation2id, entity2id,
            #                          cfg['max_query_word'], cfg['use_inverse_relation'], valid_batch_size)
            # test_data = data_loader(cfg['data_folder'] + cfg['test_data'], subgraph_file, word2id, relation2id, entity2id,
            #                         cfg['max_query_word'], cfg['use_inverse_relation'], test_batch_size)


    # create model & set parameters
    my_model = get_model(cfg, train_data.num_kb_relation, len(entity2id), len(word2id))
    trainable_parameters = [p for p in my_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg['learning_rate'])

    best_dev_acc = 0.0
    best_test_acc = 0.0
    global_iter = 0
    for epoch in range(cfg['num_epoch']):
        try:
            print('epoch', epoch)
            train_data.reset_batches(is_sequential = cfg['is_debug'])
            # Train
            my_model.train()
            train_loss, train_acc, train_max_acc = [], [], []
            for iteration in tqdm(range(train_data.num_data // cfg['batch_size'])):
                # start = time.time()
                batch = train_data.get_batch(iteration, cfg['batch_size'], cfg['fact_dropout'])
                # end = time.time()
                # print("Get batch time: {}".format(end - start))
                # start = time.time()
                loss, pred, _ = my_model(batch)
                pred = pred.data.cpu().numpy()
                acc, max_acc = cal_accuracy(pred, batch[-1])
                train_loss.append(loss.cpu().data)
                train_acc.append(acc)
                train_max_acc.append(max_acc)

                writer.add_scalar('Loss/loss', loss, global_iter)
                writer.add_scalar('Evaluation/Train Accuracy', acc, global_iter)
                writer.add_scalar('Evaluation/Train Max Accuracy', max_acc, global_iter)
                writer.add_scalar('Parameter/Prev Weight', my_model.weight_prev, global_iter)
                writer.add_scalar('Parameter/New Weight', my_model.weight_new, global_iter)

                writer.add_scalar('Parameter/query_rel_score_weight', my_model.query_rel_score_weight, global_iter)
                writer.add_scalar('Parameter/state_rel_score_weight', my_model.state_rel_score_weight, global_iter)

                # writer.add_scalar('Parameter/Rel Weight', my_model.weight_rel, global_iter)
                # writer.add_scalar('Parameter/KGE Weight', my_model.weight_kge, global_iter)

                # back propagate
                my_model.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(my_model.parameters(), cfg['gradient_clip'])
                # torch.nn.utils.clip_grad_value_(my_model.parameters(), cfg['gradient_clip'])

                optimizer.step()
                global_iter += 1

                # my_model.regularize(max_norm=10)

                # end = time.time()
                # print("Train time: {}".format(end - start))

            print("weight prev: {}".format(my_model.weight_prev))
            print("weight new: {}".format(my_model.weight_new))

            print('avg_training_loss', sum(train_loss) / len(train_loss))
            print('max_training_acc', sum(train_max_acc) / len(train_max_acc))
            print('avg_training_acc', sum(train_acc) / len(train_acc))

            print("validating ...")
            eval_acc = inference(my_model, valid_data, entity2id, cfg)
            print("testing ...")
            test_acc = inference(my_model, test_data, entity2id, cfg)

            if eval_acc > best_dev_acc:
                if cfg['to_save_model']:
                    print("saving model to", cfg['save_model_file'])
                    torch.save(my_model.state_dict(), cfg['save_model_file'])
                best_dev_acc = eval_acc
                best_test_acc = test_acc

            writer.add_scalar('Evaluation/Eval Accuracy', eval_acc, global_iter)
            writer.add_scalar('Evaluation/Test Accuracy', test_acc, global_iter)

            print("Best Evaluation Accuracy: {}, Best Test Accuracy: {}".format(best_dev_acc, best_test_acc))
        except KeyboardInterrupt:
            break

    # Test set evaluation
    print("evaluating on test")
    print('loading model from ...', cfg['save_model_file'])
    my_model.load_state_dict(torch.load(cfg['save_model_file']))
    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)

    return test_acc

"""vanilla inference"""
def inference(my_model, valid_data, entity2id, cfg, log_info=False):
    # Evaluation
    my_model.eval()
    eval_loss, eval_acc, eval_max_acc = [], [], []
    id2entity = {idx: entity for entity, idx in entity2id.items()}
    valid_data.reset_batches(is_sequential = True)
    test_batch_size = cfg['test_batch_size']

    # fb_name_dict_filepath = os.path.join(cfg['data_folder'], cfg['freebase_id_map_file'])
    # fb_name_dict = read_dict(fb_name_dict_filepath)

    # if log_info:
    #     f_pred = open(cfg['pred_file'], 'w')
    for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
        # valid data
        batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
        loss, pred, pred_dist = my_model(batch)
        pred = pred.data.cpu().numpy()
        acc, max_acc = cal_accuracy(pred, batch[-1])
        # if log_info:
        #     output_pred_dist(pred_dist, batch[-1], id2entity, iteration * test_batch_size, valid_data, f_pred, fb_name_dict)
        eval_loss.append(loss.data)
        eval_acc.append(acc)
        eval_max_acc.append(max_acc)

    print('avg_loss', sum(eval_loss) / len(eval_loss))
    print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
    print('avg_acc', sum(eval_acc) / len(eval_acc))

    return sum(eval_acc) / len(eval_acc)

"""inference for rank"""
# def inference(my_model, valid_data, entity2id, cfg, log_info=False):
#     # Evaluation
#     my_model.eval()
#     eval_loss, eval_acc, eval_max_acc = [], [], []
#     id2entity = {idx: entity for entity, idx in entity2id.items()}
#     valid_data.reset_batches(is_sequential = True)
#     test_batch_size = cfg['test_batch_size']
#
#     # fb_name_dict_filepath = os.path.join(cfg['data_folder'], cfg['freebase_id_map_file'])
#     # fb_name_dict = read_dict(fb_name_dict_filepath)
#
#     total_ranks = []
#     for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
#         # valid data
#         batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
#         loss, pred, pred_dist = my_model(batch)
#         pred = pred.data.cpu().numpy()
#         pred_dist = pred_dist.data.cpu().numpy()
#         ranks = cal_rank(pred_dist, batch[-1])
#         total_ranks.extend(ranks)
#
#     with open(cfg['rank_file'], 'w') as f:
#         for rank in total_ranks:
#             f.write("{}\n".format(rank))

"""inference for hits"""
# def inference(my_model, valid_data, entity2id, cfg, log_info=False):
#     # Evaluation
#     my_model.eval()
#     eval_loss, eval_acc, eval_max_acc = [], [], []
#     id2entity = {idx: entity for entity, idx in entity2id.items()}
#     valid_data.reset_batches(is_sequential = True)
#     test_batch_size = cfg['test_batch_size']
#
#     # fb_name_dict_filepath = os.path.join(cfg['data_folder'], cfg['freebase_id_map_file'])
#     # fb_name_dict = read_dict(fb_name_dict_filepath)
#
#     h1, h3, h10 = [], [], []
#     for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
#         # valid data
#         batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
#         loss, pred, pred_dist = my_model(batch)
#         pred = pred.data.cpu().numpy()
#         pred_dist = pred_dist.data.cpu().numpy()
#
#         hits1, hits3, hits10 = cal_hits(pred_dist, batch[-1])
#
#         h1.append(hits1)
#         h3.append(hits3)
#         h10.append(hits10)
#
#     print('avg_h1', sum(h1) / len(h1))
#     print('avg_h3', sum(h3) / len(h3))
#     print('avg_h10', sum(h10) / len(h10))

"""inference for saving scores"""
# def inference(my_model, valid_data, entity2id, cfg, log_info=False):
#     # Evaluation
#     my_model.eval()
#     eval_loss, eval_acc, eval_max_acc = [], [], []
#     id2entity = {idx: entity for entity, idx in entity2id.items()}
#     valid_data.reset_batches(is_sequential = True)
#     test_batch_size = cfg['test_batch_size']
#
#     # fb_name_dict_filepath = os.path.join(cfg['data_folder'], cfg['freebase_id_map_file'])
#     # fb_name_dict = read_dict(fb_name_dict_filepath)
#
#     import numpy as np
#     final_pred_dist = None
#     final_acc_list = []
#     for iteration in tqdm(range(valid_data.num_data // test_batch_size)):
#         # valid data
#         batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0)
#         loss, pred, pred_dist = my_model(batch)
#         pred = pred.data.cpu().numpy()
#         pred_dist = pred_dist.data.cpu().numpy()
#
#         if final_pred_dist is None:
#             final_pred_dist = pred_dist
#         else:
#             final_pred_dist = np.concatenate((final_pred_dist, pred_dist), 0)
#
#         acc, max_acc = cal_accuracy(pred, batch[-1])
#
#         eval_loss.append(loss.data)
#         eval_acc.append(acc)
#         eval_max_acc.append(max_acc)
#
#         acc_list = cal_accuracy_list(pred, batch[-1])
#         final_acc_list.extend(acc_list)
#
#     print("shape: {}".format(final_pred_dist.shape))
#     np.save('/home/jycai/code/multihop_kbqa/analysis/data/metaqa_2hop_test_fine.npy', final_pred_dist)
#
#     with open('/home/jycai/code/multihop_kbqa/analysis/data/metaqa_2hop_test_fine_acc.txt', 'w') as f:
#         for acc in final_acc_list:
#             f.write("{}\n".format(acc))
#
#     print('avg_loss', sum(eval_loss) / len(eval_loss))
#     print('max_acc', sum(eval_max_acc) / len(eval_max_acc))
#     print('avg_acc', sum(eval_acc) / len(eval_acc))
#
#     return sum(eval_acc) / len(eval_acc)

"""vanilla test"""
# def test(cfg):
#     entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
#     word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
#     relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])
#
#     data_dir = cfg['data_folder']
#     cache_dir = os.path.join(data_dir, 'cache', cfg['model'])
#
#     with open(os.path.join(cache_dir, 'test_loader.pkl'), 'rb') as f:
#         test_data = pickle.load(f)
#
#     # test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], word2id, relation2id, entity2id, cfg['max_query_word'], cfg['use_inverse_relation'])
#
#     my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id))
#     test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)
#     return test_acc


def test(cfg):
    entity2id = load_dict(cfg['data_folder'] + cfg['entity2id'])
    word2id = load_dict(cfg['data_folder'] + cfg['word2id'])
    relation2id = load_dict(cfg['data_folder'] + cfg['relation2id'])

    dataset = cfg['dataset']
    data_dir = cfg['data_folder']
    cache_dir = os.path.join(data_dir, 'cache', cfg['model'])

    if dataset == 'webqsp' and os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        with open(os.path.join(cache_dir, 'test_loader.pkl'), 'rb') as f:
            test_data = pickle.load(f)
    elif 'metaqa' in dataset:
        data_loader = MetaQaGraphMemDataLoader

        print("loading pickled adj data ...")
        start_time = time.time()
        # adj_data_file = os.path.join(cfg['data_folder'], 'data.pkl')
        adj_data_file = os.path.join(cfg['data_folder'], 'data_global.pkl')
        with open(adj_data_file, 'rb') as f:
            subgraph_file = pickle.load(f)
        end_time = time.time()
        print("load time: {}".format(end_time - start_time))

        train_batch_size = cfg['batch_size']
        test_batch_size = cfg['test_batch_size']

        adj_data = MetaQaAdjMat(subgraph_file, train_batch_size, len(entity2id), 2 * len(relation2id))
        batch_adj_data = adj_data.batch_adj_data

        test_data = data_loader(cfg['data_folder'] + cfg['test_data'], subgraph_file, word2id, relation2id, entity2id,
                                cfg['max_query_word'], cfg['use_inverse_relation'], test_batch_size, batch_adj_data)


    # test_data = DataLoader(cfg['data_folder'] + cfg['test_data'], word2id, relation2id, entity2id, cfg['max_query_word'], cfg['use_inverse_relation'])

    my_model = get_model(cfg, test_data.num_kb_relation, len(entity2id), len(word2id))
    test_acc = inference(my_model, test_data, entity2id, cfg, log_info=True)
    return test_acc

def get_model(cfg, num_kb_relation, num_entities, num_vocab):
    word_emb_file = None if cfg['word_emb_file'] is None else cfg['data_folder'] + cfg['word_emb_file']
    entity_emb_file = None if cfg['entity_emb_file'] is None else cfg['data_folder'] + cfg['entity_emb_file']
    entity_kge_file = None if cfg['entity_kge_file'] is None else cfg['data_folder'] + cfg['entity_kge_file']
    relation_emb_file = None if cfg['relation_emb_file'] is None else cfg['data_folder'] + cfg['relation_emb_file']
    relation_kge_file = None if cfg['relation_kge_file'] is None else cfg['data_folder'] + cfg['relation_kge_file']

    if cfg['model'] == 'graftnet':
        my_model = use_cuda(
            GraftNet(word_emb_file, entity_emb_file, entity_kge_file, relation_emb_file, relation_kge_file,
                     cfg['num_layer'], num_kb_relation, num_entities, num_vocab, cfg['entity_dim'], cfg['word_dim'],
                     cfg['kge_dim'], cfg['pagerank_lambda'], cfg['fact_scale'], cfg['lstm_dropout'],
                     cfg['linear_dropout'])
        )
    elif cfg['model'] == 'kvmem' or cfg['model'] == 'graphmem' or cfg['model'] == 'simple_graphmem':
        model_config = {}
        model_config['use_inverse_relation'] = cfg['use_inverse_relation']
        model_config['pretrained_word_embedding_file'] = word_emb_file
        model_config['pretrained_entity_emb_file'] = entity_emb_file
        model_config['pretrained_entity_kge_file'] = entity_kge_file
        model_config['pretrained_relation_emb_file'] = relation_emb_file
        model_config['pretrained_relation_kge_file'] = relation_kge_file
        model_config['num_layer'] = cfg['num_layer']
        model_config['num_relation'] = num_kb_relation
        model_config['num_entity'] = num_entities
        model_config['num_word'] = num_vocab
        model_config['entity_dim'] = cfg['entity_dim']
        model_config['word_dim'] = cfg['word_dim']
        model_config['kge_dim'] = cfg['kge_dim']
        model_config['lstm_dropout'] = cfg['lstm_dropout']
        model_config['linear_dropout'] = cfg['linear_dropout']

        if cfg['model'] == 'kvmem':
            my_model = use_cuda(
                KvMem(model_config)
            )
        elif cfg['model'] == 'simple_graphmem':
            my_model = use_cuda(
                SimpleGraphMem(model_config)
            )
        elif cfg['model'] == 'graphmem':
            my_model = use_cuda(
                GraphMem(model_config)
            )

    if cfg['load_model_file'] is not None:
        print('loading model from', cfg['load_model_file'])
        pretrained_model_states = torch.load(cfg['load_model_file'])
        if word_emb_file is not None:
            del pretrained_model_states['word_embedding.weight']
        if entity_emb_file is not None:
            del pretrained_model_states['entity_embedding.weight']
        my_model.load_state_dict(pretrained_model_states, strict=False)
    
    return my_model

if __name__ == "__main__":
    config_file = sys.argv[2]
    CFG = get_config(config_file)
    if '--train' == sys.argv[1]:
        train(CFG)
    elif '--test' == sys.argv[1]:
        test(CFG)
    else:
        assert False, "--train or --test?"

