import yaml
import torch
import os
import re
import json
import pickle
import nltk
import ipdb
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm


class MetaQaAdjMat(object):
    def __init__(self, subgraph_file, batch_size, max_local_entity, num_kb_relation):
        (triplet_ids, kb_adj_mats, kb_fact_rels) = subgraph_file
        self.batch_size = batch_size
        self.max_facts = len(triplet_ids) * 2
        self.max_local_entity = max_local_entity
        self.num_kb_relation = num_kb_relation

        self.kb_adj_mats = kb_adj_mats

        print("building adj mat ...")
        kb_adj_data = self._build_kb_adj_mat()
        print("building batch adj mat ...")
        self.batch_adj_data = self._build_batch_kb_adj_mat(kb_adj_data)

    def _build_kb_adj_mat(self):
        """Create sparse matrix representation for batched data"""
        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats0_rev_batch = np.array([], dtype=int)
        mats0_rev_0 = np.array([], dtype=int)
        mats0_rev_1 = np.array([], dtype=int)
        vals0_rev = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        mats2_batch = np.array([], dtype=int)
        mats2_0 = np.array([], dtype=int)
        mats2_1 = np.array([], dtype=int)
        vals2 = np.array([], dtype=float)

        mats3_batch = np.array([], dtype=int)
        mats3_0 = np.array([], dtype=int)
        mats3_1 = np.array([], dtype=int)
        vals3 = np.array([], dtype=float)

        mats4_batch = np.array([], dtype=int)
        mats4_0 = np.array([], dtype=int)
        mats4_1 = np.array([], dtype=int)
        vals4 = np.array([], dtype=float)

        for i, sample_id in enumerate(tqdm(range(self.batch_size))):

            (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
            (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
            (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats
            # assert len(val0) == len(val1)
            num_fact = len(val0)
            num_ent = len(val2)
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0)
            mats0_1 = np.append(mats0_1, mat0_1)
            vals0 = np.append(vals0, val0)

            # mat0_rev
            mats0_rev_batch = np.append(mats0_rev_batch, np.full(num_fact, i, dtype=int))
            mats0_rev_0 = np.append(mats0_rev_0, mat0_rev_0)
            mats0_rev_1 = np.append(mats0_rev_1, mat0_rev_1)
            vals0_rev = np.append(vals0_rev, val0_rev)


            # mat1
            mats1_batch = np.append(mats1_batch, np.full(len(val1), i, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0)
            mats1_1 = np.append(mats1_1, mat1_1)
            vals1 = np.append(vals1, val1)

            # mat2
            mats2_batch = np.append(mats2_batch, np.full(num_ent, i, dtype=int))
            mats2_0 = np.append(mats2_0, mat2_0)
            mats2_1 = np.append(mats2_1, mat2_1)
            vals2 = np.append(vals2, val2)

            # mat3
            mats3_batch = np.append(mats3_batch, np.full(num_fact, i, dtype=int))
            mats3_0 = np.append(mats3_0, mat3_0)
            mats3_1 = np.append(mats3_1, mat3_1)
            vals3 = np.append(vals3, val3)

            # mat4
            mats4_batch = np.append(mats4_batch, np.full(num_fact, i, dtype=int))
            mats4_0 = np.append(mats4_0, mat4_0)
            mats4_1 = np.append(mats4_1, mat4_1)
            vals4 = np.append(vals4, val4)

        return (mats0_batch, mats0_0, mats0_1, vals0), \
               (mats0_rev_batch, mats0_rev_0, mats0_rev_1, vals0_rev), \
               (mats1_batch, mats1_0, mats1_1, vals1), \
               (mats2_batch, mats2_0, mats2_1, vals2), \
               (mats3_batch, mats3_0, mats3_1, vals3), \
               (mats4_batch, mats4_0, mats4_1, vals4)

    def _build_batch_kb_adj_mat(self, kb_adj_data):
        batch_size = self.batch_size
        max_fact = self.max_facts
        max_local_entity = self.max_local_entity
        num_relation = self.num_kb_relation

        # build kb_adj_matrix from sparse matrix
        (e2f_batch, e2f_f, e2f_e, e2f_val), \
        (e2f_rev_batch, e2f_rev_f, e2f_rev_e, e2f_rev_val), \
        (f2e_batch, f2e_e, f2e_f, f2e_val), \
        (e2e_batch, e2e_h, e2e_t, e2e_val), \
        (r2f_batch, r2f_f, r2f_r, r2f_val), \
        (f2r_batch, f2r_r, f2r_f, f2r_val), \
            = kb_adj_data

        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        entity2fact_val = torch.FloatTensor(e2f_val)
        # [batch_size, max_fact, max_local_entity]
        fact2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
            [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity

        entity2fact_rev_index = torch.LongTensor([e2f_rev_batch, e2f_rev_f, e2f_rev_e])
        entity2fact_rev_val = torch.FloatTensor(e2f_rev_val)
        # [batch_size, max_fact, max_local_entity]
        fact2entity_rev_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_rev_index, entity2fact_rev_val, torch.Size(
            [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity

        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        fact2entity_val = torch.FloatTensor(f2e_val)
        # [batch_size, max_local_entity, max_fact]
        entity2fact_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val,
                                                            torch.Size([batch_size, max_local_entity, max_fact])))

        entity2entity_index = torch.LongTensor([e2e_batch, e2e_h, e2e_t])
        entity2entity_val = torch.FloatTensor(e2e_val)
        # [batch_size, max_fact, max_local_entity]
        entity2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2entity_index,
                                                              entity2entity_val,
                                                              torch.Size([batch_size, max_local_entity,
                                                                          max_local_entity])))  # batch_size, max_fact, max_local_entity

        relation2fact_index = torch.LongTensor([r2f_batch, r2f_f, r2f_r])
        relation2fact_val = torch.FloatTensor(r2f_val)
        # [batch_size, num_relation, max_fact]
        relation2fact_mat = use_cuda(torch.sparse.FloatTensor(relation2fact_index, relation2fact_val, torch.Size(
            [batch_size, num_relation + 1, max_fact])))

        fact2relation_index = torch.LongTensor([f2r_batch, f2r_r, f2r_f])
        fact2relation_val = torch.FloatTensor(f2r_val)
        # [batch_size, max_local_entity, max_fact]
        fact2relation_mat = use_cuda(torch.sparse.FloatTensor(fact2relation_index, fact2relation_val,
                                                              torch.Size([batch_size, max_fact, num_relation + 1])))

        return (fact2entity_mat, fact2entity_rev_mat, entity2fact_mat, entity2entity_mat, relation2fact_mat, fact2relation_mat)

def read_dict(filename):
    fb_name_dict = {}
    with open(filename, 'r') as f:
        for line in f.readlines():
            sample_list = line.strip().split('\t')
            fb_id, fb_name = sample_list[0], sample_list[1]
            fb_name_dict[fb_id] = fb_name

    return fb_name_dict


def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config

def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def save_model(the_model, path):
    if os.path.exists(path):
        path = path + '_copy'
    print("saving model to ...", path)
    torch.save(the_model, path)


def load_model(path):
    if not os.path.exists(path):
        assert False, 'cannot find model: ' + path
    print("loading model from ...", path)
    return torch.load(path)

def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def load_triplet(filename):
    triplets = []
    with open(filename) as f:
        for line in f:
            if '|' in line:
                head, rel, tail = line.strip().split('|')
            else:
                head, rel, tail = line.strip().split('\t')
            triplets.append((head, rel, tail))
    return triplets

def load_documents(document_file):
    print('loading document from', document_file)
    documents = dict()
    with open(document_file) as f_in:
        for line in tqdm(f_in):
            passage = json.loads(line)
            # tokenize document
            document_token = nltk.word_tokenize(passage['document']['text'])
            if 'title' in passage:
                title_token = nltk.word_tokenize(passage['title']['text'])
                passage['tokens'] = document_token + ['|'] + title_token
            else:
                passage['tokens'] = document_token
            documents[int(passage['documentId'])] = passage
    return documents

def index_document_entities(documents, word2id, entity2id, max_document_word):
    print('indexing documents ...')
    document_entity_indices = dict()
    document_texts = dict()
    document_texts[-1] = np.full(max_document_word, len(word2id), dtype=int)
    for next_id, document in tqdm(documents.items()):
        global_entity_ids = []
        word_ids = []
        word_weights = []
        if 'title' in document:
            for entity in document['title']['entities']:
                entity_len = entity['end'] - entity['start']
                global_entity_ids += [entity2id[entity['text']]] * entity_len
                word_ids += range(entity['start'], entity['end'])
                word_weights += [1.0 / entity_len] * entity_len
            title_len = len(nltk.word_tokenize(document['title']['text']))
        else:
            title_len = 0
        for entity in document['document']['entities']:
            # word_ids are off by (title_len + 1) because document is concatenated after title, and with an extra '|'
            if entity['start'] + title_len + 1 >= max_document_word:
                continue
            entity_len = min(max_document_word, entity['end'] + title_len + 1) - (entity['start'] + title_len + 1)
            global_entity_ids += [entity2id[entity['text']]] * entity_len
            word_ids += range(entity['start'] + title_len + 1, entity['start'] + title_len + 1 + entity_len)
            if entity_len != 0:
                word_weights += [1.0 / entity_len] * entity_len

        assert len(word_weights) == len(word_ids)
        document_entity_indices[next_id] = (global_entity_ids, word_ids, word_weights)
        
        one_doc_text = np.full(max_document_word, len(word2id), dtype=int)
        for t, token in enumerate(document['tokens']):
            if t < max_document_word:
                if token in word2id:
                    one_doc_text[t] = word2id[token]
                else:
                    one_doc_text[t] = word2id['__unk__']
        
        document_texts[next_id] = one_doc_text
    
    return document_entity_indices, document_texts

def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)

def cal_accuracy_list(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    results = []
    for i, l in enumerate(pred):
        results.append(int(answer_dist[i, l] != 0))
    return results

def cal_rank(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    ranks = []
    for i in range(pred.shape[0]):
        answers = answer_dist[i].nonzero()[0].tolist()
        for answer in answers:
            rank = np.sum(pred[i] >= pred[i][answer])
            ranks.append(rank)
    return ranks

def cal_hits(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    h1, h3, h10 = 0, 0, 0
    for i in range(pred.shape[0]):
        answers = answer_dist[i].nonzero()[0].tolist()
        local_h1, local_h3, local_h10 = 0, 0, 0
        for answer in answers:
            rank = np.sum(pred[i] > pred[i][answer]) + 1
            if rank == 1:
                local_h1 = 1
            if rank <= 3:
                local_h3 = 1
            if rank <= 10:
                local_h10 = 1
        h1 += local_h1
        h3 += local_h3
        h10 += local_h10

    h1 = float(h1) / float(pred.shape[0])
    h3 = float(h3) / float(pred.shape[0])
    h10 = float(h10) / float(pred.shape[0])

    return h1, h3, h10


def output_pred_dist(pred_dist, answer_dist, id2entity, start_id, data_loader, f_pred, fb_name_dict):
    for i, p_dist in enumerate(pred_dist):
        data_id = start_id + i
        question = data_loader.data[i]['question']
        l2g = {l:g for g, l in data_loader.global2local_entity_maps[data_id].items()}
        output_dist = {id2entity[l2g[j]]: float(prob) for j, prob in enumerate(p_dist.data.cpu().numpy()) if j < len(l2g)}
        output_dist = {k: v for k, v in sorted(output_dist.items(), key=lambda item: item[1], reverse=True)[:10]}
        answers = [answer['text'] if type(answer['kb_id']) == int else answer['kb_id'] for answer in data_loader.data[data_id]['answers']]
        # f_pred.write(json.dumps({'dist': output_dist,
        #                          'answers':answers,
        #                          'seeds': data_loader.data[data_id]['entities'],
        #                          'tuples': data_loader.data[data_id]['subgraph']['tuples']}) + '\n')
        # f_pred.write(json.dumps({'question': question,
        #                          'answers': answers,
        #                          'seeds': data_loader.data[data_id]['entities'],
        #                          'dist': output_dist
        #                          }, indent=4) + '\n')

        pattern_fb = re.compile(r'<fb:m\.(.*)>')
        # output dist
        output_dist_readable = {}
        for k, v in output_dist.items():
            match_obj_fb = re.match(pattern_fb, k)
            if match_obj_fb:
                fb_id = "kg:/m/" + match_obj_fb.group(1)
                if fb_id not in fb_name_dict.keys():
                    key = k
                else:
                    key = fb_name_dict[fb_id]
            else:
                key = k
            output_dist_readable[key] = v

        # answers
        for j, ans in enumerate(answers):
            match_obj_fb = re.match(pattern_fb, ans)
            if match_obj_fb:
                fb_id = "kg:/m/" + match_obj_fb.group(1)
                if fb_id not in fb_name_dict.keys():
                    continue
                else:
                    answers[j] = fb_name_dict[fb_id]

        # seeds
        seeds = data_loader.data[data_id]['entities']
        for j, seed in enumerate(seeds):
            match_obj_fb = re.match(pattern_fb, seed['kb_id'])
            if match_obj_fb:
                fb_id = "kg:/m/" + match_obj_fb.group(1)
                if fb_id not in fb_name_dict.keys():
                    continue
                else:
                    seeds[j]['text'] = fb_name_dict[fb_id]

        f_pred.write(json.dumps({'question': question,
                                 'answers': answers,
                                 'seeds': seeds,
                                 'dist': output_dist_readable
                                 }, indent=4) + '\n')
        # f_pred.write(json.dumps({'question': question}, indent=4) + '\n')

class LeftMMFixed(torch.autograd.Function):
    """
    Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
    This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
    """

    def __init__(self):
        super(LeftMMFixed, self).__init__()
        self.sparse_weights = None

    def forward(self, sparse_weights, x):
        if self.sparse_weights is None:
            self.sparse_weights = sparse_weights
        return torch.mm(self.sparse_weights, x)

    def backward(self, grad_output):
        sparse_weights = self.sparse_weights
        return None, torch.mm(sparse_weights.t(), grad_output)


def sparse_bmm(X, Y):
    """Batch multiply X and Y where X is sparse, Y is dense.
    Args:
        X: Sparse tensor of size BxMxN. Consists of two tensors,
            I:3xZ indices, and V:1xZ values.
        Y: Dense tensor of size BxNxK.
    Returns:
        batched-matmul(X, Y): BxMxK
    """
    I = X._indices()
    V = X._values()
    B, M, N = X.size()
    _, _, K = Y.size()
    Z = I.size()[1]
    lookup = Y[I[0, :], I[2, :], :]
    X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
    S = use_cuda(Variable(torch.cuda.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
    prod_op = LeftMMFixed()
    prod = prod_op(S, lookup)
    return prod.view(B, M, K)


def read_padded(my_lstm, document_emb, document_mask):
    """
    this function take an embedded array, pack it, read, and pad it.
    in order to use Packed_Sequence, we should sort by length, and then reverse to the original order
    :document_emb: num_document, max_document_word, hidden_size
    :document_mask: num_document, max_document_word
    :my_lstm: lstm
    """
    num_document, max_document_word, _ = document_emb.size()
    hidden_size = my_lstm.hidden_size
    document_lengths = torch.sum(document_mask, dim=1).type('torch.IntTensor') # num_document
    document_lengths, perm_idx = document_lengths.sort(0, descending=True)
    document_emb = document_emb[use_cuda(perm_idx)]
    inverse_perm_idx = [0] * len(perm_idx)
    for i, idx in enumerate(perm_idx):
        inverse_perm_idx[idx.data] = i
    inverse_perm_idx = torch.LongTensor(inverse_perm_idx)

    document_lengths_np = document_lengths.data.cpu().numpy()
    document_lengths_np[document_lengths_np == 0] = 1 # skip warning: length could be 0

    num_layer = 2 if my_lstm.bidirectional else 1
    hidden = (use_cuda(Variable(torch.zeros(num_layer, num_document, hidden_size))), 
                use_cuda(Variable(torch.zeros(num_layer, num_document, hidden_size))))

    document_emb = pack_padded_sequence(document_emb, document_lengths_np, batch_first=True) # padded array
    document_emb, hidden = my_lstm(document_emb, hidden) # [batch_size * max_relevant_doc, max_document_word, entity_dim * 2] [2, batch_size * max_relevant_doc, entity_dim]
    document_emb, _ = pad_packed_sequence(document_emb, batch_first=True)

    document_emb = document_emb[use_cuda(inverse_perm_idx)]
    hidden = (hidden[0][:, use_cuda(inverse_perm_idx), :], hidden[1][:, use_cuda(inverse_perm_idx), :])
    batch_max_document_word = document_emb.size()[1]
    if batch_max_document_word < max_document_word:
        all_zeros = use_cuda(Variable(torch.zeros((num_document, max_document_word, hidden_size * num_layer))))
        all_zeros[:, : batch_max_document_word, :] = document_emb
        document_emb = all_zeros

    assert (num_document, max_document_word, hidden_size * num_layer) == document_emb.size()

    return document_emb, hidden


if __name__  == "__main__":
    load_documents('datasets/wikimovie/full_doc/documents.json')
