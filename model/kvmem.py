import torch
import numpy as np
import ipdb
from torch.autograd import Variable
import torch.nn as nn
from util import use_cuda, sparse_bmm, read_padded

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class KvMem(nn.Module):
    def __init__(self, config):
        """
        num_relation: number of relation including self-connection
        """
        super(KvMem, self).__init__()
        self.num_layer = config['num_layer']
        self.num_relation = config['num_relation']
        self.num_entity = config['num_entity']
        self.num_word = config['num_word']
        self.entity_dim = config['entity_dim']
        self.word_dim = config['word_dim']
        self.has_entity_kge = False
        self.has_relation_kge = False

        # initialize entity embedding
        self.entity_embedding = nn.Embedding(num_embeddings=config['num_entity']+1, embedding_dim=config['word_dim'],
                                             padding_idx=config['num_entity'])
        if config['pretrained_entity_emb_file'] is not None:
            self.entity_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_entity_emb_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.entity_embedding.weight.requires_grad = False
        if config['pretrained_entity_kge_file'] is not None:
            self.has_entity_kge = True
            self.entity_kge = nn.Embedding(num_embeddings=config['num_entity']+1,
                                           embedding_dim=config['kge_dim'], padding_idx=config['num_entity'])
            self.entity_kge.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_entity_kge_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.entity_kge.weight.requires_grad = False

        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=config['word_dim'] + config['kge_dim'], out_features=config['entity_dim'])
        else:
            self.entity_linear = nn.Linear(in_features=config['word_dim'], out_features=config['entity_dim'])

        # initialize relation embedding
        self.relation_embedding = nn.Embedding(num_embeddings=config['num_relation']+1, embedding_dim=2 * config['word_dim'],
                                               padding_idx=config['num_relation'])
        if config['pretrained_relation_emb_file'] is not None:
            self.relation_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_relation_emb_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
        if config['pretrained_relation_kge_file'] is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=config['num_relation']+1, embedding_dim=config['kge_dim'],
                                             padding_idx=config['num_relation'])
            self.relation_kge.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_relation_kge_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))

        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * config['word_dim'] + config['kge_dim'], out_features=config['entity_dim'])
        else:
            self.relation_linear = nn.Linear(in_features=2 * config['word_dim'], out_features=config['entity_dim'])

        # initialize text embeddings
        self.word_embedding = nn.Embedding(num_embeddings=config['num_word'] + 1, embedding_dim=config['word_dim'], padding_idx=config['num_word'])
        if config['pretrained_word_embedding_file'] is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_word_embedding_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

        self.A = nn.Linear(config['entity_dim'], config['entity_dim'])

        self.R = nn.ModuleList([nn.Linear(config['entity_dim'], config['entity_dim']) for i in range(self.num_layer)])

        # create LSTMs
        self.node_encoder = nn.LSTM(input_size=config['word_dim'], hidden_size=config['entity_dim'], batch_first=True, bidirectional=False)

        # dropout
        self.lstm_drop = nn.Dropout(p=config['lstm_dropout'])
        self.linear_drop = nn.Dropout(p=config['linear_dropout'])
        # non-linear activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        # scoring function
        self.score_func = nn.Linear(in_features=config['entity_dim'], out_features=1)
        # loss
        self.log_softmax = nn.LogSoftmax(dim=1)  # pylint: disable=E1123
        self.softmax_d1 = nn.Softmax(dim=1)

        self.bce_loss = nn.BCELoss()
        self.bce_loss_logits = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        """
        :local_entity: global_id of each entity                     (batch_size, max_local_entity)
        :q2e_adj_mat: adjacency matrices (dense)                    (batch_size, max_local_entity, 1)
        :kb_adj_mat: adjacency matrices (sparse)                    (batch_size, max_fact, max_local_entity), (batch_size, max_local_entity, max_fact)
        :kb_fact_rel:                                               (batch_size, max_fact)
        :query_text: a list of words in the query                   (batch_size, max_query_word)
        :document_text:                                             (batch_size, max_relevant_doc, max_document_word)
        :entity_pos: sparse entity_pos_mat                          (batch_size, max_local_entity, max_relevant_doc * max_document_word)
        :answer_dist: an distribution over local_entity             (batch_size, max_local_entity)
        """
        local_entity, head_ids, rel_ids, tail_ids, tripet_mask, query_text, answer_dist = batch
        head_ids = use_cuda(Variable(torch.from_numpy(head_ids).type('torch.LongTensor'), requires_grad=False))
        rel_ids = use_cuda(Variable(torch.from_numpy(rel_ids).type('torch.LongTensor'), requires_grad=False))
        tail_ids = use_cuda(Variable(torch.from_numpy(tail_ids).type('torch.LongTensor'), requires_grad=False))
        tripet_mask = use_cuda(Variable(torch.from_numpy(tripet_mask).type('torch.FloatTensor'), requires_grad=False))

        batch_size, max_local_entity = local_entity.shape

        # numpy to tensor
        local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))

        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))

        answer_dist = use_cuda(Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor'), requires_grad=False))

        # encode question
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(1, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim

        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        # query_node_emb = self.A(query_node_emb).squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim

        # load entity embedding
        local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
        # key-value embedding
        key_emb_head = self.entity_embedding(head_ids)
        key_emb_rel = self.relation_embedding(rel_ids)
        value_emb = self.entity_embedding(tail_ids)

        if self.has_entity_kge:
            local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)), dim=2)
            key_emb_head = torch.cat((key_emb_head, self.entity_kge(head_ids)), dim=2)
            key_emb_rel = torch.cat((key_emb_head, self.relation_kge(rel_ids)), dim=2)
            value_emb = torch.cat((value_emb, self.entity_kge(tail_ids)), dim=2)
        if self.word_dim != self.entity_dim:
            local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim
            key_emb_head = self.entity_linear(key_emb_head)
            key_emb_rel = self.relation_linear(key_emb_rel)
            value_emb = self.entity_linear(value_emb)

        # key_emb: [batch_size, max_facts, dim]
        key_emb = key_emb_head + key_emb_rel
        attn_mask = (1.0 - tripet_mask) * (-1e10)

        # key_emb = self.A(key_emb)
        # value_emb = self.A(value_emb)

        # memory network, multi-hop
        for i in range(self.num_layer):
            # (1). Key Addressing
            attn_scores = torch.bmm(query_node_emb, key_emb.transpose(-1, -2)).squeeze()
            # attn_scores = torch.softmax(attn_scores + attn_mask, dim=-1)
            attn_scores = torch.softmax((attn_scores + attn_mask) * (1 / np.sqrt(self.entity_dim)), dim=-1)

            # (2). Value Reading
            value = (attn_scores.unsqueeze(-1) * value_emb).sum(1).squeeze()

            # (3). Query Update
            query_node_emb = query_node_emb + value.unsqueeze(1)
            # query_node_emb = self.R[i](query_node_emb + value.unsqueeze(1))


        # calculate scores for all candidate entities
        score = torch.bmm(local_entity_emb, query_node_emb.transpose(-1, -2)).squeeze()
        loss = self.bce_loss_logits(score, answer_dist)

        score = score + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.max(score, dim=1)[1]

        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))


