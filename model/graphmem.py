import torch
import numpy as np
import ipdb
import time
from torch.autograd import Variable
import torch.nn as nn
from util import use_cuda, sparse_bmm, read_padded

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class GraphMem(nn.Module):
    def __init__(self, config):
        """
        num_relation: number of relation including self-connection
        """
        super(GraphMem, self).__init__()
        self.num_layer = config['num_layer']
        self.num_relation = config['num_relation']
        self.num_entity = config['num_entity']
        self.num_word = config['num_word']
        self.entity_dim = config['entity_dim']
        self.word_dim = config['word_dim']
        self.has_entity_kge = False
        self.has_relation_kge = False

        # initialize entity embedding
        if config['pretrained_entity_kge_file'] is not None:
            self.has_entity_kge = True
            self.entity_kge = nn.Embedding(num_embeddings=config['num_entity']+1,
                                           embedding_dim=config['kge_dim'], padding_idx=config['num_entity'])
            self.entity_kge.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_entity_kge_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            # self.entity_kge.weight.requires_grad = False

        self.entity_linear = nn.Linear(in_features= config['kge_dim'], out_features=config['entity_dim'])
        self.relation_embedding = nn.Embedding(num_embeddings=config['num_relation']+1, embedding_dim=config['word_dim'],
                                               padding_idx=config['num_relation'])
        if config['pretrained_relation_emb_file'] is not None:
            if config['use_inverse_relation']:
                tmp_arr = np.load(config['pretrained_relation_emb_file'])
                self.relation_embedding.weight = nn.Parameter(
                    torch.from_numpy(
                        np.pad(np.concatenate((tmp_arr, tmp_arr), axis=0), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
            else:
                self.relation_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(config['pretrained_relation_emb_file']), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
        if config['pretrained_relation_kge_file'] is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=config['num_relation']+1, embedding_dim=config['kge_dim'],
                                             padding_idx=config['num_relation'])
            if config['use_inverse_relation']:
                self.relation_kge.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(config['pretrained_relation_kge_file']), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))

        if self.has_relation_kge:
            # self.relation_linear = nn.Linear(in_features=2 * config['word_dim'] + config['kge_dim'], out_features=config['entity_dim'])
            self.relation_linear = nn.Linear(in_features=config['word_dim'] + config['kge_dim'], out_features=config['entity_dim'])
        else:
            # self.relation_linear = nn.Linear(in_features=2 * config['word_dim'], out_features=config['entity_dim'])
            self.relation_linear = nn.Linear(in_features=config['word_dim'], out_features=config['entity_dim'])

        # initialize text embeddings
        self.word_embedding = nn.Embedding(num_embeddings=config['num_word'] + 1, embedding_dim=config['word_dim'], padding_idx=config['num_word'])
        if config['pretrained_word_embedding_file'] is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(np.pad(np.load(config['pretrained_word_embedding_file']), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

        # create LSTMs
        self.node_encoder = nn.LSTM(input_size=config['word_dim'], hidden_size=config['entity_dim'], batch_first=True, bidirectional=True)
        self.mask_embd = nn.Parameter(torch.rand(config['entity_dim']))

        self.decoder = nn.GRUCell(config['entity_dim'], config['entity_dim'])

        # dropout
        self.lstm_drop = nn.Dropout(p=config['lstm_dropout'])
        self.linear_drop = nn.Dropout(p=config['linear_dropout'])
        # non-linear activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.elu = nn.ELU()
        # scoring function
        self.score_func = nn.Linear(in_features=config['entity_dim'], out_features=1)
        self.diag = nn.Parameter(torch.ones(config['entity_dim']))

        # loss
        self.log_softmax = nn.LogSoftmax(dim=1)  # pylint: disable=E1123
        self.softmax_d1 = nn.Softmax(dim=1)

        self.bce_loss = nn.BCELoss()
        self.bce_loss_logits = nn.BCEWithLogitsLoss()
        self.lstm_linear = nn.Linear(in_features=2*self.entity_dim, out_features=self.entity_dim)
        self.big_linear = nn.Linear(in_features=3*self.entity_dim, out_features=self.entity_dim)

        self.weight_prev = nn.Parameter(torch.ones(1, 1))
        self.weight_new = nn.Parameter(torch.ones(1, 1))
        self.query_rel_score_weight = nn.Parameter(torch.ones(1, 1))
        self.state_rel_score_weight = nn.Parameter(torch.ones(1, 1))

        self.linears = nn.ModuleList([nn.Linear(self.entity_dim, self.entity_dim) for _ in range(self.num_layer)])
        self.trans_linears = nn.ModuleList([nn.Linear(self.entity_dim, self.entity_dim) for _ in range(self.num_layer)])

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
        local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_text, answer_dist = batch

        (fact2entity_mat, fact2entity_rev_mat, entity2fact_mat, entity2entity_mat, relation2fact_mat, fact2relation_mat) = kb_adj_mat
        batch_size, max_local_entity = local_entity.shape
        _, max_fact = kb_fact_rel.shape
        num_relation = self.num_relation

        # numpy to tensor
        local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        local_entity_mask = use_cuda((local_entity != self.num_entity).type('torch.FloatTensor'))
        kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor'), requires_grad=False))
        kb_fact_rel_mask = use_cuda((kb_fact_rel != self.num_relation).type('torch.FloatTensor'))

        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != self.num_word).type('torch.FloatTensor'))

        max_query_word = query_text.shape[1]

        answer_dist = use_cuda(Variable(torch.from_numpy(answer_dist).type('torch.FloatTensor'), requires_grad=False))

        # encode query
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        # query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.entity_dim)) # 1, batch_size, entity_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                                  self.init_hidden(2, batch_size,
                                                                                   self.entity_dim))  # 1, batch_size, entity_dim

        query_hidden_emb = (query_hidden_emb[:, :, :self.entity_dim] + query_hidden_emb[:, :, self.entity_dim:]) / 2

        query_node_emb = (query_node_emb[0] + query_node_emb[1]) * 0.5

        # load entity embedding
        local_entity_emb = self.entity_kge(local_entity)
        local_entity_emb = self.entity_linear(local_entity_emb)

        local_rel_emb = self.relation_embedding.weight
        if self.has_relation_kge:
            local_rel_emb = torch.cat((local_rel_emb, self.relation_kge.weight),
                                      dim=1)  # num_rel, 2 * word_dim + kge_dim
        # batch_size, num_rel, entity_dim
        local_rel_emb = self.relation_linear(local_rel_emb).unsqueeze(0).repeat(batch_size, 1, 1)

        # [batch_size, max_local_entity]
        start_entities = use_cuda(
            Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=False)).squeeze(dim=2)
        end_entities = None
        traverse_entity_record = start_entities
        entity_scores = use_cuda(Variable(torch.FloatTensor(local_entity.shape).zero_()))
        kge_entity_scores = use_cuda(Variable(torch.FloatTensor(local_entity.shape).zero_()))

        entity_state = torch.bmm(start_entities.unsqueeze(1), local_entity_emb).squeeze()

        # compute scores between query and relations
        # [batch_size, num_rel, max_query_word]
        # rel2query_score = torch.bmm(local_rel_emb, query_hidden_emb.transpose(1, 2)) * (1 / np.sqrt(self.entity_dim)) + \
        #                   (1.0 - query_mask.unsqueeze(1)) * (-1e10)
        # rel2query_attn_score = torch.softmax(rel2query_score, dim=2)
        # attn_rel_query = torch.bmm(rel2query_attn_score, query_hidden_emb)

        # [batch size, dim]
        hidden_state = query_node_emb
        decoder_input = torch.zeros_like(hidden_state)
        aggr_rel_history = None
        for i in range(self.num_layer):
            hidden_state = self.decoder(decoder_input, hidden_state)
            query_rel_score = torch.bmm(local_rel_emb, hidden_state.unsqueeze(-1)).squeeze()
            q2r_attn = torch.softmax(query_rel_score * (1 / np.sqrt(self.entity_dim)), dim=1)
            aggr_rel = torch.bmm(q2r_attn.unsqueeze(1), local_rel_emb).squeeze()

            if aggr_rel_history is None:
                aggr_rel_history = aggr_rel
            else:
                aggr_rel_history = aggr_rel_history * aggr_rel

            decoder_input = aggr_rel

        latent_rel = aggr_rel_history * self.state_rel_score_weight + query_node_emb * self.query_rel_score_weight

        kge_entity_scores = torch.bmm((entity_state * latent_rel).unsqueeze(1), local_entity_emb.transpose(1, 2)).squeeze()

        for i in range(self.num_layer):
            # find ent entities
            end_entities = torch.bmm(entity2entity_mat, start_entities.unsqueeze(2)).squeeze() + traverse_entity_record
            end_entities[end_entities > 0] = 1
            end_entities = end_entities - traverse_entity_record

            # update
            start_entities = end_entities
            traverse_entity_record = traverse_entity_record + end_entities
            end_entities = None

        entity_scores = kge_entity_scores

        # calculate loss and make prediction
        loss = self.bce_loss_logits(entity_scores, answer_dist)

        score = entity_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER + (1 - traverse_entity_record) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.max(score, dim=1)[1]

        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))


