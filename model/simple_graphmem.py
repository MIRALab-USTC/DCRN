import torch
import numpy as np
import ipdb
from torch.autograd import Variable
import torch.nn as nn
from util import use_cuda, sparse_bmm, read_padded

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class SimpleGraphMem(nn.Module):
    def __init__(self, config):
        """
        num_relation: number of relation including self-connection
        """
        super(SimpleGraphMem, self).__init__()
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
                tmp_arr = np.load(config['pretrained_relation_kge_file'])
                self.relation_kge.weight = nn.Parameter(
                    torch.from_numpy(
                        np.pad(np.concatenate((tmp_arr, tmp_arr), axis=0), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
            else:
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
        # self.node_encoder = nn.LSTM(input_size=config['word_dim'], hidden_size=config['entity_dim'], batch_first=True, bidirectional=False)
        self.node_encoder = nn.LSTM(input_size=config['word_dim'], hidden_size=config['entity_dim'], batch_first=True, bidirectional=True)

        self.mask_embd = nn.Parameter(torch.rand(config['entity_dim']))

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

        for i in range(self.num_layer):
            self.add_module('query_linear' + str(i), nn.Linear(in_features=self.entity_dim, out_features=self.entity_dim))
            self.add_module('kb_ent_linear' + str(i), nn.Linear(in_features=self.entity_dim, out_features=self.entity_dim))
            self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=self.entity_dim, out_features=self.entity_dim))
            self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=self.entity_dim, out_features=self.entity_dim))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=2 * self.entity_dim, out_features=self.entity_dim))

    """new2"""
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

        batch_size, max_local_entity = local_entity.shape
        _, max_fact = kb_fact_rel.shape

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
        query_word_emb = self.word_embedding(query_text) # batch_size, max_query_word, word_dim
        # query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.entity_dim)) # 1, batch_size, entity_dim
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(2, batch_size, self.entity_dim)) # 1, batch_size, entity_dim
        # query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1) # batch_size, 1, entity_dim
        # query_hidden_emb = self.lstm_linear(query_hidden_emb)
        query_hidden_emb = (query_hidden_emb[:,:,:self.entity_dim] + query_hidden_emb[:,:,self.entity_dim:]) / 2
        query_node_emb = query_node_emb.mean(dim=0).unsqueeze(dim=1) # batch_size, 1, entity_dim

        # build kb_adj_matrix from sparse matrix
        (e2f_batch, e2f_f, e2f_e, e2f_val), \
        (e2f_rev_batch, e2f_rev_f, e2f_rev_e, e2f_rev_val), \
        (f2e_batch, f2e_e, f2e_f, f2e_val), \
        (e2e_batch, e2e_h, e2e_t, e2e_val) = kb_adj_mat

        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        entity2fact_val = torch.FloatTensor(e2f_val)
        # [batch_size, max_fact, max_local_entity]
        fact2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size([batch_size, max_fact, max_local_entity]))) # batch_size, max_fact, max_local_entity

        entity2fact_rev_index = torch.LongTensor([e2f_rev_batch, e2f_rev_f, e2f_rev_e])
        entity2fact_rev_val = torch.FloatTensor(e2f_rev_val)
        # [batch_size, max_fact, max_local_entity]
        fact2entity_rev_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_rev_index, entity2fact_rev_val, torch.Size([batch_size, max_fact, max_local_entity]))) # batch_size, max_fact, max_local_entity


        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        fact2entity_val = torch.FloatTensor(f2e_val)
        # [batch_size, max_local_entity, max_fact]
        entity2fact_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val, torch.Size([batch_size, max_local_entity, max_fact])))

        entity2entity_index = torch.LongTensor([e2e_batch, e2e_h, e2e_t])
        entity2entity_val = torch.FloatTensor(e2e_val)
        # [batch_size, max_fact, max_local_entity]
        entity2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2entity_index,
                                                              entity2entity_val,
                                                              torch.Size([batch_size, max_local_entity, max_local_entity]))) # batch_size, max_fact, max_local_entity

        # load entity embedding
        local_entity_emb = self.entity_embedding(local_entity) # batch_size, max_local_entity, word_dim
        if self.has_entity_kge:
            local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)), dim=2) # batch_size, max_local_entity, word_dim + kge_dim
            local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim

        # load relation embedding for each fact
        local_fact_emb = self.relation_embedding(kb_fact_rel) # batch_size, max_fact, 2 * word_dim
        if self.has_relation_kge:
            local_fact_emb = torch.cat((local_fact_emb, self.relation_kge(kb_fact_rel)), dim=2) # batch_size, max_fact, 2 * word_dim + kge_dim
        local_fact_emb = self.relation_linear(local_fact_emb) # batch_size, max_fact, entity_dim

        local_rel_emb = self.relation_embedding.weight
        if self.has_relation_kge:
            local_rel_emb = torch.cat((local_rel_emb, self.relation_kge.weight), dim=1)  # num_rel, 2 * word_dim + kge_dim
        local_rel_emb = self.relation_linear(local_rel_emb).unsqueeze(0).repeat(batch_size, 1, 1)  # batch_size, num_rel, entity_dim

        # [batch_size, max_local_entity]
        start_entities = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=False)).squeeze(dim=2)
        end_entities = None
        traverse_entity_record = start_entities
        entity_scores = use_cuda(Variable(torch.FloatTensor(local_entity.shape).zero_()))

        entity_state = torch.bmm(start_entities.unsqueeze(1), local_entity_emb).squeeze()

        num_neighbor = torch.sparse.sum(entity2entity_mat, 1).to_dense()
        num_neighbor[num_neighbor == 0] = 1e10

        for i in range(self.num_layer):
            # init_entity_scores = torch.bmm(query_hidden_emb, local_entity_emb.transpose(1, 2)).sum(1) * local_entity_mask / query_mask.sum(1).unsqueeze(1)
            init_entity_scores = torch.bmm(query_hidden_emb, local_entity_emb.transpose(1, 2)).sum(1) * local_entity_mask / query_mask.sum(1).unsqueeze(1)

            query_rel_score = torch.bmm(query_node_emb, local_rel_emb.transpose(1, 2)).squeeze()
            state_rel_score = torch.bmm(entity_state.unsqueeze(1), local_rel_emb.transpose(1, 2)).squeeze()

            # [batch_size, rel]
            q2r_attn = torch.softmax((query_rel_score + state_rel_score) * (1 / np.sqrt(self.entity_dim)), dim=1)
            aggr_rel = torch.bmm(q2r_attn.unsqueeze(1), local_rel_emb).squeeze()

            entity_state = entity_state + aggr_rel
            query_node_emb = query_node_emb - aggr_rel.unsqueeze(1)

            # find ent entities
            end_entities = sparse_bmm(entity2entity_mat, start_entities.unsqueeze(2)).squeeze() + traverse_entity_record
            end_entities[end_entities > 0] = 1
            end_entities = end_entities - traverse_entity_record

            # find facts to traverse
            # bmm([batch_size, 1, max_local_entity], [batch_size, max_local_entity, max_fact]) -> [batch_size, 1, max_fact]
            # with transpose calculation
            traverse_facts = sparse_bmm(fact2entity_mat, start_entities.unsqueeze(2)).squeeze()
            # facts ends with traverse_facts
            end_facts = sparse_bmm(fact2entity_rev_mat, end_entities.unsqueeze(2)).squeeze()
            traverse_facts = traverse_facts * end_facts

            # compute scores for each pair of (fact's relation, word in question)
            # [batch_size, max_fact, max_query_word]
            fact_rel_word_scores = torch.bmm(local_fact_emb, query_hidden_emb.transpose(1, 2))
            # masking
            fact_rel_word_scores = fact_rel_word_scores * query_mask.unsqueeze(1) * (kb_fact_rel_mask * traverse_facts).unsqueeze(2)

            # # compute query word - relation attention
            fact_rel_word_attn_scores = fact_rel_word_scores + (-1e10) * (1.0 - (kb_fact_rel_mask * traverse_facts).unsqueeze(2)).repeat(1, 1,
                                                                                                                 max_query_word)
            fact_rel_word_attn_scores = fact_rel_word_attn_scores + (-1e10) * (1.0 - query_mask.unsqueeze(1)).repeat(1, max_fact, 1)
            fact_rel_word_attn_scores = fact_rel_word_attn_scores * (1 / np.sqrt(local_fact_emb.shape[-1]))

            # [batch_size, max_fact, max_query_word]
            fact2word_attn = torch.softmax(fact_rel_word_attn_scores, dim=2)
            # [batch_size, max_fact, entity_dim]
            weighted_fact = torch.sum(fact2word_attn.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=1), dim=2)
            fact_rel_scores = (weighted_fact * local_fact_emb).sum(2) * (kb_fact_rel_mask * traverse_facts)

            # [batch_size, max_fact, max_query_word]
            word2fact_attn = torch.softmax(fact_rel_word_attn_scores, dim=1)
            # [batch_size, max_query_word, entity_dim]
            weighted_word = torch.sum(word2fact_attn.unsqueeze(dim=3) * local_fact_emb.unsqueeze(dim=2), dim=1)
            word_scores = (weighted_word * query_hidden_emb).sum(2) * query_mask


            # fact -> tail entities
            # bmm([batch_size, 1, max_fact], [batch_size, max_fact, max_local_entity])
            # -> [batch_size, 1, max_fact]
            # end_entity_scores_new = sparse_bmm(entity2fact_mat, fact_rel_scores.unsqueeze(2)).squeeze()
            end_entity_scores_new = sparse_bmm(entity2fact_mat, fact_rel_scores.unsqueeze(2)).squeeze()


            end_entity_scores_prev = sparse_bmm(entity2entity_mat, entity_scores.unsqueeze(2)).squeeze()
            end_entity_scores_prev = end_entity_scores_prev * (1.0 / num_neighbor)

            end_entity_scores_local = init_entity_scores

            # masking
            SCALE = 1.0 if i == 0 else 0.5
            end_entity_scores = ((end_entity_scores_new + end_entity_scores_prev) * SCALE + end_entity_scores_local) * end_entities
            # entity_scores = entity_scores + end_entity_scores
            entity_scores = entity_scores + end_entity_scores + torch.bmm(entity_state.unsqueeze(1), local_entity_emb.transpose(1, 2)).squeeze()


            # # [batch_size, max_query_word, max_ent]
            # word2ent_attn = torch.bmm(query_hidden_emb, local_entity_emb.transpose(1, 2))
            # word2ent_attn = word2ent_attn + (-1e10) * (1.0 - query_mask).unsqueeze(-1) + (-1e10) * (1.0 - end_entities).unsqueeze(1)
            # word2ent_attn = torch.softmax(word2ent_attn * (1/np.sqrt(self.entity_dim)), dim=2)
            #
            # query_hidden_emb = query_hidden_emb - torch.bmm(word2ent_attn, local_entity_emb)

            # sigmoid for each word
            soft_mask = torch.sigmoid(fact_rel_word_scores.sum(1))
            query_hidden_emb = (1.0 - soft_mask).unsqueeze(-1) * query_hidden_emb
            query_hidden_emb = (1.0 - soft_mask).unsqueeze(-1) * query_hidden_emb + \
                               soft_mask.unsqueeze(-1) * self.mask_embd.view(1, 1, self.entity_dim)

            # update
            start_entities = end_entities
            traverse_entity_record = traverse_entity_record + end_entities
            end_entities = None

        # entity_scores = entity_scores + torch.bmm(query_node_emb, local_entity_emb.transpose(1, 2)).squeeze() * local_entity_mask
        # calculate loss and make prediction
        loss = self.bce_loss_logits(entity_scores, answer_dist)

        score = entity_scores + (1 - local_entity_mask) * VERY_NEG_NUMBER
        pred_dist = self.sigmoid(score) * local_entity_mask
        pred = torch.max(score, dim=1)[1]

        return loss, pred, pred_dist

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))),
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))


