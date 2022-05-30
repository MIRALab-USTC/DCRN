import os
import json
import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from util import load_dict
from tqdm import tqdm
import ipdb

class DataLoader():
    def __init__(self, data_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.count_max_query_word = 0

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)

        print('loading data from', data_file)
        self.data = []
        with open(data_file) as f_in:
            for line in tqdm(f_in.readlines()):
                line = json.loads(line)
                self.data.append(line)
                self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))

        print('max_facts: ', self.max_facts)

        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i:entity for entity, i in entity2id.items()}

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()
        print('max_local_entity', self.max_local_entity)


        print('preparing data ...')
        # local entity id -> global entity id
        self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        
        self._prepare_data()
        print("count max query words: {}".format(self.count_max_query_word))

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50

        for sample in tqdm(self.data):
            # local entity id -> global entity id
            g2l = self.global2local_entity_maps[next_id]
            for global_entity, local_entity in g2l.items():
                if local_entity != 0: # skip question node
                    self.local_entities[next_id, local_entity] = global_entity

            # for building sparse adjacency matrices
            # entity2fact, head entity -> fact
            # fact2entity, fact -> tail entity
            entity2fact_e, entity2fact_f = [], []
            fact2entity_f, fact2entity_e = [], []

            # relations in local KB
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                if not self.use_inverse_relation:
                    entity2fact_e += [g2l[self.entity2id[sbj['text']]]]
                    entity2fact_f += [i]
                    fact2entity_f += [i]
                    fact2entity_e += [g2l[self.entity2id[obj['text']]]]
                    self.kb_fact_rels[next_id, i] = self.relation2id[rel['text']]
                else:
                    entity2fact_e += [g2l[self.entity2id[sbj['text']]], g2l[self.entity2id[obj['text']]]]
                    entity2fact_f += [2 * i, 2 * i + 1]
                    fact2entity_f += [2 * i, 2 * i + 1]
                    fact2entity_e += [g2l[self.entity2id[obj['text']]], g2l[self.entity2id[sbj['text']]]]
                    self.kb_fact_rels[next_id, 2 * i] = self.relation2id[rel['text']]
                    self.kb_fact_rels[next_id, 2 * i + 1] = self.relation2id[rel['text']] + len(self.relation2id)
                    
            # build connection between question and entities in it
            for j, entity in enumerate(sample['entities']):
                self.q2e_adj_mats[next_id, g2l[self.entity2id[str(entity['text'])]], 0] = 1.0

            # tokenize question
            self.count_max_query_word = max(self.count_max_query_word, len(sample['question'].split()))
            count_query_length[len(sample['question'].split())] += 1
            for j, word in enumerate(sample['question'].split()):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else: 
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            # construct distribution for answers
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                if self.entity2id[answer[keyword]] in g2l:
                    self.answer_dists[next_id, g2l[self.entity2id[answer[keyword]]]] = 1.0

            self.kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int),
                                         np.array(entity2fact_e, dtype=int),
                                         np.array([1.0] * len(entity2fact_f))), \
                                        (np.array(fact2entity_e, dtype=int),
                                         np.array(fact2entity_f, dtype=int),
                                         np.array([1.0] * len(fact2entity_e)))

            next_id += 1


    def _build_kb_adj_mat(self, sample_ids, fact_dropout):
        """Create sparse matrix representation for batched data"""
        mats0_batch = np.array([], dtype=int)
        mats0_0 = np.array([], dtype=int)
        mats0_1 = np.array([], dtype=int)
        vals0 = np.array([], dtype=float)

        mats1_batch = np.array([], dtype=int)
        mats1_0 = np.array([], dtype=int)
        mats1_1 = np.array([], dtype=int)
        vals1 = np.array([], dtype=float)

        for i, sample_id in enumerate(sample_ids):
            (mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1) = self.kb_adj_mats[sample_id]
            assert len(val0) == len(val1)
            num_fact = len(val0)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            # mat0
            mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
            mats0_0 = np.append(mats0_0, mat0_0)
            mats0_1 = np.append(mats0_1, mat0_1)
            vals0 = np.append(vals0, val0)
            # mat1
            mats1_batch = np.append(mats1_batch, np.full(num_fact, i, dtype=int))
            mats1_0 = np.append(mats1_0, mat1_0)
            mats1_1 = np.append(mats1_1, mat1_1)
            vals1 = np.append(vals1, val1)

        return (mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)


    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
        return self.local_entities[sample_ids], \
               self.q2e_adj_mats[sample_ids], \
               (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
               self.kb_fact_rels[sample_ids], \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]


    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps


    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity in entities:
            entity_text = entity['text']
            entity_global_id = entity2id[entity_text]
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)


class KvMemDataLoader():
    def __init__(self, data_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)

        print('loading data from', data_file)
        self.data = []
        with open(data_file) as f_in:
            for line in tqdm(f_in.readlines()):
                line = json.loads(line)
                self.data.append(line)
                self.max_facts = max(self.max_facts, len(line['subgraph']['tuples']))

        print('max_facts: ', self.max_facts)

        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id

        self.id2entity = {i: entity for entity, i in entity2id.items()}

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()
        print('max_local_entity', self.max_local_entity)

        print('preparing data ...')
        # local entity id -> global entity id
        self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)

        self.head_entity_ids = np.full(shape=(self.num_data, self.max_facts), fill_value=len(self.entity2id), dtype=int)
        self.relation_ids = np.full(shape=(self.num_data, self.max_facts), fill_value=len(self.relation2id), dtype=int)
        self.tail_entity_ids = np.full(shape=(self.num_data, self.max_facts), fill_value=len(self.entity2id), dtype=int)
        self.triplet_mask = np.zeros(shape=(self.num_data, self.max_facts), dtype=float)

        self._prepare_data()

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50

        for sample_id, sample in tqdm(enumerate(self.data)):
            # local entity id -> global entity id
            g2l = self.global2local_entity_maps[next_id]
            for global_entity, local_entity in g2l.items():
                if local_entity != 0:  # skip question node
                    self.local_entities[next_id, local_entity] = global_entity

            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                self.head_entity_ids[sample_id, i] = self.entity2id[sbj['text']]
                self.relation_ids[sample_id, i] = self.relation2id[rel['text']]
                self.tail_entity_ids[sample_id, i] = self.entity2id[obj['text']]
                self.triplet_mask[sample_id, i] = 1.0

            # tokenize question
            count_query_length[len(sample['question'].split())] += 1
            for j, word in enumerate(sample['question'].split()):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            # construct distribution for answers
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                if self.entity2id[answer[keyword]] in g2l:
                    self.answer_dists[next_id, g2l[self.entity2id[answer[keyword]]]] = 1.0

            next_id += 1

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
        return self.local_entities[sample_ids], \
               self.head_entity_ids[sample_ids], \
               self.relation_ids[sample_ids], \
               self.tail_entity_ids[sample_ids], \
               self.triplet_mask[sample_ids], \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity in entities:
            entity_text = entity['text']
            entity_global_id = entity2id[entity_text]
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)


class GraphMemDataLoader():
    def __init__(self, data_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
        self.use_inverse_relation = use_inverse_relation
        self.max_local_entity = 0
        self.max_facts = 0
        self.max_query_word = max_query_word
        self.count_max_query_word = 0

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)

        print('loading data from', data_file)
        self.data = []
        with open(data_file) as f_in:
            for line in tqdm(f_in.readlines()):
                line = json.loads(line)
                self.data.append(line)
                if self.use_inverse_relation:
                    self.max_facts = max(self.max_facts, 2 * len(line['subgraph']['tuples']))
                else:
                    self.max_facts = max(self.max_facts, len(line['subgraph']['tuples']))

        print('max_facts: ', self.max_facts)

        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id

        self.id2entity = {i: entity for entity, i in entity2id.items()}

        print('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()
        print('max_local_entity', self.max_local_entity)

        print('preparing data ...')
        # local entity id -> global entity id
        self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)

        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
        self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)

        self._prepare_data()
        print("max query word count: {}".format(self.count_max_query_word))

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50

        for sample_id, sample in tqdm(enumerate(self.data)):
            # local entity id -> global entity id
            g2l = self.global2local_entity_maps[next_id]
            for global_entity, local_entity in g2l.items():
                if local_entity != 0:  # skip question node
                    self.local_entities[next_id, local_entity] = global_entity

            # for building sparse adjacency matrices
            # entity2fact, head entity -> fact
            # fact2entity, fact -> tail entity
            entity2fact_e, entity2fact_f = [], []
            entity2fact_reverse_e, entity2fact_reverse_f = [], []
            fact2entity_f, fact2entity_e = [], []
            ent_adj_head, ent_adj_tail = [], []
            relation2fact_r, relation2fact_f = [], []
            fact2relation_f, fact2relation_r = [], []

            # relations in local KB
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                if not self.use_inverse_relation:
                    entity2fact_e += [g2l[self.entity2id[sbj['text']]]]
                    entity2fact_f += [i]
                    fact2entity_f += [i]
                    fact2entity_e += [g2l[self.entity2id[obj['text']]]]
                    self.kb_fact_rels[next_id, i] = self.relation2id[rel['text']]
                else:
                    entity2fact_e += [g2l[self.entity2id[sbj['text']]], g2l[self.entity2id[obj['text']]]]
                    entity2fact_f += [2 * i, 2 * i + 1]

                    entity2fact_reverse_e += [g2l[self.entity2id[obj['text']]], g2l[self.entity2id[sbj['text']]]]
                    entity2fact_reverse_f += [2 * i, 2 * i + 1]

                    fact2entity_f += [2 * i, 2 * i + 1]
                    fact2entity_e += [g2l[self.entity2id[obj['text']]], g2l[self.entity2id[sbj['text']]]]

                    relation2fact_r += [self.relation2id[rel['text']], self.relation2id[rel['text']] + len(self.relation2id)]
                    relation2fact_f += [2 * i, 2 * i + 1]

                    fact2relation_f += [2 * i, 2 * i + 1]
                    fact2relation_r += [self.relation2id[rel['text']], self.relation2id[rel['text']] + len(self.relation2id)]

                    self.kb_fact_rels[next_id, 2 * i] = self.relation2id[rel['text']]
                    self.kb_fact_rels[next_id, 2 * i + 1] = self.relation2id[rel['text']] + len(self.relation2id)

                    ent_adj_head += [g2l[self.entity2id[sbj['text']]], g2l[self.entity2id[obj['text']]]]
                    ent_adj_tail += [g2l[self.entity2id[obj['text']]], g2l[self.entity2id[sbj['text']]]]

            # build connection between question and entities in it
            for j, entity in enumerate(sample['entities']):
                self.q2e_adj_mats[next_id, g2l[self.entity2id[str(entity['text'])]], 0] = 1.0

            # tokenize question
            self.count_max_query_word = max(self.count_max_query_word, len(sample['question'].split()))
            count_query_length[len(sample['question'].split())] += 1
            for j, word in enumerate(sample['question'].split()):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            # construct distribution for answers
            for answer in sample['answers']:
                keyword = 'text' if type(answer['kb_id']) == int else 'kb_id'
                if self.entity2id[answer[keyword]] in g2l:
                    self.answer_dists[next_id, g2l[self.entity2id[answer[keyword]]]] = 1.0

            self.kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int),
                                         np.array(entity2fact_e, dtype=int),
                                         np.array([1.0] * len(entity2fact_f))), \
                                        (np.array(entity2fact_reverse_f, dtype=int),
                                         np.array(entity2fact_reverse_e, dtype=int),
                                         np.array([1.0] * len(entity2fact_reverse_f))), \
                                        (np.array(fact2entity_e, dtype=int),
                                         np.array(fact2entity_f, dtype=int),
                                         np.array([1.0] * len(fact2entity_e))), \
                                        (np.array(ent_adj_head, dtype=int),
                                         np.array(ent_adj_tail, dtype=int),
                                         np.array([1.0] * len(ent_adj_tail))), \
                                        (np.array(relation2fact_r, dtype=int),
                                         np.array(relation2fact_f, dtype=int),
                                         np.array([1.0] * len(relation2fact_r))), \
                                        (np.array(fact2relation_f, dtype=int),
                                         np.array(fact2relation_r, dtype=int),
                                         np.array([1.0] * len(fact2relation_f)))
            next_id += 1

    def _build_kb_adj_mat(self, sample_ids, fact_dropout):
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

        for i, sample_id in enumerate(sample_ids):
            (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
            (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
            (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats[sample_id]
            # assert len(val0) == len(val1)
            num_fact = len(val0)
            num_ent = len(val2)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))

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

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def get_batch(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
        return self.local_entities[sample_ids], \
               self.q2e_adj_mats[sample_ids], \
               (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
               self.kb_fact_rels[sample_ids], \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]

    def get_batch_(self, iteration, batch_size, fact_dropout):
        """
        *** return values ***
        :local_entity: global_id of each entity (batch_size, max_local_entity)
        :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
        :query_text: a list of words in the query (batch_size, max_query_word)
        :rel_document_ids: (batch_size, max_relevant_doc)
        :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
        """
        sample_ids = self.batches[batch_size * iteration: min(batch_size * (iteration + 1), self.num_data)]
        return self.local_entities[sample_ids], \
               self.q2e_adj_mats[sample_ids], \
               (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
               self.kb_fact_rels[sample_ids], \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        print('avg local entity: ', total_local_entity / next_id)
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity in entities:
            entity_text = entity['text']
            entity_global_id = entity2id[entity_text]
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)