import os
import re
import json
import torch
import time
import numpy as np
import pickle
import copy

from scipy.sparse import coo_matrix, csr_matrix
from util import load_dict, use_cuda
from tqdm import tqdm
import ipdb

# class MetaQaGraphMemDataLoader():
#     def __init__(self, triplets, data_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation, batch_size):
#         self.use_inverse_relation = use_inverse_relation
#         self.max_local_entity = len(entity2id)
#         self.max_facts = 0
#         self.max_query_word = max_query_word
#         self.batch_size = batch_size
#         self.count_max_query_word = 0
#
#         self.triplet_ids = []
#         for triplet in triplets:
#             (head, rel, tail) = triplet
#             head_id, rel_id, tail_id = entity2id[head], relation2id[rel], entity2id[tail]
#             self.triplet_ids.append((head_id, rel_id, tail_id))
#
#         if self.use_inverse_relation:
#             self.num_kb_relation = 2 * len(relation2id)
#             self.max_facts = 2 * len(triplets)
#         else:
#             self.num_kb_relation = len(relation2id)
#             self.max_facts = len(triplets)
#
#         print('loading data from', data_file)
#         self.data = []
#         with open(data_file) as f_in:
#             for line in tqdm(f_in.readlines()):
#                 self.data.append(line)
#
#         print('max_facts: ', self.max_facts)
#
#         self.num_data = len(self.data)
#         self.batches = np.arange(self.num_data)
#
#         print('building word index ...')
#         self.word2id = word2id
#         self.relation2id = relation2id
#         self.entity2id = entity2id
#
#         self.id2entity = {i: entity for entity, i in entity2id.items()}
#
#         # local entity id -> global entity id
#         self.local_entities = np.tile(np.arange(self.max_local_entity), (self.batch_size, 1))
#         self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
#         self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=bool)
#
#         self.kb_adj_mats = None
#         self.batch_adj_data = None
#         self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=bool)
#         # self.q2e_adj_mats = np.zeros((self.num_data), dtype=int)
#         self.kb_fact_rels = np.full((self.batch_size, self.max_facts), self.num_kb_relation, dtype=int)
#
#         print("Preparing data...")
#         self._prepare_data()
#         print("Building adj mats...")
#         self._prepare_adj_mats()
#         print("Building batch adj mats...")
#         self._prepare_batch_adj_mats()
#
#         print("Max query words counted: {}", format(self.count_max_query_word))
#
#         self.data = None
#
#     def clear(self):
#         self.local_entities = None
#         self.query_texts = None
#         self.answer_dists = None
#         self.kb_adj_mats = None
#         self.batch_adj_data = None
#         self.q2e_adj_mats = None
#         self.kb_fact_rels = None
#
#     def save(self, dir):
#         local_entities_path = os.path.join(dir, "local_entities.npy")
#         query_texts_path = os.path.join(dir, "query_texts.npy")
#         answer_dists_path = os.path.join(dir, "answer_dists.npy")
#         batch_adj_data_path = os.path.join(dir, "batch_adj_data.pkl")
#         q2e_adj_mats_path = os.path.join(dir, "q2e_adj_mats.npy")
#         kb_fact_rels_path = os.path.join(dir, "kb_fact_rels.npy")
#
#         np.save(local_entities_path, self.local_entities)
#         np.save(query_texts_path, self.query_texts)
#         np.save(answer_dists_path, self.answer_dists)
#         np.save(q2e_adj_mats_path, self.q2e_adj_mats)
#         np.save(kb_fact_rels_path, self.kb_fact_rels)
#
#         with open(batch_adj_data_path, 'wb') as f:
#             pickle.dump(self.batch_adj_data, f)
#
#     def load(self, dir):
#         local_entities_path = os.path.join(dir, "local_entities.npy")
#         query_texts_path = os.path.join(dir, "query_texts.npy")
#         answer_dists_path = os.path.join(dir, "answer_dists.npy")
#         batch_adj_data_path = os.path.join(dir, "batch_adj_data.pkl")
#         q2e_adj_mats_path = os.path.join(dir, "q2e_adj_mats.npy")
#         kb_fact_rels_path = os.path.join(dir, "kb_fact_rels.npy")
#
#         self.local_entities = np.load(local_entities_path)
#         self.query_texts = np.load(query_texts_path)
#         self.answer_dists = np.load(answer_dists_path)
#         self.q2e_adj_mats = np.load(q2e_adj_mats_path)
#         self.kb_fact_rels = np.load(kb_fact_rels_path)
#
#         with open(batch_adj_data_path, 'rb') as f:
#             self.batch_adj_data = pickle.load(f)
#
#     def _prepare_data(self):
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             question, answers = sample.strip().split('\t')
#             answers = answers.split("|")
#
#             topic_entity = question[question.find('[')+1: question.find(']')]
#             question = re.sub('[\[!@#$,.?!()\]]', '', question)
#             question = question.replace("'s", "")
#             question = question.replace("'", "")
#             question = question.replace("-", " ")
#             tokens = question.split(' ')
#
#             self.count_max_query_word = max(self.count_max_query_word, len(tokens))
#
#             # query words
#             for j, word in enumerate(tokens):
#                 if j < self.max_query_word:
#                     if word in self.word2id:
#                         self.query_texts[sample_id, j] = self.word2id[word]
#                     else:
#                         self.query_texts[sample_id, j] = self.word2id['__unk__']
#
#             # topic entity
#             self.q2e_adj_mats[sample_id, self.entity2id[topic_entity], 0] = 1
#             # self.q2e_adj_mats[sample_id] = self.entity2id[topic_entity]
#
#             # answers
#             for answer in answers:
#                 self.answer_dists[sample_id, self.entity2id[answer]] = 1.0
#
#     def _prepare_adj_mats(self):
#         """
#         global2local_entity_maps: a map from global entity id to local entity id
#         adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
#         """
#
#         # for building sparse adjacency matrices
#         # entity2fact, head entity -> fact
#         # fact2entity, fact -> tail entity
#         entity2fact_e, entity2fact_f = [], []
#         entity2fact_reverse_e, entity2fact_reverse_f = [], []
#         fact2entity_f, fact2entity_e = [], []
#         ent_adj_head, ent_adj_tail = [], []
#         relation2fact_r, relation2fact_f = [], []
#         fact2relation_f, fact2relation_r = [], []
#
#         for i, tpl in enumerate(tqdm(self.triplet_ids)):
#             sbj, rel, obj = tpl
#             if not self.use_inverse_relation:
#                 entity2fact_e += [sbj]
#                 entity2fact_f += [i]
#                 fact2entity_f += [i]
#                 fact2entity_e += [obj]
#                 self.kb_fact_rels[:, i] = rel
#             else:
#                 entity2fact_e += [sbj, obj]
#                 entity2fact_f += [2 * i, 2 * i + 1]
#
#                 entity2fact_reverse_e += [obj, sbj]
#                 entity2fact_reverse_f += [2 * i, 2 * i + 1]
#
#                 fact2entity_f += [2 * i, 2 * i + 1]
#                 fact2entity_e += [obj, sbj]
#
#                 relation2fact_r += [rel, rel + len(self.relation2id)]
#                 relation2fact_f += [2 * i, 2 * i + 1]
#
#                 fact2relation_f += [2 * i, 2 * i + 1]
#                 fact2relation_r += [rel, rel + len(self.relation2id)]
#
#                 self.kb_fact_rels[:, 2 * i] = rel
#                 self.kb_fact_rels[:, 2 * i + 1] = rel + len(self.relation2id)
#
#                 ent_adj_head += [sbj, obj]
#                 ent_adj_tail += [obj, sbj]
#
#         self.kb_adj_mats = (np.array(entity2fact_f, dtype=int),
#                                      np.array(entity2fact_e, dtype=int),
#                                      np.array([1.0] * len(entity2fact_f))), \
#                                     (np.array(entity2fact_reverse_f, dtype=int),
#                                      np.array(entity2fact_reverse_e, dtype=int),
#                                      np.array([1.0] * len(entity2fact_reverse_f))), \
#                                     (np.array(fact2entity_e, dtype=int),
#                                      np.array(fact2entity_f, dtype=int),
#                                      np.array([1.0] * len(fact2entity_e))), \
#                                     (np.array(ent_adj_head, dtype=int),
#                                      np.array(ent_adj_tail, dtype=int),
#                                      np.array([1.0] * len(ent_adj_tail))), \
#                                     (np.array(relation2fact_r, dtype=int),
#                                      np.array(relation2fact_f, dtype=int),
#                                      np.array([1.0] * len(relation2fact_r))), \
#                                     (np.array(fact2relation_f, dtype=int),
#                                      np.array(fact2relation_r, dtype=int),
#                                      np.array([1.0] * len(fact2relation_f)))
#
#
#     def _prepare_batch_adj_mats(self):
#         """Create sparse matrix representation for batched data"""
#         mats0_batch = np.array([], dtype=int)
#         mats0_0 = np.array([], dtype=int)
#         mats0_1 = np.array([], dtype=int)
#         vals0 = np.array([], dtype=float)
#
#         mats0_rev_batch = np.array([], dtype=int)
#         mats0_rev_0 = np.array([], dtype=int)
#         mats0_rev_1 = np.array([], dtype=int)
#         vals0_rev = np.array([], dtype=float)
#
#         mats1_batch = np.array([], dtype=int)
#         mats1_0 = np.array([], dtype=int)
#         mats1_1 = np.array([], dtype=int)
#         vals1 = np.array([], dtype=float)
#
#         mats2_batch = np.array([], dtype=int)
#         mats2_0 = np.array([], dtype=int)
#         mats2_1 = np.array([], dtype=int)
#         vals2 = np.array([], dtype=float)
#
#         mats3_batch = np.array([], dtype=int)
#         mats3_0 = np.array([], dtype=int)
#         mats3_1 = np.array([], dtype=int)
#         vals3 = np.array([], dtype=float)
#
#         mats4_batch = np.array([], dtype=int)
#         mats4_0 = np.array([], dtype=int)
#         mats4_1 = np.array([], dtype=int)
#         vals4 = np.array([], dtype=float)
#
#         (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
#         (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
#         (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats
#
#         for i in tqdm(range(self.batch_size)):
#             # assert len(val0) == len(val1)
#             num_fact = len(val0)
#             num_ent = len(val2)
#
#             # mat0
#             mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
#             mats0_0 = np.append(mats0_0, mat0_0)
#             mats0_1 = np.append(mats0_1, mat0_1)
#             vals0 = np.append(vals0, val0)
#
#             # mat0_rev
#             mats0_rev_batch = np.append(mats0_rev_batch, np.full(num_fact, i, dtype=int))
#             mats0_rev_0 = np.append(mats0_rev_0, mat0_rev_0)
#             mats0_rev_1 = np.append(mats0_rev_1, mat0_rev_1)
#             vals0_rev = np.append(vals0_rev, val0_rev)
#
#
#             # mat1
#             mats1_batch = np.append(mats1_batch, np.full(len(val1), i, dtype=int))
#             mats1_0 = np.append(mats1_0, mat1_0)
#             mats1_1 = np.append(mats1_1, mat1_1)
#             vals1 = np.append(vals1, val1)
#
#             # mat2
#             mats2_batch = np.append(mats2_batch, np.full(num_ent, i, dtype=int))
#             mats2_0 = np.append(mats2_0, mat2_0)
#             mats2_1 = np.append(mats2_1, mat2_1)
#             vals2 = np.append(vals2, val2)
#
#             # mat3
#             mats3_batch = np.append(mats3_batch, np.full(num_fact, i, dtype=int))
#             mats3_0 = np.append(mats3_0, mat3_0)
#             mats3_1 = np.append(mats3_1, mat3_1)
#             vals3 = np.append(vals3, val3)
#
#             # mat4
#             mats4_batch = np.append(mats4_batch, np.full(num_fact, i, dtype=int))
#             mats4_0 = np.append(mats4_0, mat4_0)
#             mats4_1 = np.append(mats4_1, mat4_1)
#             vals4 = np.append(vals4, val4)
#
#         self.batch_adj_data = (mats0_batch, mats0_0, mats0_1, vals0), \
#                (mats0_rev_batch, mats0_rev_0, mats0_rev_1, vals0_rev), \
#                (mats1_batch, mats1_0, mats1_1, vals1), \
#                (mats2_batch, mats2_0, mats2_1, vals2), \
#                (mats3_batch, mats3_0, mats3_1, vals3), \
#                (mats4_batch, mats4_0, mats4_1, vals4)
#
#
#     def make_adj(self):
#         (e2f_batch, e2f_f, e2f_e, e2f_val), \
#         (e2f_rev_batch, e2f_rev_f, e2f_rev_e, e2f_rev_val), \
#         (f2e_batch, f2e_e, f2e_f, f2e_val), \
#         (e2e_batch, e2e_h, e2e_t, e2e_val), \
#         (r2f_batch, r2f_f, r2f_r, r2f_val), \
#         (f2r_batch, f2r_r, f2r_f, f2r_val), \
#             = self.batch_adj_data
#
#         batch_size = self.batch_size
#         max_fact = self.max_facts
#         max_local_entity = self.max_local_entity
#         num_relation = self.num_kb_relation
#
#         entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
#         entity2fact_val = torch.FloatTensor(e2f_val)
#         # [batch_size, max_fact, max_local_entity]
#         fact2entity_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
#             [batch_size, max_fact, max_local_entity])) # batch_size, max_fact, max_local_entity
#
#         entity2fact_rev_index = torch.LongTensor([e2f_rev_batch, e2f_rev_f, e2f_rev_e])
#         entity2fact_rev_val = torch.FloatTensor(e2f_rev_val)
#         # [batch_size, max_fact, max_local_entity]
#         fact2entity_rev_mat = torch.sparse.FloatTensor(entity2fact_rev_index, entity2fact_rev_val, torch.Size(
#             [batch_size, max_fact, max_local_entity]))  # batch_size, max_fact, max_local_entity
#
#         fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
#         fact2entity_val = torch.FloatTensor(f2e_val)
#         # [batch_size, max_local_entity, max_fact]
#         entity2fact_mat = torch.sparse.FloatTensor(fact2entity_index, fact2entity_val,
#                                                             torch.Size([batch_size, max_local_entity, max_fact]))
#
#         entity2entity_index = torch.LongTensor([e2e_batch, e2e_h, e2e_t])
#         entity2entity_val = torch.FloatTensor(e2e_val)
#         # [batch_size, max_fact, max_local_entity]
#         entity2entity_mat = torch.sparse.FloatTensor(entity2entity_index,
#                                                               entity2entity_val,
#                                                               torch.Size([batch_size, max_local_entity,
#                                                                           max_local_entity]))  # batch_size, max_fact, max_local_entity
#
#         relation2fact_index = torch.LongTensor([r2f_batch, r2f_f, r2f_r])
#         relation2fact_val = torch.FloatTensor(r2f_val)
#         # [batch_size, num_relation, max_fact]
#         relation2fact_mat = torch.sparse.FloatTensor(relation2fact_index, relation2fact_val, torch.Size(
#             [batch_size, num_relation+1, max_fact]))
#
#         fact2relation_index = torch.LongTensor([f2r_batch, f2r_r, f2r_f])
#         fact2relation_val = torch.FloatTensor(f2r_val)
#         # [batch_size, max_local_entity, max_fact]
#         fact2relation_mat = torch.sparse.FloatTensor(fact2relation_index, fact2relation_val,
#                                                             torch.Size([batch_size, max_fact, num_relation+1]))
#
#         self.batch_adj_data = fact2entity_mat, fact2entity_rev_mat, entity2fact_mat, \
#                               entity2entity_mat, relation2fact_mat, fact2relation_mat
#
#     def reset_batches(self, is_sequential=True):
#         if is_sequential:
#             self.batches = np.arange(self.num_data)
#         else:
#             self.batches = np.random.permutation(self.num_data)
#
#     def get_batch(self, iteration, batch_size, fact_dropout):
#         """
#         *** return values ***
#         :local_entity: global_id of each entity (batch_size, max_local_entity)
#         :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
#         :query_text: a list of words in the query (batch_size, max_query_word)
#         :rel_document_ids: (batch_size, max_relevant_doc)
#         :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
#         """
#         sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
#         return self.local_entities, \
#                self.q2e_adj_mats[sample_ids], \
#                (self.batch_adj_data), \
#                self.kb_fact_rels, \
#                self.query_texts[sample_ids], \
#                self.answer_dists[sample_ids]

"backup"
# class MetaQaGraphMemDataLoader():
#     def __init__(self, data_file, subgraph_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
#         self.use_inverse_relation = use_inverse_relation
#         self.max_local_entity = 0
#         self.max_facts = 0
#         self.max_query_word = max_query_word
#         self.count_max_query_word = 0
#
#         if self.use_inverse_relation:
#             self.num_kb_relation = 2 * len(relation2id)
#         else:
#             self.num_kb_relation = len(relation2id)
#
#         print('loading data from', data_file)
#         self.data = []
#         with open(data_file) as f_in:
#             for line in tqdm(f_in.readlines()):
#                 self.data.append(line)
#
#         print("Loading subgraph ...")
#         with open(subgraph_file, 'rb') as f:
#             self.subgraphs = pickle.load(f)
#
#         for ent_id, val in self.subgraphs.items():
#             if self.use_inverse_relation:
#                 self.max_facts = max(self.max_facts, 2 * len(val[1]))
#             else:
#                 self.max_facts = max(self.max_facts, len(val[1]))
#
#         print('max_facts: ', self.max_facts)
#
#         self.num_data = len(self.data)
#         self.batches = np.arange(self.num_data)
#
#         print('building word index ...')
#         self.word2id = word2id
#         self.relation2id = relation2id
#         self.entity2id = entity2id
#
#         self.id2entity = {i: entity for entity, i in entity2id.items()}
#
#         print("preprocessing data ...")
#         self._preprocess_data()
#
#         print('converting global to local entity index ...')
#         self.global2local_entity_maps = self._build_global2local_entity_maps()
#         print('max_local_entity', self.max_local_entity)
#
#         print('preparing data ...')
#         # local entity id -> global entity id
#         self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
#         self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
#         self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
#
#         self.kb_adj_mats = np.empty(self.num_data, dtype=object)
#         self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
#         self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
#
#         self._prepare_data()
#         print("max query word count: {}".format(self.count_max_query_word))
#         self.subgraphs = None
#
#     def _preprocess_data(self):
#         processed_data = []
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             question, answers = sample.strip().split('\t')
#             answers = answers.split("|")
#
#             topic_entity = question[question.find('[')+1: question.find(']')]
#
#             question = re.sub('[\[!@#$,.?!()\]]', '', question)
#             question = question.replace("'s", "")
#             question = question.replace("'", "")
#             question = question.replace("-", " ")
#             tokens = question.split(' ')
#
#             processed_data.append((tokens, answers, topic_entity))
#         self.data = processed_data
#
#     def _prepare_data(self):
#         """
#         global2local_entity_maps: a map from global entity id to local entity id
#         adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
#         """
#         next_id = 0
#         count_query_length = [0] * 50
#         total_miss_answer = 0
#
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             # local entity id -> global entity id
#             g2l = self.global2local_entity_maps[next_id]
#             for global_entity, local_entity in g2l.items():
#                 if local_entity != 0:  # skip question node
#                     self.local_entities[next_id, local_entity] = global_entity
#
#             # for building sparse adjacency matrices
#             # entity2fact, head entity -> fact
#             # fact2entity, fact -> tail entity
#             entity2fact_e, entity2fact_f = [], []
#             entity2fact_reverse_e, entity2fact_reverse_f = [], []
#             fact2entity_f, fact2entity_e = [], []
#             ent_adj_head, ent_adj_tail = [], []
#             relation2fact_r, relation2fact_f = [], []
#             fact2relation_f, fact2relation_r = [], []
#
#             (tokens, answers, topic_entity) = sample
#             subgraph = self.subgraphs[self.entity2id[topic_entity]]
#             subgraph_entities, subgraph_triplets = subgraph
#
#             # relations in local KB
#             for i, tpl in enumerate(subgraph_triplets):
#                 sbj, rel, obj = tpl
#                 if not self.use_inverse_relation:
#                     entity2fact_e += [g2l[sbj]]
#                     entity2fact_f += [i]
#                     fact2entity_f += [i]
#                     fact2entity_e += [g2l[obj]]
#                     self.kb_fact_rels[next_id, i] = rel
#                 else:
#                     entity2fact_e += [g2l[sbj], g2l[obj]]
#                     entity2fact_f += [2 * i, 2 * i + 1]
#
#                     entity2fact_reverse_e += [g2l[obj], g2l[sbj]]
#                     entity2fact_reverse_f += [2 * i, 2 * i + 1]
#
#                     fact2entity_f += [2 * i, 2 * i + 1]
#                     fact2entity_e += [g2l[obj], g2l[sbj]]
#
#                     relation2fact_r += [rel, rel + len(self.relation2id)]
#                     relation2fact_f += [2 * i, 2 * i + 1]
#
#                     fact2relation_f += [2 * i, 2 * i + 1]
#                     fact2relation_r += [rel, rel + len(self.relation2id)]
#
#                     self.kb_fact_rels[next_id, 2 * i] = rel
#                     self.kb_fact_rels[next_id, 2 * i + 1] = rel + len(self.relation2id)
#
#                     ent_adj_head += [g2l[sbj], g2l[obj]]
#                     ent_adj_tail += [g2l[obj], g2l[sbj]]
#
#             # build connection between question and entities in it
#             self.q2e_adj_mats[next_id, g2l[self.entity2id[topic_entity]], 0] = 1.0
#
#             # tokenize question
#             self.count_max_query_word = max(self.count_max_query_word, len(tokens))
#             for j, word in enumerate(tokens):
#                 if j < self.max_query_word:
#                     if word in self.word2id:
#                         self.query_texts[next_id, j] = self.word2id[word]
#                     else:
#                         self.query_texts[next_id, j] = self.word2id['__unk__']
#
#             # construct distribution for answers
#             miss_answer = 0
#             for answer in answers:
#                 if self.entity2id[answer] in g2l:
#                     self.answer_dists[next_id, g2l[self.entity2id[answer]]] = 1.0
#                 else:
#                     miss_answer = 1
#                     # print("!!!! answer not in subgraph")
#                     # print("question: {}".format(tokens))
#                     # print("missed answer: {}".format(answer))
#
#             self.kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int),
#                                          np.array(entity2fact_e, dtype=int),
#                                          np.array([1.0] * len(entity2fact_f))), \
#                                         (np.array(entity2fact_reverse_f, dtype=int),
#                                          np.array(entity2fact_reverse_e, dtype=int),
#                                          np.array([1.0] * len(entity2fact_reverse_f))), \
#                                         (np.array(fact2entity_e, dtype=int),
#                                          np.array(fact2entity_f, dtype=int),
#                                          np.array([1.0] * len(fact2entity_e))), \
#                                         (np.array(ent_adj_head, dtype=int),
#                                          np.array(ent_adj_tail, dtype=int),
#                                          np.array([1.0] * len(ent_adj_tail))), \
#                                         (np.array(relation2fact_r, dtype=int),
#                                          np.array(relation2fact_f, dtype=int),
#                                          np.array([1.0] * len(relation2fact_r))), \
#                                         (np.array(fact2relation_f, dtype=int),
#                                          np.array(fact2relation_r, dtype=int),
#                                          np.array([1.0] * len(fact2relation_f)))
#             next_id += 1
#             total_miss_answer += miss_answer
#
#         print("Total data: {}".format(len(self.data)))
#         print("Miss answer: {}".format(total_miss_answer))
#
#     def _build_kb_adj_mat(self, sample_ids, fact_dropout):
#         """Create sparse matrix representation for batched data"""
#         mats0_batch = np.array([], dtype=int)
#         mats0_0 = np.array([], dtype=int)
#         mats0_1 = np.array([], dtype=int)
#         vals0 = np.array([], dtype=float)
#
#         mats0_rev_batch = np.array([], dtype=int)
#         mats0_rev_0 = np.array([], dtype=int)
#         mats0_rev_1 = np.array([], dtype=int)
#         vals0_rev = np.array([], dtype=float)
#
#         mats1_batch = np.array([], dtype=int)
#         mats1_0 = np.array([], dtype=int)
#         mats1_1 = np.array([], dtype=int)
#         vals1 = np.array([], dtype=float)
#
#         mats2_batch = np.array([], dtype=int)
#         mats2_0 = np.array([], dtype=int)
#         mats2_1 = np.array([], dtype=int)
#         vals2 = np.array([], dtype=float)
#
#         mats3_batch = np.array([], dtype=int)
#         mats3_0 = np.array([], dtype=int)
#         mats3_1 = np.array([], dtype=int)
#         vals3 = np.array([], dtype=float)
#
#         mats4_batch = np.array([], dtype=int)
#         mats4_0 = np.array([], dtype=int)
#         mats4_1 = np.array([], dtype=int)
#         vals4 = np.array([], dtype=float)
#
#         for i, sample_id in enumerate(sample_ids):
#             (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
#             (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
#             (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats[sample_id]
#             # assert len(val0) == len(val1)
#             num_fact = len(val0)
#             num_ent = len(val2)
#             num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
#
#             # mat0
#             mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
#             mats0_0 = np.append(mats0_0, mat0_0)
#             mats0_1 = np.append(mats0_1, mat0_1)
#             vals0 = np.append(vals0, val0)
#
#             # mat0_rev
#             mats0_rev_batch = np.append(mats0_rev_batch, np.full(num_fact, i, dtype=int))
#             mats0_rev_0 = np.append(mats0_rev_0, mat0_rev_0)
#             mats0_rev_1 = np.append(mats0_rev_1, mat0_rev_1)
#             vals0_rev = np.append(vals0_rev, val0_rev)
#
#
#             # mat1
#             mats1_batch = np.append(mats1_batch, np.full(len(val1), i, dtype=int))
#             mats1_0 = np.append(mats1_0, mat1_0)
#             mats1_1 = np.append(mats1_1, mat1_1)
#             vals1 = np.append(vals1, val1)
#
#             # mat2
#             mats2_batch = np.append(mats2_batch, np.full(num_ent, i, dtype=int))
#             mats2_0 = np.append(mats2_0, mat2_0)
#             mats2_1 = np.append(mats2_1, mat2_1)
#             vals2 = np.append(vals2, val2)
#
#             # mat3
#             mats3_batch = np.append(mats3_batch, np.full(num_fact, i, dtype=int))
#             mats3_0 = np.append(mats3_0, mat3_0)
#             mats3_1 = np.append(mats3_1, mat3_1)
#             vals3 = np.append(vals3, val3)
#
#             # mat4
#             mats4_batch = np.append(mats4_batch, np.full(num_fact, i, dtype=int))
#             mats4_0 = np.append(mats4_0, mat4_0)
#             mats4_1 = np.append(mats4_1, mat4_1)
#             vals4 = np.append(vals4, val4)
#
#         return (mats0_batch, mats0_0, mats0_1, vals0), \
#                (mats0_rev_batch, mats0_rev_0, mats0_rev_1, vals0_rev), \
#                (mats1_batch, mats1_0, mats1_1, vals1), \
#                (mats2_batch, mats2_0, mats2_1, vals2), \
#                (mats3_batch, mats3_0, mats3_1, vals3), \
#                (mats4_batch, mats4_0, mats4_1, vals4)
#
#     def reset_batches(self, is_sequential=True):
#         if is_sequential:
#             self.batches = np.arange(self.num_data)
#         else:
#             self.batches = np.random.permutation(self.num_data)
#
#     def get_batch(self, iteration, batch_size, fact_dropout):
#         """
#         *** return values ***
#         :local_entity: global_id of each entity (batch_size, max_local_entity)
#         :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
#         :query_text: a list of words in the query (batch_size, max_query_word)
#         :rel_document_ids: (batch_size, max_relevant_doc)
#         :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
#         """
#         sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
#         return self.local_entities[sample_ids], \
#                self.q2e_adj_mats[sample_ids], \
#                (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
#                self.kb_fact_rels[sample_ids], \
#                self.query_texts[sample_ids], \
#                self.answer_dists[sample_ids]
#
#     def _build_global2local_entity_maps(self):
#         """Create a map from global entity id to local entity of each sample"""
#         global2local_entity_maps = [None] * self.num_data
#         total_local_entity = 0.0
#         next_id = 0
#         for sample in tqdm(self.data):
#             (tokens, answers, topic_entity) = sample
#
#             subgraph = self.subgraphs[self.entity2id[topic_entity]]
#             subgraph_entities, subgraph_triplets = subgraph
#             g2l = dict()
#             self._add_entity_to_map({self.entity2id[topic_entity]}, g2l)
#             # construct a map from global entity id to local entity id
#             self._add_entity_to_map(subgraph_entities, g2l)
#
#             global2local_entity_maps[next_id] = g2l
#             total_local_entity += len(g2l)
#             self.max_local_entity = max(self.max_local_entity, len(g2l))
#             next_id += 1
#         print('avg local entity: ', total_local_entity / next_id)
#         return global2local_entity_maps
#
#     @staticmethod
#     def _add_entity_to_map(entities, g2l):
#         for entity in entities:
#             entity_global_id = entity
#             if entity_global_id not in g2l:
#                 g2l[entity_global_id] = len(g2l)

"""backup simplify"""
# class MetaQaGraphMemDataLoader():
#     def __init__(self, data_file, subgraph_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
#         self.use_inverse_relation = use_inverse_relation
#         self.max_local_entity = 0
#         self.max_facts = 0
#         self.max_query_word = max_query_word
#         self.count_max_query_word = 0
#
#         if self.use_inverse_relation:
#             self.num_kb_relation = 2 * len(relation2id)
#         else:
#             self.num_kb_relation = len(relation2id)
#
#         print('loading data from', data_file)
#         with open(data_file+".pkl", 'rb') as f:
#             (self.sample2topic_entity, self.data) = pickle.load(f)
#
#         # print("loading pickled adj data ...")
#         # start_time = time.time()
#         # adj_data_file = os.path.join(os.path.dirname(data_file), 'data.pkl')
#         # with open(adj_data_file, 'rb') as f:
#         #     (max_facts, global2local_entity_maps, kb_adj_mats, kb_fact_rels) = pickle.load(f)
#         # end_time = time.time()
#         # print("load time: {}".format(end_time-start_time))
#         (max_facts, global2local_entity_maps, kb_adj_mats, kb_fact_rels) = subgraph_file
#         self.max_facts = max_facts
#
#         print('max_facts: ', self.max_facts)
#         self.num_data = len(self.data)
#         self.batches = np.arange(self.num_data)
#
#         print('building word index ...')
#         self.word2id = word2id
#         self.relation2id = relation2id
#         self.entity2id = entity2id
#
#         self.id2entity = {i: entity for entity, i in entity2id.items()}
#
#         print('converting global to local entity index ...')
#         self.global2local_entity_maps = global2local_entity_maps
#         for topic_entity_id, map in self.global2local_entity_maps.items():
#             self.max_local_entity = max(self.max_local_entity, len(map))
#
#         print('max_local_entity', self.max_local_entity)
#
#         print('preparing data ...')
#         # local entity id -> global entity id
#         self.local_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
#         self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
#         self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
#
#         self.kb_adj_mats = kb_adj_mats
#         self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
#         self.kb_fact_rels = kb_fact_rels
#
#         self._prepare_data()
#         print("max query word count: {}".format(self.count_max_query_word))
#
#     def _preprocess_data(self):
#         processed_data = []
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             question, answers = sample.strip().split('\t')
#             answers = answers.split("|")
#
#             topic_entity = question[question.find('[')+1: question.find(']')]
#
#             question = re.sub('[\[!@#$,.?!()\]]', '', question)
#             question = question.replace("'s", "")
#             question = question.replace("'", "")
#             question = question.replace("-", " ")
#             tokens = question.split(' ')
#
#             processed_data.append((tokens, answers, topic_entity))
#         self.data = processed_data
#
#     def _prepare_data(self):
#         """
#         global2local_entity_maps: a map from global entity id to local entity id
#         adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
#         """
#         next_id = 0
#         count_query_length = [0] * 50
#         total_miss_answer = 0
#
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             (tokens, answers, topic_entity) = sample
#             topic_entity_id = self.entity2id[topic_entity]
#
#             # local entity id -> global entity id
#             g2l = self.global2local_entity_maps[topic_entity_id]
#             for global_entity, local_entity in g2l.items():
#                 if local_entity != 0:  # skip question node
#                     self.local_entities[next_id, local_entity] = global_entity
#
#             # build connection between question and entities in it
#             self.q2e_adj_mats[next_id, g2l[topic_entity_id], 0] = 1.0
#
#             # tokenize question
#             self.count_max_query_word = max(self.count_max_query_word, len(tokens))
#             for j, word in enumerate(tokens):
#                 if j < self.max_query_word:
#                     if word in self.word2id:
#                         self.query_texts[next_id, j] = self.word2id[word]
#                     else:
#                         self.query_texts[next_id, j] = self.word2id['__unk__']
#
#             # construct distribution for answers
#             miss_answer = 0
#             for answer in answers:
#                 if self.entity2id[answer] in g2l:
#                     self.answer_dists[next_id, g2l[self.entity2id[answer]]] = 1.0
#                 else:
#                     miss_answer = 1
#
#             next_id += 1
#             total_miss_answer += miss_answer
#
#         print("Total data: {}".format(len(self.data)))
#         print("Miss answer: {}".format(total_miss_answer))
#
#     def _build_kb_adj_mat(self, sample_ids, fact_dropout):
#         """Create sparse matrix representation for batched data"""
#         mats0_batch = np.array([], dtype=int)
#         mats0_0 = np.array([], dtype=int)
#         mats0_1 = np.array([], dtype=int)
#         vals0 = np.array([], dtype=float)
#
#         mats0_rev_batch = np.array([], dtype=int)
#         mats0_rev_0 = np.array([], dtype=int)
#         mats0_rev_1 = np.array([], dtype=int)
#         vals0_rev = np.array([], dtype=float)
#
#         mats1_batch = np.array([], dtype=int)
#         mats1_0 = np.array([], dtype=int)
#         mats1_1 = np.array([], dtype=int)
#         vals1 = np.array([], dtype=float)
#
#         mats2_batch = np.array([], dtype=int)
#         mats2_0 = np.array([], dtype=int)
#         mats2_1 = np.array([], dtype=int)
#         vals2 = np.array([], dtype=float)
#
#         mats3_batch = np.array([], dtype=int)
#         mats3_0 = np.array([], dtype=int)
#         mats3_1 = np.array([], dtype=int)
#         vals3 = np.array([], dtype=float)
#
#         mats4_batch = np.array([], dtype=int)
#         mats4_0 = np.array([], dtype=int)
#         mats4_1 = np.array([], dtype=int)
#         vals4 = np.array([], dtype=float)
#
#         for i, sample_id in enumerate(sample_ids):
#             # topic_entity_id = self.entity2id[self.data[sample_id][2]]
#             topic_entity_id = self.sample2topic_entity[sample_id]
#
#             (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
#             (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
#             (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats[topic_entity_id]
#             # assert len(val0) == len(val1)
#             num_fact = len(val0)
#             num_ent = len(val2)
#             num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
#
#             # mat0
#             mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
#             mats0_0 = np.append(mats0_0, mat0_0)
#             mats0_1 = np.append(mats0_1, mat0_1)
#             vals0 = np.append(vals0, val0)
#
#             # mat0_rev
#             mats0_rev_batch = np.append(mats0_rev_batch, np.full(num_fact, i, dtype=int))
#             mats0_rev_0 = np.append(mats0_rev_0, mat0_rev_0)
#             mats0_rev_1 = np.append(mats0_rev_1, mat0_rev_1)
#             vals0_rev = np.append(vals0_rev, val0_rev)
#
#
#             # mat1
#             mats1_batch = np.append(mats1_batch, np.full(len(val1), i, dtype=int))
#             mats1_0 = np.append(mats1_0, mat1_0)
#             mats1_1 = np.append(mats1_1, mat1_1)
#             vals1 = np.append(vals1, val1)
#
#             # mat2
#             mats2_batch = np.append(mats2_batch, np.full(num_ent, i, dtype=int))
#             mats2_0 = np.append(mats2_0, mat2_0)
#             mats2_1 = np.append(mats2_1, mat2_1)
#             vals2 = np.append(vals2, val2)
#
#             # mat3
#             mats3_batch = np.append(mats3_batch, np.full(num_fact, i, dtype=int))
#             mats3_0 = np.append(mats3_0, mat3_0)
#             mats3_1 = np.append(mats3_1, mat3_1)
#             vals3 = np.append(vals3, val3)
#
#             # mat4
#             mats4_batch = np.append(mats4_batch, np.full(num_fact, i, dtype=int))
#             mats4_0 = np.append(mats4_0, mat4_0)
#             mats4_1 = np.append(mats4_1, mat4_1)
#             vals4 = np.append(vals4, val4)
#
#         return (mats0_batch, mats0_0, mats0_1, vals0), \
#                (mats0_rev_batch, mats0_rev_0, mats0_rev_1, vals0_rev), \
#                (mats1_batch, mats1_0, mats1_1, vals1), \
#                (mats2_batch, mats2_0, mats2_1, vals2), \
#                (mats3_batch, mats3_0, mats3_1, vals3), \
#                (mats4_batch, mats4_0, mats4_1, vals4)
#
#     def reset_batches(self, is_sequential=True):
#         if is_sequential:
#             self.batches = np.arange(self.num_data)
#         else:
#             self.batches = np.random.permutation(self.num_data)
#
#     def get_batch(self, iteration, batch_size, fact_dropout):
#         """
#         *** return values ***
#         :local_entity: global_id of each entity (batch_size, max_local_entity)
#         :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
#         :query_text: a list of words in the query (batch_size, max_query_word)
#         :rel_document_ids: (batch_size, max_relevant_doc)
#         :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
#         """
#         sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
#         return self.local_entities[sample_ids], \
#                self.q2e_adj_mats[sample_ids], \
#                (self._build_kb_adj_mat(sample_ids, fact_dropout=fact_dropout)), \
#                self.kb_fact_rels[self.sample2topic_entity[sample_ids]], \
#                self.query_texts[sample_ids], \
#                self.answer_dists[sample_ids]
#
#     def _build_global2local_entity_maps(self):
#         """Create a map from global entity id to local entity of each sample"""
#         # global2local_entity_maps = [None] * self.num_data
#         global2local_entity_maps = dict()
#         next_id = 0
#         for sample in tqdm(self.data):
#             (tokens, answers, topic_entity) = sample
#             topic_entity_id = self.entity2id[topic_entity]
#             if topic_entity_id in global2local_entity_maps.keys():
#                 continue
#             subgraph = self.subgraphs[topic_entity_id]
#             subgraph_entities, subgraph_triplets = subgraph
#             g2l = dict()
#             self._add_entity_to_map({self.entity2id[topic_entity]}, g2l)
#             # construct a map from global entity id to local entity id
#             self._add_entity_to_map(subgraph_entities, g2l)
#
#             # global2local_entity_maps[next_id] = g2l
#             global2local_entity_maps[topic_entity_id] = g2l
#             self.max_local_entity = max(self.max_local_entity, len(g2l))
#             next_id += 1
#         return global2local_entity_maps
#
#     @staticmethod
#     def _add_entity_to_map(entities, g2l):
#         for entity in entities:
#             entity_global_id = entity
#             if entity_global_id not in g2l:
#                 g2l[entity_global_id] = len(g2l)


""" no subgraph """
# class MetaQaGraphMemDataLoader():
#     def __init__(self, data_file, subgraph_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation, batch_size):
#         self.use_inverse_relation = use_inverse_relation
#         self.max_local_entity = 0
#         self.max_facts = 0
#         self.max_query_word = max_query_word
#         self.count_max_query_word = 0
#         self.batch_size = batch_size
#
#         if self.use_inverse_relation:
#             self.num_kb_relation = 2 * len(relation2id)
#         else:
#             self.num_kb_relation = len(relation2id)
#
#         print('loading data from', data_file)
#         with open(data_file+".pkl", 'rb') as f:
#             (self.sample2topic_entity, self.data) = pickle.load(f)
#
#         (triplet_ids, kb_adj_mats, kb_fact_rels) = subgraph_file
#
#         self.max_facts = len(triplet_ids) * 2
#
#         print('max_facts: ', self.max_facts)
#         self.num_data = len(self.data)
#         self.batches = np.arange(self.num_data)
#
#         print('building word index ...')
#         self.word2id = word2id
#         self.relation2id = relation2id
#         self.entity2id = entity2id
#
#         self.id2entity = {i: entity for entity, i in entity2id.items()}
#
#         print('converting global to local entity index ...')
#         self.max_local_entity = len(entity2id)
#         print('max_local_entity', self.max_local_entity)
#
#         print('preparing data ...')
#         # local entity id -> global entity id
#         self.local_entities = np.arange(self.max_local_entity)
#         self.local_entities = np.tile(np.reshape(self.local_entities, (1, -1)), (batch_size, 1))
#
#         self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
#         self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
#
#         self.kb_adj_mats = kb_adj_mats
#         self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
#         self.kb_fact_rels = np.tile(kb_fact_rels, (batch_size, 1))
#
#         self._prepare_data()
#         print("max query word count: {}".format(self.count_max_query_word))
#         print("build kb adj mat ...")
#         kb_adj_data = self._build_kb_adj_mat()
#         print("build batch kb adj data")
#         self.batch_kbadj_data = self._build_batch_kb_adj_mat(kb_adj_data)
#
#     def _preprocess_data(self):
#         processed_data = []
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             question, answers = sample.strip().split('\t')
#             answers = answers.split("|")
#
#             topic_entity = question[question.find('[')+1: question.find(']')]
#
#             question = re.sub('[\[!@#$,.?!()\]]', '', question)
#             question = question.replace("'s", "")
#             question = question.replace("'", "")
#             question = question.replace("-", " ")
#             tokens = question.split(' ')
#
#             processed_data.append((tokens, answers, topic_entity))
#         self.data = processed_data
#
#     def _prepare_data(self):
#         """
#         global2local_entity_maps: a map from global entity id to local entity id
#         adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
#         """
#         next_id = 0
#         count_query_length = [0] * 50
#         total_miss_answer = 0
#
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             (tokens, answers, topic_entity) = sample
#             topic_entity_id = self.entity2id[topic_entity]
#
#             # build connection between question and entities in it
#             self.q2e_adj_mats[next_id, topic_entity_id, 0] = 1.0
#
#             # tokenize question
#             self.count_max_query_word = max(self.count_max_query_word, len(tokens))
#             for j, word in enumerate(tokens):
#                 if j < self.max_query_word:
#                     if word in self.word2id:
#                         self.query_texts[next_id, j] = self.word2id[word]
#                     else:
#                         self.query_texts[next_id, j] = self.word2id['__unk__']
#
#             # construct distribution for answers
#             miss_answer = 0
#             for answer in answers:
#                 self.answer_dists[next_id, self.entity2id[answer]] = 1.0
#
#             next_id += 1
#
#     def _build_kb_adj_mat(self):
#         """Create sparse matrix representation for batched data"""
#         mats0_batch = np.array([], dtype=int)
#         mats0_0 = np.array([], dtype=int)
#         mats0_1 = np.array([], dtype=int)
#         vals0 = np.array([], dtype=float)
#
#         mats0_rev_batch = np.array([], dtype=int)
#         mats0_rev_0 = np.array([], dtype=int)
#         mats0_rev_1 = np.array([], dtype=int)
#         vals0_rev = np.array([], dtype=float)
#
#         mats1_batch = np.array([], dtype=int)
#         mats1_0 = np.array([], dtype=int)
#         mats1_1 = np.array([], dtype=int)
#         vals1 = np.array([], dtype=float)
#
#         mats2_batch = np.array([], dtype=int)
#         mats2_0 = np.array([], dtype=int)
#         mats2_1 = np.array([], dtype=int)
#         vals2 = np.array([], dtype=float)
#
#         mats3_batch = np.array([], dtype=int)
#         mats3_0 = np.array([], dtype=int)
#         mats3_1 = np.array([], dtype=int)
#         vals3 = np.array([], dtype=float)
#
#         mats4_batch = np.array([], dtype=int)
#         mats4_0 = np.array([], dtype=int)
#         mats4_1 = np.array([], dtype=int)
#         vals4 = np.array([], dtype=float)
#
#         for i, sample_id in enumerate(range(self.batch_size)):
#
#             (mat0_0, mat0_1, val0), (mat0_rev_0, mat0_rev_1, val0_rev), \
#             (mat1_0, mat1_1, val1), (mat2_0, mat2_1, val2), \
#             (mat3_0, mat3_1, val3), (mat4_0, mat4_1, val4) = self.kb_adj_mats
#             # assert len(val0) == len(val1)
#             num_fact = len(val0)
#             num_ent = len(val2)
#             # mat0
#             mats0_batch = np.append(mats0_batch, np.full(num_fact, i, dtype=int))
#             mats0_0 = np.append(mats0_0, mat0_0)
#             mats0_1 = np.append(mats0_1, mat0_1)
#             vals0 = np.append(vals0, val0)
#
#             # mat0_rev
#             mats0_rev_batch = np.append(mats0_rev_batch, np.full(num_fact, i, dtype=int))
#             mats0_rev_0 = np.append(mats0_rev_0, mat0_rev_0)
#             mats0_rev_1 = np.append(mats0_rev_1, mat0_rev_1)
#             vals0_rev = np.append(vals0_rev, val0_rev)
#
#
#             # mat1
#             mats1_batch = np.append(mats1_batch, np.full(len(val1), i, dtype=int))
#             mats1_0 = np.append(mats1_0, mat1_0)
#             mats1_1 = np.append(mats1_1, mat1_1)
#             vals1 = np.append(vals1, val1)
#
#             # mat2
#             mats2_batch = np.append(mats2_batch, np.full(num_ent, i, dtype=int))
#             mats2_0 = np.append(mats2_0, mat2_0)
#             mats2_1 = np.append(mats2_1, mat2_1)
#             vals2 = np.append(vals2, val2)
#
#             # mat3
#             mats3_batch = np.append(mats3_batch, np.full(num_fact, i, dtype=int))
#             mats3_0 = np.append(mats3_0, mat3_0)
#             mats3_1 = np.append(mats3_1, mat3_1)
#             vals3 = np.append(vals3, val3)
#
#             # mat4
#             mats4_batch = np.append(mats4_batch, np.full(num_fact, i, dtype=int))
#             mats4_0 = np.append(mats4_0, mat4_0)
#             mats4_1 = np.append(mats4_1, mat4_1)
#             vals4 = np.append(vals4, val4)
#
#         return (mats0_batch, mats0_0, mats0_1, vals0), \
#                (mats0_rev_batch, mats0_rev_0, mats0_rev_1, vals0_rev), \
#                (mats1_batch, mats1_0, mats1_1, vals1), \
#                (mats2_batch, mats2_0, mats2_1, vals2), \
#                (mats3_batch, mats3_0, mats3_1, vals3), \
#                (mats4_batch, mats4_0, mats4_1, vals4)
#
#     def _build_batch_kb_adj_mat(self, kb_adj_data):
#         batch_size = self.batch_size
#         max_fact = self.max_facts
#         max_local_entity = self.max_local_entity
#         num_relation = self.num_kb_relation
#
#         # build kb_adj_matrix from sparse matrix
#         (e2f_batch, e2f_f, e2f_e, e2f_val), \
#         (e2f_rev_batch, e2f_rev_f, e2f_rev_e, e2f_rev_val), \
#         (f2e_batch, f2e_e, f2e_f, f2e_val), \
#         (e2e_batch, e2e_h, e2e_t, e2e_val), \
#         (r2f_batch, r2f_f, r2f_r, r2f_val), \
#         (f2r_batch, f2r_r, f2r_f, f2r_val), \
#             = kb_adj_data
#
#         entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
#         entity2fact_val = torch.FloatTensor(e2f_val)
#         # [batch_size, max_fact, max_local_entity]
#         fact2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
#             [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity
#
#         entity2fact_rev_index = torch.LongTensor([e2f_rev_batch, e2f_rev_f, e2f_rev_e])
#         entity2fact_rev_val = torch.FloatTensor(e2f_rev_val)
#         # [batch_size, max_fact, max_local_entity]
#         fact2entity_rev_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_rev_index, entity2fact_rev_val, torch.Size(
#             [batch_size, max_fact, max_local_entity])))  # batch_size, max_fact, max_local_entity
#
#         fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
#         fact2entity_val = torch.FloatTensor(f2e_val)
#         # [batch_size, max_local_entity, max_fact]
#         entity2fact_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val,
#                                                             torch.Size([batch_size, max_local_entity, max_fact])))
#
#         entity2entity_index = torch.LongTensor([e2e_batch, e2e_h, e2e_t])
#         entity2entity_val = torch.FloatTensor(e2e_val)
#         # [batch_size, max_fact, max_local_entity]
#         entity2entity_mat = use_cuda(torch.sparse.FloatTensor(entity2entity_index,
#                                                               entity2entity_val,
#                                                               torch.Size([batch_size, max_local_entity,
#                                                                           max_local_entity])))  # batch_size, max_fact, max_local_entity
#
#         relation2fact_index = torch.LongTensor([r2f_batch, r2f_f, r2f_r])
#         relation2fact_val = torch.FloatTensor(r2f_val)
#         # [batch_size, num_relation, max_fact]
#         relation2fact_mat = use_cuda(torch.sparse.FloatTensor(relation2fact_index, relation2fact_val, torch.Size(
#             [batch_size, num_relation + 1, max_fact])))
#
#         fact2relation_index = torch.LongTensor([f2r_batch, f2r_r, f2r_f])
#         fact2relation_val = torch.FloatTensor(f2r_val)
#         # [batch_size, max_local_entity, max_fact]
#         fact2relation_mat = use_cuda(torch.sparse.FloatTensor(fact2relation_index, fact2relation_val,
#                                                               torch.Size([batch_size, max_fact, num_relation + 1])))
#
#         return (fact2entity_mat, fact2entity_rev_mat, entity2fact_mat, entity2entity_mat, relation2fact_mat, fact2relation_mat)
#
#
#     def reset_batches(self, is_sequential=True):
#         if is_sequential:
#             self.batches = np.arange(self.num_data)
#         else:
#             self.batches = np.random.permutation(self.num_data)
#
#     def get_batch(self, iteration, batch_size, fact_dropout):
#         """
#         *** return values ***
#         :local_entity: global_id of each entity (batch_size, max_local_entity)
#         :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
#         :query_text: a list of words in the query (batch_size, max_query_word)
#         :rel_document_ids: (batch_size, max_relevant_doc)
#         :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
#         """
#         sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
#         return self.local_entities, \
#                self.q2e_adj_mats[sample_ids], \
#                self.batch_kbadj_data, \
#                self.kb_fact_rels, \
#                self.query_texts[sample_ids], \
#                self.answer_dists[sample_ids]
#
#     def _build_global2local_entity_maps(self):
#         """Create a map from global entity id to local entity of each sample"""
#         # global2local_entity_maps = [None] * self.num_data
#         global2local_entity_maps = dict()
#         next_id = 0
#         for sample in tqdm(self.data):
#             (tokens, answers, topic_entity) = sample
#             topic_entity_id = self.entity2id[topic_entity]
#             if topic_entity_id in global2local_entity_maps.keys():
#                 continue
#             subgraph = self.subgraphs[topic_entity_id]
#             subgraph_entities, subgraph_triplets = subgraph
#             g2l = dict()
#             self._add_entity_to_map({self.entity2id[topic_entity]}, g2l)
#             # construct a map from global entity id to local entity id
#             self._add_entity_to_map(subgraph_entities, g2l)
#
#             # global2local_entity_maps[next_id] = g2l
#             global2local_entity_maps[topic_entity_id] = g2l
#             self.max_local_entity = max(self.max_local_entity, len(g2l))
#             next_id += 1
#         return global2local_entity_maps
#
#     @staticmethod
#     def _add_entity_to_map(entities, g2l):
#         for entity in entities:
#             entity_global_id = entity
#             if entity_global_id not in g2l:
#                 g2l[entity_global_id] = len(g2l)

""" no subgraph + preload adj """
class MetaQaGraphMemDataLoader():
    def __init__(self, data_file, subgraph_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation, batch_size, batch_kbadj_data):
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
        with open(data_file+".pkl", 'rb') as f:
            (self.sample2topic_entity, self.data) = pickle.load(f)

        (triplet_ids, kb_adj_mats, kb_fact_rels) = subgraph_file

        self.max_facts = len(triplet_ids) * 2

        print('max_facts: ', self.max_facts)
        self.num_data = len(self.data)
        self.batches = np.arange(self.num_data)

        print('building word index ...')
        self.word2id = word2id
        self.relation2id = relation2id
        self.entity2id = entity2id

        self.id2entity = {i: entity for entity, i in entity2id.items()}

        print('converting global to local entity index ...')
        self.max_local_entity = len(entity2id)
        print('max_local_entity', self.max_local_entity)

        print('preparing data ...')
        # local entity id -> global entity id
        self.local_entities = np.arange(self.max_local_entity)
        self.local_entities = np.tile(np.reshape(self.local_entities, (1, -1)), (batch_size, 1))

        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)

        self.kb_adj_mats = kb_adj_mats
        self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=float)
        self.kb_fact_rels = np.tile(kb_fact_rels, (batch_size, 1))

        self._prepare_data()
        print("max query word count: {}".format(self.count_max_query_word))

        self.batch_kbadj_data = copy.copy(batch_kbadj_data)

    def _preprocess_data(self):
        processed_data = []
        for sample_id, sample in enumerate(tqdm(self.data)):
            question, answers = sample.strip().split('\t')
            answers = answers.split("|")

            topic_entity = question[question.find('[')+1: question.find(']')]

            question = re.sub('[\[!@#$,.?!()\]]', '', question)
            question = question.replace("'s", "")
            question = question.replace("'", "")
            question = question.replace("-", " ")
            tokens = question.split(' ')

            processed_data.append((tokens, answers, topic_entity))
        self.data = processed_data

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        count_query_length = [0] * 50
        total_miss_answer = 0

        for sample_id, sample in enumerate(tqdm(self.data)):
            (tokens, answers, topic_entity) = sample
            topic_entity_id = self.entity2id[topic_entity]

            # build connection between question and entities in it
            self.q2e_adj_mats[next_id, topic_entity_id, 0] = 1.0

            # tokenize question
            self.count_max_query_word = max(self.count_max_query_word, len(tokens))
            for j, word in enumerate(tokens):
                if j < self.max_query_word:
                    if word in self.word2id:
                        self.query_texts[next_id, j] = self.word2id[word]
                    else:
                        self.query_texts[next_id, j] = self.word2id['__unk__']

            # construct distribution for answers
            miss_answer = 0
            for answer in answers:
                self.answer_dists[next_id, self.entity2id[answer]] = 1.0

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
        return self.local_entities, \
               self.q2e_adj_mats[sample_ids], \
               self.batch_kbadj_data, \
               self.kb_fact_rels, \
               self.query_texts[sample_ids], \
               self.answer_dists[sample_ids]

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        # global2local_entity_maps = [None] * self.num_data
        global2local_entity_maps = dict()
        next_id = 0
        for sample in tqdm(self.data):
            (tokens, answers, topic_entity) = sample
            topic_entity_id = self.entity2id[topic_entity]
            if topic_entity_id in global2local_entity_maps.keys():
                continue
            subgraph = self.subgraphs[topic_entity_id]
            subgraph_entities, subgraph_triplets = subgraph
            g2l = dict()
            self._add_entity_to_map({self.entity2id[topic_entity]}, g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(subgraph_entities, g2l)

            # global2local_entity_maps[next_id] = g2l
            global2local_entity_maps[topic_entity_id] = g2l
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entities, g2l):
        for entity in entities:
            entity_global_id = entity
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)

"""new"""
# class MetaQaGraphMemDataLoader():
#     def __init__(self, triplets, data_file, word2id, relation2id, entity2id, max_query_word, use_inverse_relation):
#         self.use_inverse_relation = use_inverse_relation
#         self.max_local_entity = len(entity2id)
#         self.max_facts = 0
#         self.max_query_word = max_query_word
#         self.count_max_query_word = 0
#
#         self.triplet_ids = []
#         for triplet in triplets:
#             (head, rel, tail) = triplet
#             head_id, rel_id, tail_id = entity2id[head], relation2id[rel], entity2id[tail]
#             self.triplet_ids.append((head_id, rel_id, tail_id))
#
#         if self.use_inverse_relation:
#             self.num_kb_relation = 2 * len(relation2id)
#             self.max_facts = 2 * len(triplets)
#         else:
#             self.num_kb_relation = len(relation2id)
#             self.max_facts = len(triplets)
#
#         print('loading data from', data_file)
#         self.data = []
#         with open(data_file) as f_in:
#             for line in tqdm(f_in.readlines()):
#                 self.data.append(line)
#
#         print('max_facts: ', self.max_facts)
#
#         self.num_data = len(self.data)
#         self.batches = np.arange(self.num_data)
#
#         print('building word index ...')
#         self.word2id = word2id
#         self.relation2id = relation2id
#         self.entity2id = entity2id
#
#         self.id2entity = {i: entity for entity, i in entity2id.items()}
#
#         # local entity id -> global entity id
#         self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
#         self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=bool)
#
#         self.kb_adj_mats = None
#         self.batch_adj_data = None
#         self.q2e_adj_mats = np.zeros((self.num_data, self.max_local_entity, 1), dtype=bool)
#
#         print("Preparing data...")
#         self._prepare_data()
#
#         self.data = None
#
#     def _prepare_data(self):
#         for sample_id, sample in enumerate(tqdm(self.data)):
#             question, answers = sample.strip().split('\t')
#             answers = answers.split("|")
#
#             topic_entity = question[question.find('[')+1: question.find(']')]
#             question = re.sub('[\[!@#$,.?!()\]]', '', question)
#             question = question.replace("'s", "")
#             question = question.replace("'", "")
#             question = question.replace("-", " ")
#             tokens = question.split(' ')
#
#             self.count_max_query_word = max(self.count_max_query_word, len(tokens))
#
#             # query words
#             for j, word in enumerate(tokens):
#                 if j < self.max_query_word:
#                     if word in self.word2id:
#                         self.query_texts[sample_id, j] = self.word2id[word]
#                     else:
#                         self.query_texts[sample_id, j] = self.word2id['__unk__']
#
#             # topic entity
#             self.q2e_adj_mats[sample_id, self.entity2id[topic_entity], 0] = 1
#
#             # answers
#             for answer in answers:
#                 self.answer_dists[sample_id, self.entity2id[answer]] = 1.0
#
#
#     def reset_batches(self, is_sequential=True):
#         if is_sequential:
#             self.batches = np.arange(self.num_data)
#         else:
#             self.batches = np.random.permutation(self.num_data)
#
#     def get_batch(self, iteration, batch_size, fact_dropout):
#         """
#         *** return values ***
#         :local_entity: global_id of each entity (batch_size, max_local_entity)
#         :adj_mat: adjacency matrices (batch_size, num_relation, max_local_entity, max_local_entity)
#         :query_text: a list of words in the query (batch_size, max_query_word)
#         :rel_document_ids: (batch_size, max_relevant_doc)
#         :answer_dist: an distribution over local_entity (batch_size, max_local_entity)
#         """
#         sample_ids = self.batches[batch_size * iteration: batch_size * (iteration + 1)]
#         return self.q2e_adj_mats[sample_ids], \
#                self.query_texts[sample_ids], \
#                self.answer_dists[sample_ids]