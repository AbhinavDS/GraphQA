"""
Learner Module for RL
"""

import torch 
from torch import nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn

class Learner(nn.Module):

	def __init__(self, Model, args, word2vec=None, rel_word2vec=None, obj_name_word2vec=None):
		
		super(Learner, self).__init__()
		
		self.model_1 = Model(args, word2vec=word2vec, rel_word2vec=rel_word2vec, obj_name_word2vec=obj_name_word2vec)
		self.model_2 = Model(args, word2vec=word2vec, rel_word2vec=rel_word2vec, obj_name_word2vec=obj_name_word2vec)
		self.args = args

		self.actor_layer = nn.Sequential(
					nn.ReLU(),
					nn.Linear(self.args.n_ans, self.args.n_ans)
				)
		self.softmax = nn.Softmax(dim=-1)
		self.critic_layer = nn.Sequential(
					nn.ReLU(),
					nn.Linear(self.args.n_ans, 1)
				)

	def forward(self, img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds):

		model_out_1 = self.model_1(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)
		model_out_2 = self.model_2(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)

		policy_distrib = self.actor_layer(model_out_1)
		value = self.critic_layer(model_out_1)

		return self.softmax(policy_distrib), value, policy_distrib 