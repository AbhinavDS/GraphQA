"""
Module that computes the Graph Convolution using word features of objects instead of image features
"""

import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class GCN(nn.Module):
	def __init__(self, args, rel_word2vec=None, obj_name_word2vec=None):
		super(GCN, self).__init__()
		self.device = args.device
		self.obj_emb_dim = args.obj_emb_dim
		self.max_rels = args.max_rels
		self.max_num_objs = args.max_num_objs
		self.max_objs = args.max_objs
		self.rel_emb_dim = args.rel_emb_dim
		self.gcn_depth = args.gcn_depth
		self.weights_init = args.weights_init
		self.layers = nn.ModuleList()
		self.obj_emb_dim = args.obj_emb_dim

		assert (self.obj_emb_dim == self.rel_emb_dim)

		for i in range(self.gcn_depth):
			self.add_layer(nn.Linear(self.obj_emb_dim, self.obj_emb_dim))
			self.add_layer(nn.Linear(self.obj_emb_dim, self.obj_emb_dim))
		self.a = nn.Tanh()

		if rel_word2vec is not None:
			print('Using Pre-trained Word2vec for Relation')
			assert rel_word2vec.size(0) == self.max_rels
			assert rel_word2vec.size(1) == self.rel_emb_dim
			self.relation_embedding = nn.Embedding(self.max_rels, self.rel_emb_dim)
			self.relation_embedding.weight = nn.Parameter(rel_word2vec)
		else:
			self.relation_embedding = nn.Embedding(self.max_rels, self.rel_emb_dim)
		
		if obj_name_word2vec is not None:
			print('Using Pre-trained Word2vec for Object Names')
			assert obj_name_word2vec.size(0) == self.max_objs
			assert obj_name_word2vec.size(1) == self.obj_emb_dim
			self.obj_name_embedding = nn.Embedding(self.max_objs, self.obj_emb_dim)
			self.obj_name_embedding.weight = nn.Parameter(obj_name_word2vec)
		else:
			self.obj_name_embedding = nn.Embedding(self.max_objs, self.obj_emb_dim)

	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			if self.weights_init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				nn.init.constant_(self.layers[-1].weight,0)

	def forward(self, obj_feats, obj_rels):
		
		#obj_rels: batch_size X O X O (Adjacency matrix with relation ids instead of 1s and 0s)
		# Implemented as ResNet
		# Graph convolution: feature at object p = weighted feature at object p + sum of weighted features from the neighbours
		# Use relational embedding when calculating features from neighbours, instead of just x

		batch_size, objects, _ = obj_feats.size()

		x = torch.LongTensor(list(obj_id for obj_id in range(self.max_num_objs))).to(self.device)
		x = x.repeat(batch_size).view(batch_size, self.max_num_objs)
		x = self.obj_name_embedding(x)

		rel_embed = self.relation_embedding(obj_rels.long())
		
		A = (obj_rels > 0).float()
		
		denom = A.sum(dim=2, keepdim=False)
		denom[denom == 0] = 1
		denom = denom.unsqueeze(2)

		for i in range(0,len(self.layers),2):
			xr = x.repeat(1, objects, 1).view(batch_size, objects, objects, -1)
			xr = self.layers[i+1](torch.mul(xr, rel_embed)).permute(0, 2, 1, 3).contiguous().view(batch_size, objects, -1)
			bp = torch.bmm(A, xr).view(batch_size, objects, objects, -1).sum(dim=2, keepdim=False)
			x = self.a(self.layers[i](x) + torch.div(bp, denom) + x)
		
		return torch.cat([x, obj_feats], 2)

