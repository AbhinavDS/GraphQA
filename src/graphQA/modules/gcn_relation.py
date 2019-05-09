import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm as wn


class GCN(nn.Module):
	def __init__(self, args, rel_word2vec=None):
		super(GCN, self).__init__()
		self.device = args.device
		self.n_img_feats = args.n_img_feats
		self.r_img_feats = args.obj_emb_dim # kept same as obj dimension
		self.max_rels = args.max_rels
		self.rel_emb_dim = args.rel_emb_dim
		self.gcn_depth = args.gcn_depth
		self.weights_init = args.weights_init
		self.layers = nn.ModuleList()

		self.reduced = wn(nn.Linear(self.n_img_feats, self.r_img_feats));
		nn.init.xavier_uniform_(self.reduced.weight);

		assert (self.gcn_depth >= 0)
		for i in range(self.gcn_depth):
			self.add_layer(wn(nn.Linear(self.r_img_feats,self.r_img_feats)))
			self.add_layer(wn(nn.Linear(self.r_img_feats + self.rel_emb_dim, self.r_img_feats)))
		self.a = nn.Tanh()
		if rel_word2vec is not None:
			print('Using Pre-trained Word2vec for Relation')
			assert rel_word2vec.size(0) == self.max_rels
			assert rel_word2vec.size(1) == self.rel_emb_dim
			self.relation_embedding = nn.Embedding(self.max_rels, self.rel_emb_dim)
			self.relation_embedding.weight = nn.Parameter(rel_word2vec)
		else:
			self.relation_embedding = nn.Embedding(self.max_rels, self.rel_emb_dim)
	
	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			if self.weights_init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				nn.init.constant_(self.layers[-1].weight,0)

	def forward(self, x, A):
		#x: batch_size x O x n_img_feats (O = number of objects, torch tensor)
		#A: batch_size x O x OR (Adjacency Matrix) numpy array

		# Implemented as ResNet
		# Graph convolution: feature at object p = weighted feature at object p + sum of weighted features from the neighbours
		# Use relational embedding when calculating features from neighbours, instead of just x

		x = self.reduced(x)
		batch_size, objects, _ = x.size()
		rel_input = torch.LongTensor(list(rel for rel in range(self.max_rels))).to(self.device)
		rel_input = rel_input.unsqueeze(0).unsqueeze(1).repeat(batch_size, objects, 1).view(batch_size, -1)
		rel_embed = self.relation_embedding(rel_input)
		for i in range(0,len(self.layers),2):
			xr = x.unsqueeze(2).repeat(1, 1, self.max_rels, 1).view(batch_size, self.max_rels * objects, -1)
			xr = torch.cat((xr, rel_embed), 2)
			x = self.a(self.layers[i](x)+torch.bmm(A,self.layers[i+1](xr)) + x)
		return x

