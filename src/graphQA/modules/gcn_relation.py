import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class GCN(nn.Module):
	def __init__(self, args):
		super(GCN, self).__init__()
		self.n_img_feats = args.n_img_feats
		self.max_rels = args.max_rels
		self.relation_embedding_size = args.relation_embedding_size
		self.gcn_depth = args.gcn_depth
		self.weights_init = args.weights_init
		self.layers = nn.ModuleList()
		for i in range(self.gcn_depth):
			self.add_layer(nn.Linear(self.n_img_feats,self.n_img_feats))
			self.add_layer(nn.Linear(self.n_img_feats + self.relation_embedding_size, self.n_img_feats))
		self.a = nn.Tanh()
		self.relation_embedding = nn.Embedding(args.max_rels, args.relation_embedding_size)
	
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

		batch_size, objects, _ = x.size()
		rel_input = torch.LongTensor(list(rel for rel in range(self.max_rels)))
		rel_input = rel_input.unsqueeze(0).unsqueeze(1).repeat(bs, o, 1).view(bs, -1)
		rel_embed = self.relation_embedding(rel_input)
		for i in range(0,len(self.layers),2):
			xr = x.unsqueeze(2).repeat(1, 1, self.max_rels, 1).view(batch_size, self.max_rels * objects, -1)
			xr = torch.cat((xr, rel_embed), 3)
			x = self.a(self.layers[i](x)+torch.bmm(A,self.layers[i+1](xr)) + x)
		return x

