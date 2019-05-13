import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils.weight_norm import weight_norm as wn

class GCN(nn.Module):
	def __init__(self, args):
		
		super(GCN, self).__init__()

		if self.args.reduce_img_feats:
			self.reduce_img_feats = self.args.reduce_img_feats
			n_img_feats = args.rel_emb_dim
		else:
			n_img_feats = args.n_img_feats

		self.n_img_feats = n_img_feats
		self.gcn_depth = args.gcn_depth
		self.weights_init = args.weights_init
		self.layers = nn.ModuleList()
		assert (self.gcn_depth >= 0)
		for i in range(self.gcn_depth):
			self.add_layer(wn(nn.Linear(self.n_img_feats, self.n_img_feats)))
			self.add_layer(wn(nn.Linear(self.n_img_feats, self.n_img_feats)))
		self.a = nn.Tanh()
	
	def add_layer(self, layer, init=True):
		self.layers.append(layer)
		if init:
			if self.weights_init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				nn.init.constant_(self.layers[-1].weight,0)

	def forward(self, x, A):
		#x: batch_size x O x n_img_feats (O = number of objects)
		#A: batch_size x O x O (Adjacency Matrix)
		for i in range(0, len(self.layers), 2):
			# Implemented as ResNet
			# Graph convolution: feature at object p = weighted feature at object p + sum of weighted features from the neighbours
			# Use relational embedding when calculating features from neighbours, instead of just x
			x = self.a(self.layers[i](x)+torch.bmm(A, self.layers[i+1](x)) + x)
		return x
