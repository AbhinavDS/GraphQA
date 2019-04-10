import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class GCN(nn.Module):
	def __init__(self, args):
		super(GCN, self).__init__()
		self.n_img_feats = args.n_img_feats
		self.relation_embedding_size = args.relation_embedding_size
		self.gcn_depth = args.gcn_depth
		self.weights_init = args.weights_init
		self.layers = nn.ModuleList()
		for i in range(self.gcn_depth):
			self.add_layer(nn.Linear(self.n_img_feats,self.n_img_feats))
			self.add_layer(nn.Linear(self.n_img_feats + self.relation_embedding_size, self.n_img_feats))
		self.a = nn.Tanh()
	
	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			if self.weights_init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				nn.init.constant_(self.layers[-1].weight,0)

	def forward(self, x, A, R):
		#x: batch_size x O x n_img_feats (O = number of objects, torch tensor)
		#A: batch_size x O x OR (Adjacency Matrix) numpy array
		#R: R x embedding_size (rel_embeddings: torch tensor)

		# Implemented as ResNet
		# Graph convolution: feature at object p = weighted feature at object p + sum of weighted features from the neighbours
		# Use relational embedding when calculating features from neighbours, instead of just x
		batch_size, objects, _ = x.size()
		num_rel = R.size(0)
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		R_repeated = R.unsqueeze(0).unsqueeze(0).repeat(batch_size, objects, 1, 1)
		for i in range(0,len(self.layers),2):
			xr = torch.cat((x.unsqueeze(2).repeat(1, 1, num_rel, 1), R_repeated), 3)
			xr = xr.reshape((batch_size, num_rel* objects, -1))
			x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](xr)) + x)
		return x

# import argparse
# args = argparse.ArgumentParser()
# args.n_img_feats = 512
# args.relation_embedding_size = 128
# args.gcn_depth = 3
# A = GCN(args).cuda()
