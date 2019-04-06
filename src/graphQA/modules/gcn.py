import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class GCN(nn.Module):
	def __init__(self, params, weights_init='xavier'):
		super(GCN, self).__init__()
		self.feature_size = params.feature_size
		self.depth = params.depth
		self.weights_init = weights_init
		self.layers = nn.ModuleList()
		for i in range(self.depth):
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
			self.add_layer(nn.Linear(self.feature_size,self.feature_size))
		self.a = nn.Tanh()
	
	def add_layer(self,layer,init=True):
		self.layers.append(layer)
		if init:
			if self.weights_init == 'xavier':
				nn.init.xavier_uniform_(self.layers[-1].weight)
			else:
				nn.init.constant_(self.layers[-1].weight,0)

	def forward(self, x, A):
		#x: batch_size x O x feature_size (O = number of objects)
		#A: batch_size x O x O (Adjacency Matrix)
		temp_A = Variable(torch.Tensor(A).type(dtype),requires_grad=False)
		for i in range(0,len(self.layers),2):
			# Implemented as ResNet
			# Graph convolution: feature at object p = weighted feature at object p + sum of weighted features from the neighbours
			# Use relational embedding when calculating features from neighbours, instead of just x
			x = self.a(self.layers[i](x)+torch.bmm(temp_A,self.layers[i+1](x)) + x)
		return x

# import argparse
# params = argparse.ArgumentParser()
# params.feature_size = 128
# params.depth = 3
# A = GCN(params).cuda()