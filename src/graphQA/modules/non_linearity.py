"""
Module that implements the gated tanh function that is repeatedly used in the attention and question answering module
"""

import torch 
from torch import nn
from torch.nn.utils.weight_norm import weight_norm as wn

class NonLinearity(nn.Module):

	def __init__(self, n_inp, n_out, nl, drop_prob):
		super(NonLinearity, self).__init__()

		self.nl = nl
		self.dropout = nn.Dropout(p=drop_prob)
		if self.nl == 'gated_tanh':
			self.tanh = nn.Tanh()
			self.sigmoid = nn.Sigmoid()
			self.t_layer = wn(nn.Linear(n_inp, n_out))
			self.g_layer = wn(nn.Linear(n_inp, n_out))
		elif self.nl == 'relu':
			self.nl_layer = nn.ReLU()
			self.layer = wn(nn.Linear(n_inp, n_out))
		elif self.nl == 'tanh':
			self.nl_layer = nn.Tanh()
			self.layer = wn(nn.Linear(n_inp, n_out))
		else:
			raise('Invalid Non Linearity Specified')

	def forward(self, inp):
		if self.nl == 'gated_tanh':
			y = self.tanh(self.dropout(self.t_layer(inp)))
			g = self.sigmoid(self.dropout(self.g_layer(inp)))
			return torch.mul(y, g)
		else:
			return self.nl_layer(self.dropout(self.layer(inp)))
		


