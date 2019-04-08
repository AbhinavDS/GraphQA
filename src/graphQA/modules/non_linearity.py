"""
Module that implements the gated tanh function that is repeatedly used in the attention and question answering module
"""

import torch 
from torch import nn

class NonLinearity(nn.Module):

	def __init__(self, n_inp, n_out):
		super(NonLinearity, self).__init__()

		self.nl = args.nl
		if args.nl == 'gated_tanh':
			self.t_layer = nn.Linear(n_out, n_inp)
			self.g_layer = nn.Linear(n_out, n_inp)
		elif args.nl == 'relu':
			self.layer = nn.Linear(n_out, n_inp)
		else:
			raise('Invalid Non Linearity Specified')

	def forward(self, inp):
		if args.nl == 'gated_tanh':
			y = nn.Tanh(self.t_layer(inp))
			g = nn.Sigmoid(self.g_layer(inp))
			return torch.mul(y, g)
		elif args.nl == 'relu':
			return nn.ReLU(self.layer)
		


