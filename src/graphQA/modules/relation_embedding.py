import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class RelationEmbedding(nn.Module):
	def __init__(self, embedding_size, embedding_vocabulary):
		super(RelationEmbedding, self).__init__()
		self.embedding_size = embedding_size
		self.embedding_vocabulary = embedding_vocabulary
		# TODO: everything
