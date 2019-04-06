import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

class RelationEmbedding(nn.Module):
	def __init__(self, embedding_size, embedding_vocabulary):
		super(RelationEmbedding, self).__init__()
		self.embedding_size = embedding_size
		self.embedding_vocabulary = embedding_vocabulary
		# TODO: everything (create embedding matrix); 

# TODO: create adjacency matrix; create pooling layer

# Code flow
# 1. pooling
# 2. adjacency matrix
# 3. relation matrix/embedding; maybe initialize with some word embedding itself?
# 4. GCN (can use attention if needed, but maybe too much)
# 5. Add whole image feature (very pixelated), gcn, average relation(embedding) for attention