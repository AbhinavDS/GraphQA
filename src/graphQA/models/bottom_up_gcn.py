"""
Module that builds the Bottom Up Attention Model using Graph Convolutional Networks (GCN)
"""

import torch 
from torch import nn as nn

from ..modules.gcn import GCN
from ..modules.non_linearity import NonLinearity
from ..modules.attention import TopDownAttention

from question_parse.models.parser import Seq2seqParser

class BottomUpGCN(nn.Module):

	def __init__(self, args):

		super(BottomUpGCN, self).__init__()
		
		self.gcn = GCN(args)
		#self.ques_encoder = Seq2seqParser(args).seq2seq.encoder
		self.attn_layer = TopDownAttention(args)
		self.nl = args.nl
		self.ques_gate = NonLinearity(args.n_ques_emb, args.n_qi_gate, self.nl)
		self.img_gate = NonLinearity(args.n_img_feats, args.n_qi_gate, self.nl)
		self.ans_gate = NonLinearity(args.n_qi_gate, args.n_ans_gate, self.nl)
		self.ans_linear = nn.Linear(args.n_ans_gate, args.n_ans)

		self.max_ques_len = args.max_ques_len
		self.max_rels = args.max_rels
		self.max_num_objs = args.max_num_objs
		self.bidirectional = args.bidirectional

	def forward(self, img_feats, ques, objs, adj_mat, rels, ques_lens, num_obj):

		"""
		@param img_feats: The image features for the corresponding image of each sample. (batch_size, n_channels, width, height)
		@param ques: Tokenized question vector for each sample. (batch_size, max_ques_len)
		@param objs: List of Object coordinates for each object in the image for each sample.(batch_size, max_num_obj, 4)
		@param adj_mat: Adjacency Matrix (batch_size, max_num_obj, max_num_obj*max_rels)
		@param rels: The vector of relations in a image for a sample. (batch_size, max_rels, rel_emb_size)
		@param ques_lens: The true length of each question in the batch. (batch_size)
		@param num_obj: The number of actual objects in each image. (batch_size)
		"""

		# Obtain Object Features for the Image
		# TODO: write the RoI function to extract object features
		# obj_feats = roi(img_feats)

		gcn_obj_feats = self.gcn(obj_feats, adj_mat, rels)

		# Obtain Question Embedding
		#ques_output, ques_hidden = self.ques_encoder(ques, ques_lens)

		if args.bidirectional:
			ques_emb = torch.cat([ques_hidden[-2, :, :], ques_hidden[-1, :, :]], 1)
		else:
			ques_emb = ques_hidden[-1, :, :]

		# Obtain the attented image feature
		attn_img_feats = self.attn_layer(gcn_obj_feats, ques_emb, num_obj)

		# Use the attented image feature to obtain a probability distribution over the possible set of answers

		gated_ques = self.ques_gate(ques_emb)
		gated_img = self.img_gate(attn_img_feats)

		combined_feats = torch.mul(gated_ques, gated_img)

		ans_distrib = self.ans_linear(self.ans_gate(combined_feats))

		return ans_distrib

	def save_model(self):
		pass
		