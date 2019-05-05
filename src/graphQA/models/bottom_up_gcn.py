"""
Module that builds the Bottom Up Attention Model using Graph Convolutional Networks (GCN)
"""

import torch 
from torch import nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn


from ..modules.gcn import GCN
from ..modules.gcn_relation import GCN as GCNRelation
from ..modules.gcn_rel_words import GCN as GCNRelWords
from ..modules.non_linearity import NonLinearity
from ..modules.attention import TopDownAttention

from question_parse.models.encoder import Encoder as QuesEncoder
from roi_pooling.functions.roi_pooling import roi_pooling_2d
import utils.utils as utils

class BottomUpGCN(nn.Module):

	def __init__(self, args, word2vec=None, rel_word2vec=None, obj_name_word2vec=None):

		super(BottomUpGCN, self).__init__()
		
		self.img_gate = NonLinearity(args.n_img_feats, args.n_qi_gate, args.nl,args.drop_prob)
		if args.use_rel_emb:
			self.gcn = GCNRelation(args, rel_word2vec=rel_word2vec)
		elif args.use_rel_words:
			self.gcn = GCNRelWords(args, rel_word2vec=rel_word2vec, obj_name_word2vec=obj_name_word2vec)
			self.img_gate = NonLinearity(args.n_img_feats + args.obj_emb_dim, args.n_qi_gate, args.nl,args.drop_prob)
		else:
			self.gcn = GCN(args)

		self.ques_encoder = QuesEncoder(args.ques_vocab_sz, args.max_ques_len, args.ques_word_vec_dim, args.n_ques_emb, args.n_ques_layers, input_dropout_p=args.drop_prob, dropout_p=args.drop_prob, bidirectional=args.bidirectional, variable_lengths=args.variable_lengths, word2vec=word2vec)
		self.dropout_layer = nn.Dropout(p=args.drop_prob)
		self.attn_layer = TopDownAttention(args)
		self.nl = args.nl
		self.ques_gate = NonLinearity(args.n_ques_emb, args.n_qi_gate, args.nl, args.drop_prob)
		
		self.ans_gate = NonLinearity(args.n_qi_gate, args.n_ans_gate, args.nl, args.drop_prob)
		self.ans_linear = wn(nn.Linear(args.n_ans_gate, args.n_ans))
		if args.bidirectional:
			self.ques_proj = wn(nn.Linear(2*args.n_ques_emb, args.n_ques_emb))

		self.use_img_feats = args.use_img_feats
		self.max_ques_len = args.max_ques_len
		self.max_rels = args.max_rels
		self.max_num_objs = args.max_num_objs
		self.bidirectional = args.bidirectional
		self.roi_output_size = (3,3)
		self.avg_layer = nn.AvgPool2d(self.roi_output_size)
		self.device = args.device

	def forward(self, img_feats, ques, objs, adj_mat, ques_lens, num_obj):

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
		rois = utils.batch_roiproposals(objs, self.device)# Change this later
		obj_feats = roi_pooling_2d(img_feats, rois, self.roi_output_size)#.detach()
		obj_feats = self.avg_layer(obj_feats).view(objs.size(0), objs.size(1), -1)
		
		gcn_obj_feats = self.gcn(obj_feats, adj_mat)

		# Obtain Question Embedding
		ques_output, (ques_hidden, _) = self.ques_encoder(ques, ques_lens)
		
		if self.bidirectional:
			ques_emb = torch.cat([ques_hidden[-2, :, :], ques_hidden[-1, :, :]], 1)
			ques_emb = self.dropout_layer(self.ques_proj(ques_emb))
		else:
			ques_emb = ques_hidden[-1, :, :]

		if self.use_img_feats:
			# Obtain the attented image feature
			attn_img_feats = self.attn_layer(gcn_obj_feats, ques_emb, num_obj, img_feats)
		else:
			attn_img_feats = self.attn_layer(gcn_obj_feats, ques_emb, num_obj)

		# Use the attented image feature to obtain a probability distribution over the possible set of answers

		gated_ques = self.ques_gate(ques_emb)
		gated_img = self.img_gate(attn_img_feats)

		combined_feats = torch.mul(gated_ques, gated_img)

		ans_distrib = self.ans_linear(self.ans_gate(combined_feats))

		return ans_distrib
		