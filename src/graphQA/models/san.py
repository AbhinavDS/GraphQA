"""
Model declaration of the Stacked Attention Network
"""

"""
Module that builds the Bottom Up Attention Model using Graph Convolutional Networks (GCN)
"""

import torch 
from torch import nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn

from ..modules.gcn import GCN
from ..modules.gcn_relation import GCN as GCNRelation
from ..modules.gcn_rel_words import GCN as GCNRelWords
from ..modules.stacked_attention import StackedAttention
from ..modules.non_linearity import NonLinearity

from question_parse.models.encoder import Encoder as QuesEncoder
from roi_pooling.functions.roi_pooling import roi_pooling_2d
import utils.utils as utils

class SAN(nn.Module):

	def __init__(self, args, word2vec=None, rel_word2vec=None, obj_name_word2vec=None):

		super(SAN, self).__init__()
		
		self.san_dim_in = args.san_dim_in
		self.san_dim_mid = args.san_dim_mid
		self.drop_prob = args.drop_prob
		self.device = args.device
		self.n_attn_layers = args.n_attn_layers
		self.use_img_feats = args.use_img_feats
		self.max_ques_len = args.max_ques_len
		self.max_rels = args.max_rels
		self.max_num_objs = args.max_num_objs
		self.bidirectional = args.bidirectional
		self.roi_output_size = (3,3)
		self.avg_layer = nn.AvgPool2d(self.roi_output_size)
		self.use_rel_words = args.use_rel_words

		if args.use_rel_emb:
			self.gcn = GCNRelation(args, rel_word2vec=rel_word2vec)
			obj_feats_sz = args.obj_emb_dim
		elif args.use_rel_words:
			self.gcn = GCNRelWords(args, rel_word2vec=rel_word2vec, obj_name_word2vec=obj_name_word2vec)
			if args.use_blind:
				obj_feats_sz = args.obj_emb_dim
			else:
				obj_feats_sz = args.obj_emb_dim + args.n_img_feats
		else:
			obj_feats_sz = args.n_img_feats
			self.gcn = GCN(args)

		if self.use_img_feats:
			self.img_size = (args.pool_w, args.pool_h)
			self.avg_pool = nn.AvgPool2d(self.img_size)
			obj_feats_sz += args.n_img_feats

		self.ques_encoder = QuesEncoder(args.ques_vocab_sz, args.max_ques_len, args.ques_word_vec_dim, args.n_ques_emb, args.n_ques_layers, input_dropout_p=args.drop_prob, dropout_p=args.drop_prob, bidirectional=args.bidirectional, variable_lengths=args.variable_lengths, word2vec=word2vec)

		self.dropout_layer = nn.Dropout(p=args.drop_prob)
		self.nl = args.nl

		self.obj_proj = NonLinearity(obj_feats_sz, self.san_dim_in, self.nl, self.drop_prob)
		self.attn_layers = nn.ModuleList([StackedAttention(args, self.san_dim_in, self.san_dim_mid, self.drop_prob, self.device)] * self.n_attn_layers)
		
		# Can be replaced by changing the output dimension of the Question Encoder itself. Discuss this
		if args.bidirectional:
			self.ques_proj = NonLinearity(2*args.n_ques_emb, self.san_dim_in, self.nl, self.drop_prob)
		else:
			self.ques_proj = NonLinearity(args.n_ques_emb, self.san_dim_in, self.nl, self.drop_prob)

		self.ans_linear = wn(nn.Linear(args.san_dim_in, args.n_ans))

	def forward(self, img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds, obj_region_mask):

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
		batch_sz, max_num_objs, _ = obj_feats.size()
		
		if self.use_rel_words:
			gcn_obj_feats = self.gcn(obj_feats, obj_wrds, adj_mat)	
		else:
			gcn_obj_feats = self.gcn(obj_feats, adj_mat)

		# Obtain Question Embedding
		ques_output, (ques_hidden, _) = self.ques_encoder(ques, ques_lens)
		
		if self.bidirectional:
			ques_emb = torch.cat([ques_hidden[-2, :, :], ques_hidden[-1, :, :]], 1)
			ques_emb = self.dropout_layer(self.ques_proj(ques_emb))
		else:
			ques_emb = self.dropout_layer(self.ques_proj(ques_hidden[-1, :, :]))

		if self.use_img_feats:
			avg_img_feats = self.avg_pool(img_feats).view(batch_sz, -1).repeat(1, self.max_num_objs).reshape(batch_sz, self.max_num_objs, -1)
			gcn_obj_feats = torch.cat([gcn_obj_feats, avg_img_feats], 2)
		
		gcn_obj_feats = self.obj_proj(gcn_obj_feats)

		obj_mask = torch.zeros(batch_sz, self.max_num_objs).type(torch.ByteTensor).to(self.device)

		for i in range(self.max_num_objs):
			obj_mask[:, i] = (i >= num_obj)

		for attn_layer in self.attn_layers:
			ques_emb, pred_attn_mask, attn_wt = attn_layer(gcn_obj_feats, ques_emb, obj_mask, obj_region_mask)

		ans_output = self.ans_linear(self.dropout_layer(ques_emb))

		return ans_output, pred_attn_mask, attn_wt