"""
Module that implements the attention on the image/object features and returns the attended image features

Reference: https://arxiv.org/pdf/1707.07998.pdf
http://openaccess.thecvf.com/content_cvpr_2018/papers/Teney_Tips_and_Tricks_CVPR_2018_paper.pdf
"""

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn
from .non_linearity import NonLinearity

class TopDownAttention(nn.Module):

	def __init__(self, args):

		super(TopDownAttention, self).__init__()

		# Doubt on what to set
		self.attn_layer = wn(nn.Linear(args.n_attn, 1))
		self.attn_softmax = nn.Softmax(dim=1)
		self.device = args.device
		self.max_num_objs = args.max_num_objs
		self.use_img_feats = args.use_img_feats

		if self.use_img_feats:
			self.img_size = (args.pool_w, args.pool_h)
			self.avg_pool = nn.AvgPool2d(self.img_size)
			self.attn_gate = NonLinearity(2 * args.n_img_feats + args.n_ques_emb, args.n_attn, args.nl, args.drop_prob)
		else:
			self.attn_gate = NonLinearity(args.n_img_feats + args.n_ques_emb, args.n_attn, args.nl, args.drop_prob)

	def forward(self, obj_feats, ques_emb, num_obj, img_feats=None):

		"""
		@param obj_feats: Tensor of Image/Object Features to be attended on. Size: (B*O*F1)
		@param ques_emb: The embedding of the question which will be used to attend on the img_feats. Size: (B*F2)
		@param num_obj: Tensor for number of objects in each image for creating a mask. Size: (B)
		@return: A single image feature attended over all objects. Size: (B*F1)
		"""
		batch_sz = ques_emb.size(0)
		
		obj_mask = torch.zeros(batch_sz, self.max_num_objs).type(torch.ByteTensor).to(self.device)

		for i in range(self.max_num_objs):
			obj_mask[:, i] = (i >= num_obj)
		
		if self.use_img_feats:
			
			avg_img_feats = self.avg_pool(img_feats).view(batch_sz, -1).repeat(1, self.max_num_objs).reshape(batch_sz, self.max_num_objs, -1)

			# Computing attention
			gated_attn = self.attn_gate(torch.cat((obj_feats, avg_img_feats, ques_emb.repeat(1, self.max_num_objs).reshape(batch_sz, self.max_num_objs, -1)), 2))			

		else:
			# Computing attention
			gated_attn = self.attn_gate(torch.cat((obj_feats, ques_emb.repeat(1, self.max_num_objs).reshape(batch_sz, self.max_num_objs, -1)), 2))

		attn_layer_out = self.attn_layer(gated_attn).squeeze(2)
		
		attn_layer_out = attn_layer_out.data.masked_fill_(obj_mask, -float("inf"))
		attn_wt = self.attn_softmax(attn_layer_out)

		attn_img_feats = torch.bmm(attn_wt.unsqueeze(1), obj_feats).squeeze(1)

		return attn_img_feats





