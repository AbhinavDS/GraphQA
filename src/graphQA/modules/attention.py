"""
Module that implements the attention on the image/object features and returns the attended image features

Reference: https://arxiv.org/pdf/1707.07998.pdf
http://openaccess.thecvf.com/content_cvpr_2018/papers/Teney_Tips_and_Tricks_CVPR_2018_paper.pdf
"""

import torch
import torch.nn as nn
from modules.gated_tanh import GatedTanh

class TopDownAttention(nn.Module):

	def __init__(self, args):

		super(TopDownAttention, self).__init__()

		# Doubt on what to set
		attn_layer = nn.Linear(args.n_attn, 1)
		attn_gate = GatedTanh(args.n_img_feats + args.n_ques_emb, args.n_attn)
		attn_softmax = nn.Softmax(dim=1)


	def forward(self, img_feats, ques_emb, num_obj, device):

		"""
		@param img_feats: Tensor of Image/Object Features to be attended on. Size: (B*O*F1)
		@param ques_emb: The embedding of the question which will be used to attend on the img_feats. Size: (B*F2)
		@param num_obj: Tensor for number of objects in each image for creating a mask. Size: (B)
		@params device: The device on which tensors need to be initialized
		@return: A single image feature attended over all objects. Size: (B*F1)
		"""

		batch_sz = ques_feats.size(0)
		max_num_objs = img_feats.size(1)

		obj_mask = torch.Tensor(batch_sz, max_num_objs).to(device)

		for i in range(max_num_objs):
			obj_mask[:, i] = (i >= num_obj)

		# Computing attention
		gated_attn = attn_gate(torch.cat((img_feats, ques_emb.repeat(1, max_num_objs).reshape(batch_sz, max_num_objs, -1)), 2))

		attn_wt = attn_softmax(attn_layer(gated_attn))
		attn_wt.data.masked_fill_(obj_mask, -float("inf"))

		return torch.bmm(attn_wt, img_feats).squeeze(1)





