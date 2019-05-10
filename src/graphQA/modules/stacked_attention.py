"""
Constructs the attention module for stacked attention network
"""

import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm as wn

class StackedAttention(nn.Module):

	def __init__(self, san_dim_in, san_dim_mid, drop_prob, device):

		super(StackedAttention, self).__init__()
		self.device = device
		self.ff_image = wn(nn.Linear(san_dim_in, san_dim_mid))
		self.ff_ques = wn(nn.Linear(san_dim_in, san_dim_mid))
		self.attn_layer = wn(nn.Linear(san_dim_mid, 1))

		self.dropout_layer = nn.Dropout(p=drop_prob)
		self.tanh = nn.Tanh()
		self.attn_softmax = nn.Softmax(dim=1)

	def forward(self, obj_feats, ques_emb, obj_mask):

		"""
		Perform attention for one layer in overall stack of attentions
		"""

		img_proj = self.ff_image(obj_feats)
		ques_proj = self.ff_ques(ques_emb).unsqueeze(dim=1)

		combined_proj = self.dropout_layer(self.tanh(img_proj + ques_proj))

		attn_layer_out = self.attn_layer(combined_proj).squeeze(2)
		attn_layer_out = attn_layer_out.data.masked_fill_(obj_mask, -float("inf"))
		attn_wt = self.attn_softmax(attn_layer_out)

		attended_feats = torch.bmm(attn_wt.unsqueeze(1), obj_feats).squeeze(1) 
		query = attended_feats + ques_emb

		return query
