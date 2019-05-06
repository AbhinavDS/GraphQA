import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from .git_fc import FCNet


class Attention(nn.Module):
	def __init__(self, v_dim, q_dim, num_hid, device):
		super(Attention, self).__init__()
		self.nonlinear = FCNet([v_dim + q_dim, num_hid])
		self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)
		self.device = device

	def forward(self, v, q, num_obj):
		"""
		v: [batch, k, vdim]
		q: [batch, qdim]
		"""
		logits = self.logits(v, q)

		batch_sz, max_num_objs, _ = v.size()
		obj_mask = torch.zeros(batch_sz, max_num_objs, 1).type(torch.ByteTensor).to(self.device)
		for i in range(max_num_objs):
			obj_mask[:, i, 0] = (i >= num_obj)
		logits = logits.data.masked_fill_(obj_mask, -float("inf"))

		w = nn.functional.softmax(logits, 1)
		return w

	def logits(self, v, q):
		num_objs = v.size(1)
		q = q.unsqueeze(1).repeat(1, num_objs, 1)
		vq = torch.cat((v, q), 2)
		joint_repr = self.nonlinear(vq)
		logits = self.linear(joint_repr)
		return logits


class NewAttention(nn.Module):
	def __init__(self, v_dim, q_dim, num_hid, device, dropout=0.2):
		super(NewAttention, self).__init__()

		self.v_proj = FCNet([v_dim, num_hid])
		self.q_proj = FCNet([q_dim, num_hid])
		self.dropout = nn.Dropout(dropout)
		self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)
		self.device = device

	def forward(self, v, q, num_obj):
		"""
		v: [batch, k, vdim]
		q: [batch, qdim]
		"""
		logits = self.logits(v, q)
		
		batch_sz, max_num_objs, _ = v.size()
		obj_mask = torch.zeros(batch_sz, max_num_objs, 1).type(torch.ByteTensor).to(self.device)
		for i in range(max_num_objs):
			obj_mask[:, i, 0] = (i >= num_obj)
		logits = logits.data.masked_fill_(obj_mask, -float("inf"))
		
		w = nn.functional.softmax(logits, 1)
		return w

	def logits(self, v, q):
		batch, k, _ = v.size()
		v_proj = self.v_proj(v) # [batch, k, qdim]
		q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
		joint_repr = v_proj * q_proj
		joint_repr = self.dropout(joint_repr)
		logits = self.linear(joint_repr)
		return logits
