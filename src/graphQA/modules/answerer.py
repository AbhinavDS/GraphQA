"""
Module that generates the final probability distribution over the possible set of answers
"""

import torch
import torch.nn as nn
from modules.gated_tanh import GatedTanh

class Answerer(nn.Module):

	def __init__(self, args):
		super(Answerer, self).__init__()

		self.ques_gate = GatedTanh(args.n_ques_emb, args.n_qi_gate)
		self.img_gate = GatedTanh(args.n_img_feats, args.n_qi_gate)
		self.ans_gate = GatedTanh(args.n_qi_gate, args.n_ans_gate)
		self.ans_linear = nn.Linear(args.n_ans_gate, args.n_ans)

	def forward(self, img_feats, ques_emb):

		gated_ques = self.ques_gate(ques_emb)
		gated_img = self.img_gate(img_feats)

		combined_feats = torch.mul(gated_ques, gated_img)

		ans_distrib = self.ans_linear(self.ans_gate(combined_feats))

		return ans_distrib


