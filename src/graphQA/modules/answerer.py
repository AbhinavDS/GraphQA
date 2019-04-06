"""
Module that generates the final probability distribution over the possible set of answers
"""

import torch
import torch.nn as nn
from modules.gated_tanh import GatedTanh

class Answerer(nn.Module):

	def __init__(self, args):
		super(Answerer, self).__init__()

		ques_gate = GatedTanh(args.n_ques_emb, args.n_qi_gate)
		img_gate = GatedTanh(args.n_img_feats, args.n_qi_gate)
		ans_gate = GatedTanh(args.n_qi_gate, args.n_ans_gate)
		ans_linear = nn.Linear(args.n_ans_gate, args.n_ans)


	def forward(self, img_feats, ques_emb):

		gated_ques = ques_gate(ques_emb)
		gated_img = img_gate(img_feats)

		combined_feats = torch.mul(gated_ques, gated_img)

		ans_distrib = ans_linear(ans_gate(combined_feats))

		return ans_distrib


