"""
Module that evaluates the trained model on the given test set
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter

from .models.bottom_up_gcn import BottomUpGCN
from torch.utils.data import DataLoader

class Evaluator:

	def __init__(self, args, dataset):

		self.args = args
		self.num_epochs = args.num_epochs
		self.dataset = dataset
		self.model = BottomUpGCN(args)
		self.load_ckpt()
		self.device = self.args.device		
		self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

	def eval(self):

		print('Initiating Evaluation')
		self.model.to(self.device)
				
		self.model.eval()

		loss = 0.0
		accuracies = []

		for i, batch in enumerate(self.data_loader):

			# Unpack the items from the batch tensor
			ques_lens = batch['ques_lens'].to(self.device)
			sorted_indices = torch.argsort(ques_lens, descending=True)
			ques_lens = ques_lens[sorted_indices] 
			img_feats = batch['image_feat'].to(self.device)[sorted_indices]
			ques = batch['ques'].to(self.device)[sorted_indices]
			objs = batch['obj_bboxes'].to(self.device)[sorted_indices]
			adj_mat = batch['A'].to(self.device)[sorted_indices]
			num_obj = batch['num_objs'].to(self.device)[sorted_indices] 
			ans_output = batch['ans'].to(self.device)[sorted_indices]
			ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj)

			accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

		acc = np.mean(accuracies)
		print("Evaluation Accuracy: {}".format(acc))
		self.write_stats(acc)
			
	def get_accuracy(self, preds, correct):

		"""
		Compute the average accuracy of predictions wrt correct
		"""
		
		pred_ids = np.argmax(preds.detach().cpu().numpy(), axis = -1)

		if self.args.criterion == "bce":
			# correct is in form of one hot vector
			correct_ids = np.argmax(correct.cpu().numpy(), axis = -1)
		elif self.args.criterion == "xce":
			correct_ids = correct.cpu().numpy()
		else:
			raise("Incorrect Loss function to compute accuracy for")

		acc = np.equal(pred_ids.reshape(-1), correct_ids)
		return acc
	
	def write_stats(self, acc):

		stats_file = os.path.join(self.args.expt_res_dir, 'test_stats.json')
		stats = {'acc' : acc}

		with open(stats_file, 'w') as f:
			json.dump(stats, f, indent=4)

	def load_ckpt(self):
		"""
		Load the model checkpoint from the provided path
		"""

		# TODO: Maybe load args as well from the checkpoint

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.args.ckpt_dir, '{}.ckpt'.format(model_name))

		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		self.model.load_state_dict(ckpt['state_dict'])