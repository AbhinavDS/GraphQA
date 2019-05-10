"""
Module that controls the training of the graphQA module using Reinforcement Learning
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter

from .models.bottom_up_gcn import BottomUpGCN
from .models.san import SAN
from .models.learner import Learner
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F

class Evaluator:

	def __init__(self, args, dataset):

		self.args = args
		self.num_epochs = args.num_epochs
		self.dataset = dataset
		
		# Set the Model variable to the class that needs to be used
		if args.use_san:
			Model = SAN
		else:
			Model = BottomUpGCN

		self.model = Learner(Model, args)
		self.device = self.args.device
		self.load_ckpt()
		
		self.data_loader = DataLoader(dataset=self.dataset, batch_size=self.args.bsz, num_workers=4)

		self.get_preds = self.args.get_preds

	def eval(self):

		print('Initiating Evaluation')
		self.model.to(self.device)

		loss = 0.0
		accuracies = []

		if self.get_preds:
			preds_list = []
		else:
			preds_list = None

		if self.args.opt_met:
			valid_total, plausible_total, samples = 0.0, 0.0, 0

		self.model.eval()
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
			ques_ids = batch['ques_id'].to(self.device)[sorted_indices]
			obj_wrds = batch['obj_wrds'].to(self.device)[sorted_indices]
			valid_ans = batch['valid_ans'][sorted_indices].to(self.device)
			plausible_ans = batch['plausible_ans'][sorted_indices].to(self.device)
			
			ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)


			policy_distrib, value, ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)
			
			valid_batch, plausible_batch, sz = self.compute_metrics(ans_distrib, valid_ans, plausible_ans)
			samples += sz
			valid_total += valid_batch
			plausible_total += plausible_batch
			
			accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

			if self.get_preds:
				preds_list += self.extract_preds(ans_distrib.detach().cpu().numpy(), ques_ids.cpu().numpy())

			acc = np.mean(accuracies)
			print("After Batch: {}, Evaluation Accuracy: {}".format(i+1, acc))

			print('Validity: {}, Plausibility: {}'.format(float(valid_total/samples), float(plausible_total/samples)))

		acc = np.mean(accuracies)
		print("Evaluation Accuracy: {}".format(acc))
		self.write_stats(acc, preds_list)

		if self.args.opt_met:
			print('Validity: {}, Plausibility: {}'.format(float(valid_total/samples), float(plausible_total/samples)))

		acc = np.mean(accuracies)
		print("Evaluation Accuracy: {}".format(acc))
		self.write_stats(acc, preds_list)

	def get_accuracy(self, preds, correct):

		"""
		Compute the average accuracy of predictions wrt correct
		"""
		
		pred_ids = np.argmax(preds.detach().cpu().numpy(), axis = -1)

		if self.args.criterion == "xce":
			correct_ids = correct.cpu().numpy()
		else:
			raise("Incorrect Loss function to compute accuracy for")

		acc = np.equal(pred_ids.reshape(-1), correct_ids)
		return acc
	
	def compute_metrics(self, preds, valid_ans, plausible_ans):

		"""
		Compute the metric values which are being optimized
		"""

		# Get the predictions from probability distributions
		pred_ids = np.argmax(preds.detach().cpu().numpy(), axis = -1)
		valid_total = 0
		plausible_total = 0
		valid_ans = valid_ans.detach().cpu().numpy()
		plausible_ans = plausible_ans.detach().cpu().numpy()

		sz = len(pred_ids)
		for i in range(sz):

			if valid_ans[i][pred_ids[i]] == 1:
				valid_total += 1

			if plausible_ans[i][pred_ids[i]] == 1:
				plausible_total += 1

		return valid_total, plausible_total, sz

	def load_ckpt(self):
		"""
		Load the model checkpoint from the provided path
		"""

		# TODO: Maybe load args as well from the checkpoint

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.args.ckpt_dir, '{}.ckpt'.format(model_name))

		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		self.model.load_state_dict(ckpt['state_dict'])

	def extract_preds(self, ans_distrib, ques_ids):

		preds_list = []
		pred_ids = np.argmax(ans_distrib, axis = -1)

		for j in range(len(ques_ids)):
			
			answer_text = self.dataset.vocab['answer_idx_to_token'][pred_ids[j]]

			pred_obj = {
				'questionId': self.dataset.questions_keys[ques_ids[j]],
				'prediction': answer_text
			}

			preds_list.append(pred_obj)

		return preds_list

	def write_stats(self, acc, preds_list=None):

		stats_file = os.path.join(self.args.expt_res_dir, 'test_stats.json')
		stats = {'acc' : acc}

		with open(stats_file, 'w') as f:
			json.dump(stats, f, indent=4)

		if self.get_preds:
			with open(os.path.join(self.args.expt_res_dir, 'test_preds.json'), 'w') as f:
				json.dump(preds_list, f)