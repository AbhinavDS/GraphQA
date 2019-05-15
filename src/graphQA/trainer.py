"""
Module that controls the training of the graphQA module
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter

from .models.bottom_up_gcn import BottomUpGCN
from .models.san import SAN
from .models.git_bua import BaseModel as BottomUp
from .models.git_bua2 import BaseModel2 as BottomUp2
from torch.utils.data import DataLoader

class Trainer:

	def __init__(self, args, train_dataset, val_dataset):

		self.args = args
		self.num_epochs = args.num_epochs
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.use_glove = args.use_glove
		self.use_rel_words = args.use_rel_words
		
		# Set the Model variable to the class that needs to be used
		if args.use_san:
			Model = SAN
		elif args.use_bua:
			Model = BottomUp
		elif args.use_bua2:
			Model = BottomUp2
		else:
			Model = BottomUpGCN

		self.embeddings_mat = None
		self.rel_embeddings_mat = None
		self.obj_names_embeddings_mat = None

		if self.use_glove:
			self.embeddings_mat = self.train_dataset.embeddings_mat
			if self.args.use_rel_emb:
				self.rel_embeddings_mat = self.train_dataset.rel_embeddings_mat
			elif self.args.use_rel_words:
				self.rel_embeddings_mat = self.train_dataset.rel_embeddings_mat
				self.obj_names_embeddings_mat = self.train_dataset.obj_names_embeddings_mat
		
		self.model = Model(args, word2vec=self.embeddings_mat, rel_word2vec=self.rel_embeddings_mat, obj_name_word2vec=self.obj_names_embeddings_mat)

		self.device = self.args.device

		# Can be changed to support different optimizers
		# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.optimizer = torch.optim.Adamax(self.model.parameters())
		self.set_criterion()

		self.lambda_valid = args.lambda_valid
		self.lambda_plaus = args.lambda_plaus
		self.lambda_ground = args.lambda_ground
		self.ground_loss = nn.MSELoss()
		self.lr = self.args.lr
		
		self.train_loader = DataLoader(dataset = self.train_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

		self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

		self.log = args.log
		if self.log:
			self.writer = SummaryWriter(args.log_dir)

	def instance_bce_with_logits(self, logits, labels):
		assert logits.dim() == 2
		loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
		loss *= labels.size(1)
		return loss


	def compute_score_with_logits(self, logits, labels):
		logits = torch.max(logits, 1)[1].data # argmax
		one_hots = torch.zeros(*labels.size()).cuda()
		one_hots.scatter_(1, logits.view(-1, 1), 1)
		scores = (one_hots * labels)
		return scores

	def train(self):

		print('Initiating Training')

		# Check if the training has to be restarted
		self.check_restart_conditions()
		
		if self.resume_from_epoch >= 1:
			print('Loading from Checkpointed Model')
			# Write the logic for loading checkpoint for the model
			self.load_ckpt()

		self.model.to(self.device)
				
		for epoch in range(self.resume_from_epoch, self.num_epochs):

			lr = self.adjust_lr(epoch)
			self.model.train()

			loss = 0.0
			train_accuracies = []

			for i, batch in enumerate(self.train_loader):

				self.optimizer.zero_grad()

				# Unpack the items from the batch tensor
				ques_lens = batch['ques_lens'].to(self.device)
				sorted_indices = torch.argsort(ques_lens, descending=True)
				ques_lens = ques_lens[sorted_indices] 
				img_feats = batch['image_feat'].to(self.device)[sorted_indices]
				ques = batch['ques'].to(self.device)[sorted_indices]
				objs = batch['obj_bboxes'].to(self.device)[sorted_indices]
				adj_mat = batch['A'].to(self.device)[sorted_indices]
				num_objs = batch['num_objs'].to(self.device)[sorted_indices] 
				ans_output = batch['ans'].to(self.device)[sorted_indices]
				obj_wrds = batch['obj_wrds'].to(self.device)[sorted_indices]
				obj_region_mask = batch['obj_region_mask'].to(self.device)[sorted_indices]
				attn_mask = batch['attn_mask'].to(self.device)[sorted_indices]
				
				if self.args.use_rel_probs or self.args.use_rel_probs_sum:
					rel_prob_mat = batch['P'].to(self.device)[sorted_indices]
				else:
					rel_prob_mat = None

				if self.args.opt_met:
					valid_ans = batch['valid_ans'].to(self.device)[sorted_indices]
					plausible_ans = batch['plausible_ans'].to(self.device)[sorted_indices]

				ans_distrib, pred_attn_mask, _ = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_objs, obj_wrds, 	obj_region_mask, rel_prob_mat)
				pred_attn_mask[attn_mask <= 0] = 0

				#print(ans_distrib.size(), ans_output.size())
				if self.args.criterion == "bce" or self.args.use_bua or self.args.use_bua2:
					batch_loss = self.instance_bce_with_logits(ans_distrib, ans_output)
				else:
					batch_loss = self.criterion(ans_distrib, ans_output)

				accur_loss = (float)(batch_loss.data.detach().cpu().numpy())

				if self.args.opt_met:
					valid_loss = self.lambda_valid * self.instance_bce_with_logits(ans_distrib, valid_ans)
					plaus_loss = self.lambda_plaus * self.instance_bce_with_logits(ans_distrib, plausible_ans)
					ground_loss = self.lambda_ground * self.ground_loss(pred_attn_mask, attn_mask)
					batch_loss += (valid_loss + plaus_loss + ground_loss)

				batch_loss.backward()

				nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
				
				self.optimizer.step()
				loss += batch_loss.data
				train_accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

				if i % self.args.display_every == 0:
					if self.args.opt_met:
						print('Train Epoch: {0:}, Iteration: {1:}, Total Loss: {2:.3f}, AL: {3:.3f}, VL: {4:.3f}, PL: {5:.3f}, GL: {6:.3f}'.format(epoch, i, batch_loss.data, accur_loss, valid_loss.data, plaus_loss.data, ground_loss.data))
					else:
						print('Train Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, batch_loss.data))
				break

			train_acc = np.mean(train_accuracies)
			
			val_loss, val_acc = self.eval()

			self.log_stats(loss, val_loss, train_acc, val_acc, epoch)
			print('Valid Epoch: {}, Val Acc: {}'.format(epoch, val_acc))
			if val_acc > self.best_val_acc:
				print('Saving new best model. Better than previous accuracy: {}'.format(self.best_val_acc))
				self.best_val_acc = val_acc
				# Initiate Model Checkpointing
			else:
				print ('Not saving as best model.')
			self.save_ckpt(save_best=(val_acc==self.best_val_acc))
			self.write_status(epoch, self.best_val_acc)
			
	
	def eval(self):

		loss = 0.0
		accuracies = []

		if self.args.opt_met:
			valid_total, plausible_total, samples = 0.0, 0.0, 0

		self.model.eval()
		for i, batch in enumerate(self.val_loader):

			# Unpack the items from the batch tensor
			ques_lens = batch['ques_lens'].to(self.device)
			sorted_indices = torch.argsort(ques_lens, descending=True)
			ques_lens = ques_lens[sorted_indices] 
			img_feats = batch['image_feat'].to(self.device)[sorted_indices]
			ques = batch['ques'].to(self.device)[sorted_indices]
			objs = batch['obj_bboxes'].to(self.device)[sorted_indices]
			adj_mat = batch['A'].to(self.device)[sorted_indices]
			num_objs = batch['num_objs'].to(self.device)[sorted_indices] 
			ans_output = batch['ans'].to(self.device)[sorted_indices]
			obj_wrds = batch['obj_wrds'].to(self.device)[sorted_indices]
			obj_region_mask = batch['obj_region_mask'].to(self.device)[sorted_indices]
			attn_mask = batch['attn_mask'].to(self.device)[sorted_indices]

			if self.args.use_rel_probs or self.args.use_rel_probs_sum:
				rel_prob_mat = batch['P'].to(self.device)[sorted_indices]
			else:
				rel_prob_mat = None

			ans_distrib, pred_attn_mask, _ = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_objs, obj_wrds, obj_region_mask, rel_prob_mat)
			pred_attn_mask[attn_mask <= 0] = 0

			if self.args.opt_met:
				valid_ans = batch['valid_ans'].to(self.device)[sorted_indices]
				plausible_ans = batch['plausible_ans'].to(self.device)[sorted_indices]

			if self.args.criterion == "bce" or self.args.use_bua or self.args.use_bua2:
				batch_loss = self.instance_bce_with_logits(ans_distrib, ans_output)
			else:
				batch_loss = self.criterion(ans_distrib, ans_output)

			if self.args.opt_met:
				valid_loss = self.lambda_valid * self.instance_bce_with_logits(ans_distrib, valid_ans)
				plaus_loss = self.lambda_plaus *self.instance_bce_with_logits(ans_distrib, plausible_ans)
				ground_loss = self.lambda_ground *self.ground_loss(pred_attn_mask, attn_mask)
				batch_loss += (valid_loss + plaus_loss + ground_loss)

				valid_batch, plausible_batch, sz = self.compute_metrics(ans_distrib, valid_ans, plausible_ans)
				samples += sz
				valid_total += valid_batch
				plausible_total += plausible_batch
			
			loss += batch_loss.data

			accuracies.extend(self.get_accuracy(ans_distrib, ans_output))
			break

		if self.args.opt_met:
			print('Validity: {}, Plausibility: {}'.format(float(valid_total/samples), float(plausible_total/samples)))

		return loss, np.mean(accuracies)

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

	def check_restart_conditions(self):

		# Check for the status file corresponding to the model
		status_file = os.path.join(self.args.ckpt_dir, 'status.json')

		if os.path.exists(status_file):
			with open(status_file, 'r') as f:
				status = json.load(f)			
			self.resume_from_epoch = status['epoch']
			self.best_val_acc = status['best_val_acc']
		else:
			self.resume_from_epoch = 0
			self.best_val_acc = 0.0

	def write_status(self, epoch, best_val_acc):

		status_file = os.path.join(self.args.ckpt_dir, 'status.json')
		status = {'epoch': epoch, 'best_val_acc': best_val_acc}

		with open(status_file, 'w') as f:
			json.dump(status, f, indent=4)

	def adjust_lr(self, epoch):
		
		# Sets the learning rate to the initial LR decayed by 2 every learning_rate_decay_every epochs
		
		lr_tmp = self.lr * (0.5 ** (epoch // self.args.learning_rate_decay_every))
		return lr_tmp

	def set_criterion(self):

		if self.args.criterion == "bce":
			self.criterion = nn.BCEWithLogitsLoss()
		elif self.args.criterion == "xce":
			self.criterion = nn.CrossEntropyLoss()
		else:
			raise("Invalid loss criterion")

	def log_stats(self, train_loss, val_loss, train_acc, val_acc, epoch):
		
		"""
		Log the stats of the current
		"""

		if self.log:
			self.writer.add_scalar('train/loss', train_loss, epoch)
			self.writer.add_scalar('train/acc', train_acc, epoch)
			self.writer.add_scalar('val/loss', val_loss, epoch)
			self.writer.add_scalar('val/acc', val_acc, epoch)

	def load_ckpt(self):
		"""
		Load the model checkpoint from the provided path
		"""

		# TODO: Maybe load args as well from the checkpoint

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.args.ckpt_dir, '{}.ckpt'.format(model_name))

		ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		self.model.load_state_dict(ckpt['state_dict'])

	def save_ckpt(self, save_best=False):

		"""
		Saves the model checkpoint at the correct directory path
		"""

		model_name = self.model.__class__.__name__
		ckpt_path = os.path.join(self.args.ckpt_dir, '{}.ckpt'.format(model_name))

		# Maybe add more information to the checkpoint
		model_dict = {
			'state_dict': self.model.state_dict(),
			'args': self.args
		}

		torch.save(model_dict, ckpt_path)

		if save_best:
			best_ckpt_path = os.path.join(self.args.ckpt_dir, '{}_best.ckpt'.format(model_name))
			torch.save(model_dict, best_ckpt_path)

