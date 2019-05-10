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
		
		self.model = Learner(Model, args, word2vec=self.embeddings_mat, rel_word2vec=self.rel_embeddings_mat, obj_name_word2vec=self.obj_names_embeddings_mat)
		
		self.device = self.args.device

		self.set_criterion()
		self.set_optimizer()
		self.lr = self.args.lr
		
		self.train_loader = DataLoader(dataset = self.train_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

		self.val_loader = DataLoader(dataset=self.val_dataset, batch_size=self.args.bsz, shuffle=True, num_workers=4)

		self.log = args.log
		if self.log:
			self.writer = SummaryWriter(args.log_dir)

	def set_optimizer(self):

		if self.args.optim == 'adam':
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		elif self.args.optim == 'adamax':
			self.optimizer = torch.optim.Adamax(self.model.parameters())
		else:
			raise('Specify Correct Optimizer')
	
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
				num_obj = batch['num_objs'].to(self.device)[sorted_indices] 
				ans_output = batch['ans'].to(self.device)[sorted_indices]
				obj_wrds = batch['obj_wrds'].to(self.device)[sorted_indices]
				valid_ans = batch['valid_ans'][sorted_indices].to(self.device)
				plausible_ans = batch['plausible_ans'][sorted_indices].to(self.device)

				policy_distrib, value, ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)
				
				action, action_log_probs = self.select_action(policy_distrib)

				rewards = self.compute_reward(action, valid_ans, plausible_ans)

				advantage = rewards - value
				actor_loss = -(action_log_probs * advantage)
				critic_loss = F.smooth_l1_loss(value, rewards)
				xce_loss = self.criterion(ans_distrib, ans_output)
				batch_loss = (actor_loss + critic_loss).mean() + xce_loss

				batch_loss.backward()

				nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
				
				self.optimizer.step()
				
				train_accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

				if i % self.args.display_every == 0:
					print('Train Epoch: {}, Iteration: {}, Rewards: {}'.format(epoch, i, rewards.mean()))

			train_acc = np.mean(train_accuracies)

			val_reward, val_acc = self.eval()

			self.log_stats(rewards.mean(), val_reward, train_acc, val_acc, epoch)
			print('Valid Epoch: {}, Val Acc: {}'.format(epoch, val_acc))
			if val_acc > self.best_val_acc:
				print('Saving new best model. Better than previous accuracy: {}'.format(self.best_val_acc))
				self.best_val_acc = val_acc
				# Initiate Model Checkpointing
			else:
				print ('Not saving as best model.')
			self.save_ckpt(save_best=(val_acc==self.best_val_acc))
			self.write_status(epoch, self.best_val_acc)
			
	
	def select_action(self, policy_distrib):

		sampler = Categorical(policy_distrib)
		action = sampler.sample()

		return action, sampler.log_prob(action)

	def compute_reward(self, action, valid_ans, plausible_ans):

		"""
		Assigns the reward for each action in the batch based on the metrics to be optimized for
		"""

		batch_sz = action.size(0)
		rewards = torch.zeros(batch_sz, dtype=torch.float32).to(self.device)
		valid_ans = valid_ans.detach().cpu().numpy()
		plausible_ans = plausible_ans.detach().cpu().numpy()

		for i in range(batch_sz):
			act = int(action[i])
			rewards[i] = 0.3 * valid_ans[i][act] + 0.7 * plausible_ans[i][act]

		return rewards

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
			num_obj = batch['num_objs'].to(self.device)[sorted_indices] 
			ans_output = batch['ans'].to(self.device)[sorted_indices]
			obj_wrds = batch['obj_wrds'].to(self.device)[sorted_indices]
			ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)

			valid_ans = batch['valid_ans'][sorted_indices].to(self.device)
			plausible_ans = batch['plausible_ans'][sorted_indices].to(self.device)

			policy_distrib, value, ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)
			
			action = policy_distrib.argmax(dim=-1, keepdim=False)

			rewards = self.compute_reward(action, valid_ans, plausible_ans)

			advantage = rewards - value
			xce_loss = self.criterion(ans_distrib, ans_output)
			
			valid_batch, plausible_batch, sz = self.compute_metrics(ans_distrib, valid_ans, plausible_ans)
			samples += sz
			valid_total += valid_batch
			plausible_total += plausible_batch
			
			accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

		if self.args.opt_met:
			print('Validity: {}, Plausibility: {}'.format(float(valid_total/samples), float(plausible_total/samples)))

		return rewards.mean(), np.mean(accuracies)

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

		if self.args.criterion == "xce":
			self.criterion = nn.CrossEntropyLoss()
		else:
			raise("Invalid loss criterion")

	def log_stats(self, train_reward, val_reward, train_acc, val_acc, epoch):
		
		"""
		Log the stats of the current
		"""

		if self.log:
			self.writer.add_scalar('train/reward', train_reward, epoch)
			self.writer.add_scalar('train/acc', train_acc, epoch)
			self.writer.add_scalar('val/reward', val_reward, epoch)
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

