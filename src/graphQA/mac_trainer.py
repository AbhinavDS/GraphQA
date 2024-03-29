"""
Module that controls the training of the graphQA module using MAC network
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from .models.mac import MACNetwork as MacNetwork

class Trainer:

	def __init__(self, args, train_dataset, val_dataset):

		self.args = args
		self.num_epochs = args.num_epochs
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.use_glove = args.use_glove
		self.use_rel_words = args.use_rel_words
		
		self.device = self.args.device

		self.model = MacNetwork(args)
		self.model_running = MacNetwork(args)
		self.accumulate(self.model_running, self.model, 0)

		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.set_criterion()
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
		self.model_running.to(self.device)

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

				ans_distrib = self.model(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)
				
				batch_loss = self.criterion(ans_distrib, ans_output)
				batch_loss.backward()

				#nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
				self.optimizer.step()
				self.accumulate(self.model_running, self.model)
				loss += batch_loss.data
				train_accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

				if i % self.args.display_every == 0:
					print('Train Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, batch_loss))

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

		self.model_running.eval()
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
			ans_distrib = self.model_running(img_feats, ques, objs, adj_mat, ques_lens, num_obj, obj_wrds)

			batch_loss = self.criterion(ans_distrib, ans_output)
			loss += batch_loss.data

			accuracies.extend(self.get_accuracy(ans_distrib, ans_output))

		return loss, np.mean(accuracies)

	def get_accuracy(self, preds, correct):

		"""
		Compute the average accuracy of predictions wrt correct
		"""
		
		pred_ids = np.argmax(preds.detach().cpu().numpy(), axis = -1)
		correct_ids = correct.cpu().numpy()
		acc = np.equal(pred_ids.reshape(-1), correct_ids)
		return acc
	
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

	def accumulate(self, model1, model2, decay=0.999):

		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
		    par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

	def set_criterion(self):

		self.criterion = nn.CrossEntropyLoss()

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
		self.accumulate(self.model_running, self.model, 0)

	def save_ckpt(self, save_best=False):

		"""
		Saves the model checkpoint at the correct directory path
		"""

		model_name = self.model_running.__class__.__name__
		ckpt_path = os.path.join(self.args.ckpt_dir, '{}.ckpt'.format(model_name))

		# Maybe add more information to the checkpoint
		model_dict = {
			'state_dict': self.model_running.state_dict(),
			'args': self.args
		}

		torch.save(model_dict, ckpt_path)

		if save_best:
			best_ckpt_path = os.path.join(self.args.ckpt_dir, '{}_best.ckpt'.format(model_name))
			torch.save(model_dict, best_ckpt_path)