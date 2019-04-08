"""
Module that controls the training of the graphQA module
"""

import os
import json
import torch
from torch import nn
import numpy as np
from tensorboardX import SummaryWriter

from models.bottom_up_gcn import BottomUpGCN

class Trainer:

	def __init__(self, args, train_loader, val_loader):

		self.args = args
		self.num_epochs = args.num_epochs
		self.model = BottomUpGCN(args)
		self.device = self.args.device

		self.train_loader = train_loader
		self.val_loader = val_loader
		# Can be changed to support different optimizers
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
		self.set_criterion()

		if args.log:
			writer = SummaryWriter(args.log_dir)

	def train(self):

		print('Initiating Training')

		# Check if the training has to be restarted
		self.check_restart_conditions()
		
		if self.resume_from_epoch >= 1:
			print('Loading from Checkpointed Model')
			# Write the logic for loading checkpoint for the model
		
		self.model.to(self.device)

		for epoch in range(self.resume_from_epoch, self.num_epochs):

			lr = self.adjust_lr(epoch)
			self.model.train()

			loss = 0.0
			train_accuracies = []

			for i, batch in enumerate(self.train_loader):

				print('Training Batch: {} in epoch: {}'.format(i, epoch))
				self.optimizer.zero_grad()

				# Unpack the items from the batch tensor
				#img_feats = batch['img_feats'].to(self.device)
				#ques = batch['ques'].to(self.device)
				#objs = 
				#adj_mat = 
				#rels = 
				#ques_lens = 
				#num_obj = 
				#ans_output = 

				ans_distribbatch_loss = self.criterion(ans_distrib, ans_output)
				loss += batch_loss = self.model(img_feats, ques, objs, adj_mat, rels, ques_lens, num_obj)

				batch_loss = self.criterion(ans_distrib, ans_output)
				loss += batch_loss

				train_accuracies.extend(get_accuracy(ans_distrib, ans_output))

				self.model.backward()
				self.optimizer.step()

				if i % self.args.display_every == 0:
					print('Epoch: {}, Iteration: {}, Loss: {}'.format(epoch, i, batch_loss))

			train_acc = np.mean(train_accuracies)
			val_loss, val_acc = eval()

			self.log_stats(loss, val_loss, train_acc, val_acc)

			if val_acc > self.best_val_acc:
				self.best_val_acc = val_acc

				print('Updating Best Model after Epoch: {}, Val Acc: {}'.format(epoch, val_acc))
				# Initiate Model Checkpointing

	
	def eval(self):

		loss = 0.0
		accuracies = []

		for i, batch in enumerate(self.data_loader):

			self.model.eval()

			# Unpack the items from the batch tensor
			#img_feats = batch['img_feats'].to(self.device)
			#ques = batch['ques'].to(self.device)
			#objs = 
			#adj_mat = 
			#rels = 
			#ques_lens = 
			#num_obj = 
			#ans_output =

			ans_distrib = self.model(img_feats, ques, objs, adj_mat, rels, ques_lens, num_obj)
			batch_loss = self.criterion(ans_distrib, ans_output)
			loss += batch_loss

			accuracies.extend(get_accuracy(ans_distrib, ans_output))

		return loss, np.mean(accuracies)

	def get_accuracy(self, preds, correct):

		"""
		Compute the average accuracy of predictions wrt correct
		"""
		pred_ids = np.argmax(preds.data.cpu().numpy(), axis = -1)

		if self.args.criterion == "bce":
			# correct is in form of one hot vector
			correct_ids = np.argmax(correct.data.cpu().numpy(), axis = -1)
		elif self.args.criterion == "xce":
			correct_ids = correct.data.cpu().numpy()
		else:
			raise("Incorrect Loss function to compute accuracy for")

		return np.equal(pred_ids, correct_ids)
	
	def check_restart_conditions(self):

		# Check for the status file corresponding to the model
		status_file = os.path.join(self.args.checkpoint_dir, 'status.json')

		if os.path.exists(status_file):
			with open(status_file, 'r') as f:
				status = json.load(f)			
			self.resume_from_epoch = status['epoch']
			self.best_val_acc = status['best_val_acc']
		else:
			self.resume_from_epoch = 0
			self.best_val_acc = 0.0

	def write_status(self, epoch, best_val_acc):

		status_file = os.path.join(self.args.checkpoint_dir, 'status.json')
		status = {'epoch': epoch, 'best_val_acc': best_val_acc}

		with open(status_file, 'w') as f:
			json.dump(status, f, indent=4)

	def adjust_lr(self, epoch):
		
		# Sets the learning rate to the initial LR decayed by 2 every learning_rate_decay_every epochs
		
		lr_tmp = lr * (0.5 ** (epoch // self.args.learning_rate_decay_every))
		return lr_tmp

	def set_criterion(self):

		if self.args.criterion == "bce":
			self.criterion = nn.BCELoss()
		elif self.args.criterion == "xce":
			self.criterion = nn.CrossEntropyLoss()
		else:
			raise("Invalid loss criterion")