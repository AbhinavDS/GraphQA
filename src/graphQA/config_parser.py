"""
Module that parses the command line arguments
"""

import argparse
import os
import torch

class Config(object):

	def gen_config(self):
		
		"""
		Function to be invoked after arguments have been processed to generate additional config variables
		"""

		if self.use_cuda:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		else:
			self.device = "cpu"

		self.log_dir = os.path.join(self.expt_dir, 'logs')
		self.ckpt_dir = os.path.join(self.expt_dir, 'ckpt')
		self.create_dir(self.log_dir)
		self.create_dir(self.ckpt_dir)

	def set_config(self, config):
		for key in config:
			setattr(self, key, config[key])

	def __str__(self):
		return str(self.__dict__)

	def __repr__(self):
		return str(self.__dict__)

	def create_dir(self, dir_path):

		if not os.path.exists(dir_path):
			os.makedirs(dir_path)	

# Creating an Object for the argument parser to populate the fields
args = Config()

def parse_args():

	parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

	# Experiment Related Options
	parser.add_argument('--log', action="store_true", default=False, help="Whether to log the results or not")
	parser.add_argument('--expt_dir', type=str, help="Path to directory where all the data related to the experiment will be stored")
	parser.add_argument('--qa_data_path', type=str, help="The path of the json file containing the QA pairs")
	parser.add_argument('--sc_data_path', type=str, help="The path of the json file containing the Scene Graph")
	parser.add_argument('--img_feat_data_path', type=str, required=True, help="The path of the json file containing the Image Features")
	parser.add_argument('--val_qa_data_path', type=str, required=True, help="The path of the json file containing the QA pairs for evaluation")
	parser.add_argument('--val_sc_data_path', type=str, required=True, help="The path of the json file containing the Scene Graph for evaluation")
	parser.add_argument('--rel_vocab_path', type=str, required=True, help="The path to relations vocabulary file")
	parser.add_argument('--word_vocab_path', type=str, required=True, help="The path to word vocabulary file for question and answers")
	parser.add_argument('--img_info_path', type=str, required=True, help="The path to JSON file Containing Image info")
	
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--num_epochs', default=25, help="The number of epochs for training the model")
	parser.add_argument('--criterion', default="bce", help="The loss criterion to be used for training the model")
	parser.add_argument('--learning_rate_decay_every', type=int, default=30, help="The schedule after which the learning is decayed by half")
	parser.add_argument('--lr', default=1e-3, help="The learning rate for training the architecture")
	parser.add_argument('--bsz', default=32, help="Batch Size")
	parser.add_argument('--use_cuda', action="store_true", default=False, help="Flag to use CUDA")

	parser.add_argument('--nl', default='relu', help="Type of Non linearity to be used in the network (relu, gated_tanh)")

	# Options for Question Encoder
	parser.add_argument('--n_ques_emb', type=int, default=256, help="The dimension of the hidden layer in Question Model")
	parser.add_argument('--ques_word_vec_dim', type=int, default=300, help="The dimension of each word in the Question Encoder")
	parser.add_argument('--n_ques_layers', type=int, default=2, help="The number of layers of RNN in Question Model")
	parser.add_argument('--bidirectional', action="store_true", default=False, help="Flag to indicate if the RNN runs in both directions in Question Model")
	parser.add_argument('--use_ques_attention', action="store_true", default=False, help="Whether to use attention in Question Model")

	# Options for Image Model
	parser.add_argument('--n_img_feats', type=int, default=2048, help="The dimension of the Image Features")
	parser.add_argument('--weights_init', default='xavier', help="The initializer for weight matrices in the network")
	parser.add_argument('--gcn_depth', default=8, type=int, help="The depth of the GCN network")
	parser.add_argument('--rel_emb_dim', default=128, type=int, help="The dimensionality of the relation embedding")

	# Options for Answering Model
	parser.add_argument('--n_qi_gate', type=int, default=128, help="The dimension of the Question-Image Gate in Answerer Model")
	parser.add_argument('--n_ans_gate', type=int, default=128, help="The dimension of the Answer Gate")
	parser.add_argument('--n_attn', type=int, default=128, help="The dimension of the output feature dimension for each object for computing attention weights")

	return parser.parse_args(namespace = args)