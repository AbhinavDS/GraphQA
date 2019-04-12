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

		self.dataset = 'balanced'
		self.qa_data_path = {}
		self.sg_data_path = {}
		for mode in ['train', 'val', 'test']:
			self.qa_data_path[mode] = os.path.join(self.expt_data_dir, '{dataset}_{mode}_data.json'.format(dataset=self.dataset, mode=mode))
			self.sg_data_path[mode] = os.path.join(self.expt_data_dir, '{mode}_sceneGraphs.json'.format(mode=mode))
		
		self.img_feat_data_path = os.path.join(self.feats_data_dir, 'gqa_spatial.h5')
		self.img_info_path = os.path.join(self.feats_data_dir, 'gqa_spatial_merged_info.json')
		self.rel_vocab_path = os.path.join(self.expt_data_dir, 'sg_vocab.json')
		self.word_vocab_path = os.path.join(self.expt_data_dir, 'qa_vocab.json')
		self.meta_data_path = os.path.join(self.expt_data_dir, 'meta_data.json')

		self.log_dir = os.path.join(self.expt_res_dir, 'logs')
		self.ckpt_dir = os.path.join(self.expt_res_dir, 'ckpt')
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
	parser.add_argument('--expt_res_dir', type=str, help="Path to directory where all the data related to the experiment will be stored")
	parser.add_argument('--expt_data_dir', type=str, help="The path which contains most of the data required for the experiment")
	parser.add_argument('--feats_data_dir', type=str, help="The path of the directory containing image and object features")
	
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--num_epochs', default=10, help="The number of epochs for training the model")
	parser.add_argument('--criterion', default="xce", help="The loss criterion to be used for training the model")
	parser.add_argument('--learning_rate_decay_every', type=int, default=30, help="The schedule after which the learning is decayed by half")
	parser.add_argument('--lr', default=1e-3, help="The learning rate for training the architecture")
	parser.add_argument('--bsz', default=32, help="Batch Size")
	parser.add_argument('--use_cuda', action="store_true", default=False, help="Flag to use CUDA")
	parser.add_argument('--display_every', type=int, default=10, help="Loss statistics to display after every n batches")
	parser.add_argument('--drop_prob', default=0.0, help="Dropout probability for all linear layers")
	
	parser.add_argument('--nl', default='relu', help="Type of Non linearity to be used in the network (relu, gated_tanh)")

	# Options for Question Encoder
	parser.add_argument('--n_ques_emb', type=int, default=256, help="The dimension of the hidden layer in Question Model")
	parser.add_argument('--ques_word_vec_dim', type=int, default=300, help="The dimension of each word in the Question Encoder")
	parser.add_argument('--n_ques_layers', type=int, default=2, help="The number of layers of RNN in Question Model")
	parser.add_argument('--bidirectional', action="store_true", default=False, help="Flag to indicate if the RNN runs in both directions in Question Model")
	
	# Options for Image Model
	parser.add_argument('--n_img_feats', type=int, default=2048, help="The dimension of the Image Features")
	parser.add_argument('--weights_init', default='xavier', help="The initializer for weight matrices in the network")
	parser.add_argument('--gcn_depth', default=5, type=int, help="The depth of the GCN network")
	parser.add_argument('--rel_emb_dim', default=64, type=int, help="The dimensionality of the relation embedding")

	# Options for Answering Model
	parser.add_argument('--n_qi_gate', type=int, default=128, help="The dimension of the Question-Image Gate in Answerer Model")
	parser.add_argument('--n_ans_gate', type=int, default=128, help="The dimension of the Answer Gate")
	parser.add_argument('--n_attn', type=int, default=512, help="The dimension of the output feature dimension for each object for computing attention weights")

	return parser.parse_args(namespace = args)