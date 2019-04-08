"""
Module that parses the command line arguments
"""

import argparse
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

	def set_config(self, dict):
		for key in dict:
			setattr(self, key, dict[key])

	def __str__(self):
		return str(self.__dict__)

	def __repr__(self):
		return str(self.__dict__)		

# Creating an Object for the argument parser to populate the fields
args = Config()


def parse_args():

	parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

	# Experiment Related Options
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--log', action="store_true", default=False, help="Whether to log the results or not")
	parser.add_argument('--log_dir', type=str, help="Path to directory where the tensorboard logs")

	parser.add_argument('--nl', default='relu', help="Type of Non linearity to be used in the network (relu, gated_tanh)")

	# Options for Question Encoder
	parser.add_argument('--n_ques_emb', type=int, default=300, help="The dimension of the question embedding")

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