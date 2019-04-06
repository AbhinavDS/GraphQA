"""
Module that parses the command line arguments
"""

import argparse

class Config(object):

	def __init__(self):

		self.x = "x"

def parse_args():

	parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

	# Experiment Related Options
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--log', action="store_true", default=False, help="Whether to log the results or not")
	parser.add_argument('--log_dir', type=str, help="Path to directory where the tensorboard logs")

	# Options for Question Encoder
	parser.add_argument('--n_ques_emb', type=int, default=300, help="The dimension of the question embedding")

	# Options for Image Model
	parser.add_argument('--n_img_feats', type=int, default=512, help="The dimension of the Image Features")

	# Options for Answering Model
	parser.add_argument('--n_qi_gate', type=int, default=128, help="The dimension of the Question-Image Gate in Answerer Model")
	parser.add_argument('--n_ans_gate', type=int, default=128, help="The dimension of the Answer Gate")
	parser.add_argument('--n_attn', type=int, default=128, help="The dimension of the output feature dimension for each object for computing attention weights")

	return parser.parse_args(namespace = args)