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

		print('Executing on Device: {}'.format(self.device))

		self.dataset = 'balanced'
		self.qa_data_path = {}
		self.sg_data_path = {}
		self.choices_data_path = {}
		for mode in ['train', 'val']:
			self.qa_data_path[mode] = os.path.join(self.expt_data_dir, '{dataset}_{mode}_data.json'.format(dataset=self.dataset, mode=mode))
			self.sg_data_path[mode] = os.path.join(self.expt_data_dir, self.gen_mode, '{mode}_sceneGraphs.json'.format(mode=mode))
			self.choices_data_path[mode] = os.path.join(self.expt_data_dir, '{mode}_choices.json'.format(mode=mode)) 
		
		# Extract the test set dir path
		test_set_dir = ('/').join(self.expt_data_dir.split('/')[:-2] + [self.test_dirname])
		
		# Add the paths for test set
		self.qa_data_path['test'] = os.path.join(test_set_dir, '{dataset}_{mode}_data.json'.format(dataset=self.dataset, mode='test'))
		self.sg_data_path['test'] = os.path.join(test_set_dir, self.gen_mode, '{mode}_sceneGraphs.json'.format(mode='test'))
		self.choices_data_path['test'] = os.path.join(self.expt_data_dir, '{mode}_choices.json'.format(mode='test'))

		self.img_feat_data_path = os.path.join(self.feats_data_dir, 'gqa_spatial.h5')
		self.img_info_path = os.path.join(self.feats_data_dir, 'gqa_spatial_merged_info.json')
		self.rel_vocab_path = os.path.join(self.expt_data_dir, self.gen_mode, 'sg_vocab.json')
		self.word_vocab_path = os.path.join(self.expt_data_dir, 'qa_vocab.json')
		self.meta_data_path = os.path.join(self.expt_data_dir, 'meta_data.json')
		self.word2vec_path = os.path.join(self.expt_data_dir, 'glove.{}d.json'.format(self.ques_word_vec_dim))
		self.rel_word2vec_path = os.path.join(self.expt_data_dir, self.gen_mode, 'rel_glove.{}d.json'.format(self.rel_emb_dim))
		self.obj_name_word2vec_path = os.path.join(self.expt_data_dir, self.gen_mode, 'obj_name_glove.{}d.json'.format(self.obj_emb_dim))

		self.valid_img_ids_path = os.path.join(self.expt_data_dir, 'vg_data', 'valid_img_ids.json')
		self.expt_res_dir = os.path.join(self.expt_res_base_dir, self.expt_name)
		self.log_dir = os.path.join(self.expt_res_dir, 'logs')
		self.ckpt_dir = os.path.join(self.expt_res_dir, 'ckpt')
		self.create_dir(self.log_dir)
		self.create_dir(self.ckpt_dir)

		# Enforcing a particular combination of flags for certain configurations
		if self.use_rl:
			self.opt_met = True
			self.optim = 'adam'

	def set_config(self, config):
		for key in config:
			setattr(self, key, config[key])

	def __str__(self):
		res = ""
		for k in self.__dict__:
			res += "{}: {}\n".format(k, self.__dict__[k])
		return res

	def __repr__(self):
		res = ""
		for k in self.__dict__:
			res += "{}: {}\n".format(k, self.__dict__[k])
		return res

	def get_dict(self):
		return self.__dict__

	def create_dir(self, dir_path):

		if not os.path.exists(dir_path):
			os.makedirs(dir_path)	

# Creating an Object for the argument parser to populate the fields
args = Config()

def parse_args():

	parser = argparse.ArgumentParser(fromfile_prefix_chars = "@")

	# Experiment Related Options
	parser.add_argument('--log', action="store_true", default=False, help="Whether to log the results or not")
	parser.add_argument('--expt_res_base_dir', type=str, help="Path to base directory where all the results and logs related to the experiment will be stored")
	parser.add_argument('--expt_name', type=str, help="Name of the experiment to uniquely identify its folder")
	parser.add_argument('--expt_data_dir', type=str, help="The path which contains most of the data required for the experiment")
	parser.add_argument('--test_dirname', default="test_set", type=str, help="The path which contains the test data required for the experiment")
	parser.add_argument('--feats_data_dir', type=str, help="The path of the directory containing image and object features")
	parser.add_argument('--gen_mode', type=str, default="gold", choices=["gold","pred_cls","sg_cls","sg_gen"], help="The path of directory containing scenegraphs")
	parser.add_argument('--get_preds', default=False, action="store_true", help="Flag to indicate if the evaluator should store the predictions as well")
	
	parser.add_argument('--opt_met', action="store_true", default=False, help="Optimize for Metrics other than accuracy")
	parser.add_argument('--met_loss_wt', default=0.4, type=float, help="Weightage given to metric loss during training")
	parser.add_argument('--mode', type=str, required=True, help="Specify the mode: {train, eval}")
	parser.add_argument('--num_epochs', default=10, type=int, help="The number of epochs for training the model")
	parser.add_argument('--criterion', default="xce", help="The loss criterion to be used for training the model")
	parser.add_argument('--optim', default="adamax", help="Optimizer to be used for training the model")
	parser.add_argument('--learning_rate_decay_every', type=int, default=100, help="The schedule after which the learning is decayed by half")
	parser.add_argument('--lr', default=1e-3, type=float, help="The learning rate for training the architecture")
	parser.add_argument('--bsz', default=32, type=int, help="Batch Size")
	parser.add_argument('--use_cuda', action="store_true", default=False, help="Flag to use CUDA")
	parser.add_argument('--display_every', type=int, default=1, help="Loss statistics to display after every n batches")
	parser.add_argument('--drop_prob', default=0.0, type=float, help="Dropout probability for all linear layers")
	
	parser.add_argument('--nl', default='relu', choices=['relu', 'gated_tanh', 'tanh'], help="Type of Non linearity to be used in the network (relu, gated_tanh, tanh)")

	# Options for Question Encoder
	parser.add_argument('--n_ques_emb', type=int, default=1024, help="The dimension of the hidden layer in Question Model")
	parser.add_argument('--ques_word_vec_dim', type=int, default=300, help="The dimension of each word in the Question Encoder")
	parser.add_argument('--n_ques_layers', type=int, default=2, help="The number of layers of RNN in Question Model")
	parser.add_argument('--bidirectional', action="store_true", default=False, help="Flag to indicate if the RNN runs in both directions in Question Model")
	parser.add_argument('--use_glove', action="store_true",  default=False, help="Directive to use pre-trained Glove embeddings")
	
	# Options for Image Model
	parser.add_argument('--n_img_feats', type=int, default=2048, help="The dimension of the Image Features")
	parser.add_argument('--reduce_img_feats', action="store_true", help="Flag to indicate reduction of image features to same as relation word embedding through a linear layer")
	parser.add_argument('--weights_init', default='xavier', help="The initializer for weight matrices in the network")
	parser.add_argument('--gcn_depth', default=5, type=int, help="The depth of the GCN network")
	parser.add_argument('--rel_emb_dim', default=300, type=int, help="The dimensionality of the relation embedding")
	parser.add_argument('--use_rel_emb', action="store_true", default=False, help="Use Relation Embedding")
	parser.add_argument('--use_rel_words', action="store_true", default=False, help="Use object names and relations in GCN")
	parser.add_argument('--obj_emb_dim', default=300, type=int, help="The dimensionality of the object name embedding")
	parser.add_argument('--use_blind', action="store_true", default=False, help="Use blind model (only applicable with rel_words)")
	parser.add_argument('--use_rel_probs', action="store_true", default=False, help="Use a linear to aggregate relation probabilities (only applicable with rel_words)")
	parser.add_argument('--use_rel_probs_sum', action="store_true", default=False, help="Use sum to aggregate relation probabilities (only applicable with rel_words)")

	# Options for Answering Model
	parser.add_argument('--n_qi_gate', type=int, default=1024, help="The dimension of the Question-Image Gate in Answerer Model")
	parser.add_argument('--n_ans_gate', type=int, default=1024, help="The dimension of the Answer Gate")
	parser.add_argument('--n_attn', type=int, default=1024, help="The dimension of the output feature dimension for each object for computing attention weights")

	# Options for Attention Model
	parser.add_argument('--use_img_feats', action="store_true", default=False, help="Use Average Image Features in Attention Model")
	parser.add_argument('--pool_w', default=20, type=int, help="Pooling width for Image features in Attention Module")
	parser.add_argument('--pool_h', default=15, type=int, help="Pooling height for Image features in Attention Module")


	parser.add_argument('--use_bua', action="store_true", default=False, help="Use Git BottomUp")
	parser.add_argument('--use_bua2', action="store_true", default=False, help="Use Git BottomUp2")
	parser.add_argument('--use_mac', action="store_true", default=False, help="Use MAC based network")
	parser.add_argument('--use_memory_gate', action="store_true", default=False, help="Use memory gate in MAC network")
	parser.add_argument('--mac_max_steps', default=4, type=int, help="Number of steps to run the MAC operation for")
	parser.add_argument('--mac_self_attention', default=False, action="store_true", help="Use self attention in MAC unit")
	parser.add_argument('--mac_dim', default=512, type=int, help="The default dimension of the LSTM cells in the MAC network")

	# Options for extra metric optimizations
	parser.add_argument('--lambda_ground', type=float, default=20, help="Weight for Grounding Loss")
	parser.add_argument('--lambda_valid', type=float, default=0.02, help="Weight for Validity Loss")
	parser.add_argument('--lambda_plaus', type=float, default=0.04, help="Weight for Plausibility Loss")

	# Options for SAN
	parser.add_argument('--use_san', action="store_true", default=False, help="Use Stacked Attention Network")
	parser.add_argument('--n_attn_layers', default=2, type=int, help="Number of attentions in the stack of SAN")
	parser.add_argument('--san_dim_in', default=512, type=int, help="Dimension of the Input tensors to SAN")
	parser.add_argument('--san_dim_mid', default=256, type=int, help="Dimension of the Intermediate tensors in SAN")

	# Options for RL
	parser.add_argument('--use_rl', action="store_true", default=False, help="Use RL training method")

	return parser.parse_args(namespace = args)