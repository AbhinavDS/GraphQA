import numpy as np
import json
import h5py
import torch
from torch.utils.data import Dataset

import utils.preprocess as preprocess_utils
import utils.utils as utils

class GQADataset(Dataset):

	def __init__(self, args, qa_data_key=None, sg_data_key=None):
		
		self.args = args
		
		question_json_path = self.args.qa_data_path[qa_data_key]
		scene_graph_json_path = self.args.sg_data_path[sg_data_key]
		image_features_path = self.args.img_feat_data_path
		image_info_json_path = self.args.img_info_path
		vocab_json = self.args.word_vocab_path
		relations_vocab_json = self.args.rel_vocab_path
		meta_data_json = self.args.meta_data_path

		with open(question_json_path, 'r') as qf:
			self.questions = json.load(qf)

		self.questions_keys = list(self.questions.keys())
		
		with open(scene_graph_json_path, 'r') as sgf:
			self.scene_graphs = json.load(sgf)
		
		with open(image_info_json_path, 'r') as img_if:
			self.image_info = json.load(img_if)
		
		self.image_features_h5 = h5py.File(image_features_path, 'r')['features']
		
		self.vocab = utils.load_vocab(vocab_json)
		self.relations_vocab = utils.load_vocab(relations_vocab_json)

		self.meta_data = utils.load_vocab(meta_data_json)
		print("Data Read successful", len(self.questions_keys))

	def get_data_config(self):

		"""
		Returns the config variables wrt data in a dictionary format
		"""

		config = {}
		config['max_num_objs'] = self.meta_data['max_num_objs']
		config['max_ques_len'] = self.meta_data['max_ques_len']
		config['max_rels'] = len(self.relations_vocab['relation_token_to_idx'])
		config['variable_lengths'] = True
		config['n_ans'] = len(self.vocab['answer_token_to_idx'])
		config['ques_start_id'] = preprocess_utils.SPECIAL_TOKENS['<START>']
		config['ques_end_id'] = preprocess_utils.SPECIAL_TOKENS['<END>']
		config['ques_null_id'] = preprocess_utils.SPECIAL_TOKENS['<NULL>']
		config['ques_unk_id'] = preprocess_utils.SPECIAL_TOKENS['<UNK>']
		config['ques_vocab_sz'] = len(self.vocab['question_token_to_idx'])

		return config

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):
		if idx >= len(self):
			raise ValueError('index %d out of range (%d)' % (idx, len(self)))
		
		key = self.questions_keys[idx]

		# QA
		question = self.questions[key]['question']
		answer = self.questions[key].get('answer', '<NULL>')
		# Encode question and answer (tokenize as well)
		question_tokens = preprocess_utils.tokenize(question,
												punct_to_keep=[';', ','],
												punct_to_remove=['?', '.'])
		ques_len = len(question_tokens)
		question_encoded = preprocess_utils.encode(question_tokens,
												 self.vocab['question_token_to_idx'],
												 allow_unk=True, max_len=self.meta_data['max_ques_len'])
		answer_encoded = preprocess_utils.encode([answer],
												 self.vocab['answer_token_to_idx'],
												 allow_unk=True)
		# SG
		image_idx = self.questions[key]['imageId']
		sg = self.scene_graphs[image_idx]
		width, height = (float)(sg['width']), (float)(sg['height'])
		
		# # Get object dims for roi pooling and Create adjacency matrix with relations (FOR GCN_RELS)
		# num_relations = len(self.relations_vocab['relation_token_to_idx'])
		# A = np.zeros((self.meta_data['max_num_objs'], self.meta_data['max_num_objs'] * num_relations))
		# objects = np.zeros((self.meta_data['max_num_objs'], 4))
		# object_keys = list(sg['objects'].keys())
		# for num_objs, obj_key in enumerate(object_keys):
		# 	obj = sg['objects'][obj_key]
		# 	objects[num_objs][0] = obj['x'] / width
		# 	objects[num_objs][1] = obj['y'] / height
		# 	objects[num_objs][2] = (obj['x'] + obj['w']) / width
		# 	objects[num_objs][3] = (obj['y'] + obj['h']) / height
		# 	for relation in obj["relations"]:
		# 		rel_encoded = preprocess_utils.encode([relation['name']],
		# 										 self.relations_vocab['relation_token_to_idx'],
		# 										 allow_unk=True)
		# 		assert len(rel_encoded) == 1
		# 		obj_id = object_keys.index(relation['object'])
		# 		A_id = rel_encoded[0] * self.meta_data['max_num_objs'] + obj_id
		# 		A[num_objs][A_id] = 1# can give relation id in OxO matrix if needed
		# 	if num_objs > self.meta_data['max_num_objs']:
		# 		break

		# Get object dims for roi pooling and Create adjacency matrix with relations (FOR_SIMPLE_GCN)
		A = np.zeros((self.meta_data['max_num_objs'], self.meta_data['max_num_objs']))
		objects = np.zeros((self.meta_data['max_num_objs'], 4))
		
		object_keys = list(sg['objects'].keys())
		for num_objs, obj_key in enumerate(object_keys):
			obj = sg['objects'][obj_key]
			objects[num_objs][0] = obj['x'] / width
			objects[num_objs][1] = obj['y'] / height
			objects[num_objs][2] = (obj['x'] + obj['w']) / width
			objects[num_objs][3] = (obj['y'] + obj['h']) / height
			for relation in obj["relations"]:
				obj_id = object_keys.index(relation['object'])
				# can give relation id in OxO matrix if needed
				A[num_objs][obj_id] = 1

			if num_objs > self.meta_data['max_num_objs']:
				break

		# Get image feature from image
		h5_idx = self.image_info[image_idx]['index']
		image_feat = self.image_features_h5[h5_idx]
		
		return {
				'ques': torch.as_tensor(question_encoded, dtype=torch.long),
				'ans': answer_encoded[0],
				'ques_lens': ques_len,
				'obj_bboxes': torch.as_tensor(objects, dtype=torch.float),
				'num_objs': num_objs,
				'A': torch.as_tensor(A, dtype=torch.float),
				'image_feat': torch.as_tensor(image_feat, dtype=torch.float),
			}
