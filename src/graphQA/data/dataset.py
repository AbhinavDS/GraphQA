import numpy as np
import json
import h5py
import torch
import os
from torch.utils.data import Dataset

import utils.preprocess as preprocess_utils
import utils.utils as utils

class GQADataset(Dataset):

	def __init__(self, args, qa_data_key=None, sg_data_key=None):
		
		self.args = args
		
		question_json_path = self.args.qa_data_path[qa_data_key]
		scene_graph_json_path = self.args.sg_data_path[sg_data_key]
		self.choices_data_json_path = self.args.choices_data_path[qa_data_key]

		self.image_features_path = self.args.img_feat_data_path
		image_info_json_path = self.args.img_info_path
		vocab_json = self.args.word_vocab_path
		sg_vocab_json = self.args.rel_vocab_path
		meta_data_json = self.args.meta_data_path
		valid_img_ids_path = self.args.valid_img_ids_path

		with open(question_json_path, 'r') as qf:
			self.questions = json.load(qf)

		with open(valid_img_ids_path, 'r') as vf:
			self.valid_img_ids = json.load(vf)

		self.questions_keys = list([qk for qk in self.questions.keys() if self.questions[qk]['imageId'] in self.valid_img_ids])	
		
		with open(scene_graph_json_path, 'r') as sgf:
			self.scene_graphs = json.load(sgf)
		
		with open(image_info_json_path, 'r') as img_if:
			self.image_info = json.load(img_if)

		self.vocab = utils.load_vocab(vocab_json)

		self.sg_vocab = utils.load_vocab(sg_vocab_json)
		self.rel_embeddings_mat = None
		self.obj_names_embeddings_mat = None
		self.meta_data = utils.load_vocab(meta_data_json)
		
		attn_factor = 0.25
		self.attn_width = (int)(640 * attn_factor)
		self.attn_height = (int)(480 * attn_factor)

		if self.args.opt_met:
			self.index_choices()

		if args.use_glove:
			self.word2vec_path = args.word2vec_path
			self.ques_word_vec_dim = args.ques_word_vec_dim
			if self.args.use_rel_emb:
				self.rel_word2vec_path = args.rel_word2vec_path
				self.rel_emb_dim = args.rel_emb_dim
				self.rel_embeddings_mat = self.load_embeddings(self.sg_vocab['relation_token_to_idx'], self.rel_emb_dim, self.rel_word2vec_path)
			elif self.args.use_rel_words:
				self.rel_word2vec_path = args.rel_word2vec_path
				self.obj_name_word2vec_path = args.obj_name_word2vec_path
				self.rel_emb_dim = args.rel_emb_dim
				self.obj_emb_dim = args.obj_emb_dim
				self.rel_embeddings_mat = self.load_embeddings(self.sg_vocab['relation_token_to_idx'], self.rel_emb_dim, self.rel_word2vec_path)
				self.obj_names_embeddings_mat = self.load_embeddings(self.sg_vocab['object_token_to_idx'], self.obj_emb_dim, self.obj_name_word2vec_path)
			
			self.embeddings_mat = self.load_embeddings(self.vocab['question_token_to_idx'], self.ques_word_vec_dim, self.word2vec_path)

		print("Data Read successful", len(self.questions_keys))

	def get_data_config(self):

		"""
		Returns the config variables wrt data in a dictionary format
		"""

		config = {}
		config['max_num_objs'] = self.meta_data['max_num_objs']
		config['max_obj_names'] = len(self.sg_vocab['object_token_to_idx'])
		config['max_ques_len'] = self.meta_data['max_ques_len']
		config['max_rels'] = len(self.sg_vocab['relation_token_to_idx'])
		config['variable_lengths'] = True
		config['n_ans'] = len(self.vocab['answer_token_to_idx'])
		config['ques_start_id'] = preprocess_utils.SPECIAL_TOKENS['<START>']
		config['ques_end_id'] = preprocess_utils.SPECIAL_TOKENS['<END>']
		config['ques_null_id'] = preprocess_utils.SPECIAL_TOKENS['<NULL>']
		config['ques_unk_id'] = preprocess_utils.SPECIAL_TOKENS['<UNK>']
		config['ques_vocab_sz'] = len(self.vocab['question_token_to_idx'])
		config['attn_height'] = self.attn_height
		config['attn_width'] = self.attn_width

		return config

	def index_choices(self):

		"""
		Index the set of choices according to the answer dictionary and then delete the actual choices dictionary object
		"""

		self.choices = {}

		with open(self.choices_data_json_path, 'r') as f:
			choices_data = json.load(f)
			for qid in self.questions_keys:
				self.choices[qid] = {}

				# Index the valid answers
				self.choices[qid]['valid'] = preprocess_utils.encode(choices_data[qid]['valid'], self.vocab['answer_token_to_idx'], allow_unk=True)

				# Index the plausible answers
				self.choices[qid]['plausible'] = preprocess_utils.encode(choices_data[qid]['plausible'], self.vocab['answer_token_to_idx'], allow_unk=True)

		print('Indexing of Choices complete')
	
	def load_embeddings(self, vocab, emb_dim, word2vec_path):

		if not os.path.exists(word2vec_path):
			return None

		emb_mat = np.random.normal(scale=0.6, size=(len(vocab), emb_dim))

		with open(word2vec_path, 'r') as f:
			embedding_dict = json.load(f)

			for word in embedding_dict:
				emb_mat[vocab[word]] = np.array(embedding_dict[word])

		return torch.as_tensor(emb_mat, dtype=torch.float)

	def __len__(self):
		return len(self.questions_keys)

	def __getitem__(self, idx):
		if idx >= len(self):
			raise ValueError('index %d out of range (%d)' % (idx, len(self)))
		
		key = self.questions_keys[idx]
		image_features_h5 = h5py.File(self.image_features_path, 'r')['features']

		# QA
		question = self.questions[key]['question']
		answer = self.questions[key].get('answer', '<NULL>')

		# Attention
		attn_mask = np.zeros((self.attn_height, self.attn_width), dtype=np.float32)
		if "scene" in self.questions[key]['semanticStr']:
			attn_mask += 1
		q_attn_obj = list(self.questions[key]['annotations']['question'].values())
		a_attn_obj = list(self.questions[key]['annotations']['fullAnswer'].values())

		# Encode question and answer (tokenize as well)
		question_tokens = preprocess_utils.tokenize(question,
												punct_to_keep=[';', ','],
												punct_to_remove=['?', '.'])

		ques_len = len(question_tokens) if len(question_tokens) <= self.meta_data['max_ques_len'] else self.meta_data['max_ques_len'] 
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

		obj_wrds_mat = np.zeros((self.meta_data['max_num_objs']))
		num_relations = len(self.sg_vocab['relation_token_to_idx'])
		objects = np.zeros((self.meta_data['max_num_objs'], 4), dtype=np.float32) - 1
		obj_region_mask = np.zeros((self.meta_data['max_num_objs'], self.attn_height, self.attn_width), dtype=np.float32)


		if self.args.use_rel_emb:
			A = np.zeros((self.meta_data['max_num_objs'], self.meta_data['max_num_objs'] * num_relations))
		else:
			A = np.zeros((self.meta_data['max_num_objs'], self.meta_data['max_num_objs']))


		object_keys = list(sg['objects'].keys())
		for num_objs, obj_key in enumerate(object_keys):

			if num_objs >= self.meta_data['max_num_objs']:
				break

			obj = sg['objects'][obj_key]
			objects[num_objs][0] = max(obj['x'] / width, 0.0)
			objects[num_objs][1] = max(obj['y'] / height, 0.0)
			objects[num_objs][2] = min((obj['x'] + obj['w']) / width, 1.0)
			objects[num_objs][3] = min((obj['y'] + obj['h']) / height, 1.0)

			x1 = int(objects[num_objs][0] * self.attn_width)
			y1 = int(objects[num_objs][1] * self.attn_height)
			x2 = int(objects[num_objs][2] * self.attn_width)
			y2 = int(objects[num_objs][3] * self.attn_height)
			obj_region_mask[num_objs][y1:y2, x1:x2] = 1.0
				
			# Attended Object
			if obj_key in q_attn_obj:
				attn_mask += obj_region_mask[num_objs]
			if obj_key in a_attn_obj:
				attn_mask += obj_region_mask[num_objs]


			obj_wrds_mat[num_objs] = preprocess_utils.encode([obj['name']],
												 self.sg_vocab['object_token_to_idx'],
												 allow_unk=True)[0]

			# print (obj['x'], obj['y'], obj['x'] + obj['w'], obj['y'] + obj['h'], objects[num_objs])
			for relation in obj["relations"]:
				rel_encoded = preprocess_utils.encode([relation['name']],
												 self.sg_vocab['relation_token_to_idx'],
												 allow_unk=True)
				assert len(rel_encoded) == 1
				obj_id = object_keys.index(relation['object'])

				if obj_id >= self.meta_data['max_num_objs']:
					continue

				A_id = rel_encoded[0] * self.meta_data['max_num_objs'] + obj_id
				
				if self.args.use_rel_emb:
					A[num_objs][A_id] = 1 # can give relation id in OxO matrix if needed
				elif self.args.use_rel_words:
					A[num_objs][obj_id] = rel_encoded[0]
				else:
					A[num_objs][obj_id] = 1
	
		#Increase number of object to correct value (since indexed from 0)
		#num_objs += 1
		# Get image feature from image
		h5_idx = self.image_info[image_idx]['index']
		image_feat = torch.as_tensor(image_features_h5[h5_idx], dtype = torch.float)
		spatial_width = image_feat.size(-1)
		spatial_height = image_feat.size(-2)
		objects[:,0] *= (spatial_width-1)
		objects[:,2] *= (spatial_width-1)
		objects[:,1] *= (spatial_height-1)
		objects[:,3] *= (spatial_height-1)
		attn_mask /= max(1.0, np.max(attn_mask))

		if self.args.criterion == "bce":
			ans_output = torch.zeros(len(self.vocab['answer_token_to_idx']), dtype=torch.float32)
			ans_output[answer_encoded[0]] = 1
		else:
			ans_output = answer_encoded[0]
		
		if self.args.opt_met:

			valid_ans_mat = torch.zeros(len(self.vocab['answer_token_to_idx']), dtype=torch.float32)
			plausible_ans_mat = torch.zeros(len(self.vocab['answer_token_to_idx']), dtype=torch.float32)

			for ans_id in self.choices[key]['valid']:
				valid_ans_mat[ans_id] = 1

			for ans_id in self.choices[key]['plausible']:
				plausible_ans_mat[ans_id] = 1

		data_obj = {
				'ques': torch.as_tensor(question_encoded, dtype=torch.long),
				'ans': ans_output,
				'ques_lens': ques_len,
				'obj_bboxes': torch.as_tensor(objects, dtype=torch.float),
				'num_objs': num_objs,
				'A': torch.as_tensor(A, dtype=torch.float),
				'image_feat': torch.as_tensor(image_feat, dtype=torch.float),
				'ques_id': idx,
				'obj_wrds': torch.as_tensor(obj_wrds_mat, dtype=torch.long),
				'attn_mask': torch.as_tensor(attn_mask, dtype=torch.float),
				'obj_region_mask': torch.as_tensor(obj_region_mask, dtype=torch.float),
			}
		if self.args.opt_met:
			data_obj['valid_ans'] = valid_ans_mat
			data_obj['plausible_ans'] = plausible_ans_mat

		return data_obj
