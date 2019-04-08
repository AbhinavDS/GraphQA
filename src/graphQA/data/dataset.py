import numpy as np
import json
import h5py
import torch
from torch.utils.data import Dataset

import graphQA.utils.preprocess as preprocess_utils
import graphQA.utils.utils as utils

class GQADataset(Dataset):

	def __init__(self, question_json_path, scenegraph_json_path, image_features_path, image_info_json_path, vocab_json, relations_vocab_json, num_objects=150):
		
		q = open(question_json_path, 'r')
		self.questions = json.load(q)
		self.questions_keys = list(self.questions.keys())
		q.close()
		
		sg = open(scenegraph_json_path, 'r')
		self.scenegraphs = json.load(sg)
		sg.close()	

		info = open(image_info_json_path, 'r')
		self.image_info = json.load(info)
		info.close()	
		self.image_features_h5 = h5py.File(image_features_path, 'r')['features']
		
		self.vocab = utils.load_vocab(vocab_json)
		self.relations_vocab = utils.load_vocab(relations_vocab_json)

		self.num_objects = num_objects

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):
		if idx >= len(self):
			raise ValueError('index %d out of range (%d)' % (idx, len(self)))
		
		key = self.questions_keys[idx]

		# QA
		question = self.questions[key]['question']
		answer = self.questions[key].get('answer', None)
		# Encode question and answer (tokenize as well)
		question_tokens = preprocess_utils.tokenize(question,
												punct_to_keep=[';', ','],
												punct_to_remove=['?', '.'])
		question_encoded = preprocess_utils.encode(question_tokens,
												 self.vocab['question_token_to_idx'],
												 allow_unk=True)
		answer_encoded = preprocess_utils.encode([answer],
												 self.vocab['answer_token_to_idx'],
												 allow_unk=True)
		# SG
		image_idx = self.questions[key]['imageId']
		sg = self.scenegraphs[image_idx]
		width, height = (float)(sg['width']), (float)(sg['height'])
		# Get object dims for roi pooling
		# Create adjacency matrix with relations
		num_relations = len(self.relations_vocab['relation_token_to_idx'])
		A = np.zeros((self.num_objects, self.num_objects * num_relations))
		objects = np.zeros((self.num_objects, 4))
		counter = 0
		object_keys = list(sg['objects'].keys())
		for obj_key in object_keys:
			obj = sg['objects'][obj_key]
			objects[counter][0] = obj['x'] / width
			objects[counter][1] = obj['y'] / height
			objects[counter][2] = (obj['x'] + obj['w']) / width
			objects[counter][3] = (obj['y'] + obj['h']) / height
			for relation in obj["relations"]:
				rel_encoded = preprocess_utils.encode([relation['name']],
												 self.relations_vocab['relation_token_to_idx'],
												 allow_unk=True)
				assert len(rel_encoded) == 1
				obj_id = object_keys.index(relation['object'])
				A_id = rel_encoded[0] * self.num_objects + obj_id
				A[counter][A_id] = 1# can give relation id in OxO matrix if needed
			counter += 1
			if counter > self.num_objects:
				break

		# Get image feature from image
		# TODO: Check
		h5_idx = self.image_info[image_idx]['index']
		image_feat = self.image_features_h5[h5_idx]
		
		#TODO: Convert to torch either here or at point
		return question_encoded, answer_encoded, A, objects, image_feat
