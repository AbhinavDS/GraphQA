import os
import json
import numpy as np
import torch

def batch_roiproposals(rois):
	bs, O, dim = rois.size()
	append_id = np.zeros((bs,O,1), dtype=np.float64)
	for i in range(bs):
		append_id[i] = i
	new_rois = torch.cat((torch.as_tensor(append_id), rois), dim=2)
	return new_rois

def mkdirs(paths):
	if isinstance(paths, list):
		for path in paths:
			if not os.path.exists(path):
				os.makedirs(path)
	else:
		if not os.path.exists(paths):
			os.makedirs(paths)


def invert_dict(d):
  return {v: k for k, v in d.items()}
  

def load_vocab(path):
	with open(path, 'r') as f:
		vocab = json.load(f)
		if vocab.get('question_token_to_idx', None) is not None:
			vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
			# Sanity check: make sure <NULL>, <START>, and <END> are consistent
			assert vocab['question_token_to_idx']['<NULL>'] == 0
			assert vocab['question_token_to_idx']['<START>'] == 1
			assert vocab['question_token_to_idx']['<END>'] == 2
			
		if vocab.get('answer_token_to_idx', None) is not None:
			vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
			# Sanity check: make sure <NULL>, <START>, and <END> are consistent
			assert vocab['answer_token_to_idx']['<NULL>'] == 0
			assert vocab['answer_token_to_idx']['<START>'] == 1
			assert vocab['answer_token_to_idx']['<END>'] == 2

		if vocab.get('relation_token_to_idx', None) is not None:
			vocab['relation_idx_to_token'] = invert_dict(vocab['relation_token_to_idx'])
			# Sanity check: make sure <NULL>, <START>, and <END> are consistent
			assert vocab['relation_token_to_idx']['<NULL>'] == 0
			assert vocab['relation_token_to_idx']['<START>'] == 1
			assert vocab['relation_token_to_idx']['<END>'] == 2
	return vocab


def load_scenes(scenes_json):
	with open(scenes_json) as f:
		scenes_dict = json.load(f)['scenes']
	scenes = []
	for s in scenes_dict:
		table = []
		for i, o in enumerate(s['objects']):
			item = {}
			item['id'] = '%d-%d' % (s['image_index'], i)
			if '3d_coords' in o:
				item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
									np.dot(o['3d_coords'], s['directions']['front']),
									o['3d_coords'][2]]
			else:
				item['position'] = o['position']
			item['color'] = o['color']
			item['material'] = o['material']
			item['shape'] = o['shape']
			item['size'] = o['size']
			table.append(item)
		scenes.append(table)
	return scenes
	

def load_embedding(path):
	return torch.Tensor(np.load(path))