"""
Module that filters the dataset and the corresponding image and 
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="size changed")


import h5py
import argparse
from tqdm import tqdm
import json
import os
import numpy as np

def get_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help="prefix of the QA dataset json file")
	parser.add_argument('--inp_qa_data_dir', type=str, help="Path to the data directory where the QA json file is stored")
	parser.add_argument('--inp_sg_data_dir', type=str, help="Path to the data directory where the Scene Graph json files are stored")
	parser.add_argument('--out_data_dir', type=str, help="Path of the directory where the filtered data will be stored")
	parser.add_argument('--pct', type=int, help="Percentage of original to be used")
	parser.add_argument('--choices', type=str, required=True, help="The path to val choices file that will be used for evaluating metrics other than accuracy")

	return parser.parse_args()

def save_scene_graphs(img_ids, inp_sg, inp_split, out_split, args):

	out_sg = {}
	for idx in img_ids:
		out_sg[idx] = inp_sg[idx]

	with open(os.path.join(args.out_data_dir, '{}_sceneGraphs.json'.format(out_split)), 'w') as out_sgf:
		json.dump(out_sg, out_sgf, indent=4)

def write_data(data, img_ids, qa_ids, split_name, args):

	new_qa_ids = set([ x for x in qa_ids if data[x]['imageId'] in img_ids ])

	new_data = {}
	for idx in new_qa_ids:
		qdata = data[idx]

		# Filter the set of entailed questions
		qdata['entailed'] = list(set(data[idx]['entailed']) & new_qa_ids)

		# Filter the set of equivalent questions
		qdata['equivalent'] = list(set(data[idx]['equivalent']) & new_qa_ids)

		new_data[idx] = qdata
		
	print('{} QA Size: {}, No. of Images: {}'.format(split_name, len(new_data), len(img_ids)))

	with open(os.path.join(args.out_data_dir, '{}_{}_data.json'.format(args.dataset, split_name)), 'w') as f:
		json.dump(new_data, f, indent=4)

def write_choices(qa_ids, split_name, args):
	"""
	Filter the choices from the original file based on the questions present in the current qa_ids split.
	"""

	with open(args.choices, 'r') as f:
		choices = json.load(f)

	new_choices = {}
	for qid in qa_ids:
		new_choices[qid] = choices[qid]

	with open(os.path.join(args.out_data_dir, '{}_choices.json'.format(split_name)), 'w') as f:
		json.dump(new_choices, f)

def filter_qa(split, args):

	"""
	Implement the filtering criteria to filter the QA pairs here and return the list of images satisfying that criteria.
	"""
	
	print("Processing the {} split".format(split))

	data_path = os.path.join(args.inp_qa_data_dir, "{}_{}_questions.json".format(split, args.dataset))

	with open(data_path, 'r') as f:
		data = json.load(f)

	with open(os.path.join(args.inp_sg_data_dir, '{}_sceneGraphs.json'.format(split)), 'r') as inp_sgf:
		inp_sg = json.load(inp_sgf)
		

	with open(os.path.join(args.inp_sg_data_dir, 'invalid_img_ids.json'), 'r') as invf:
		invalid_img_ids = json.load(invf)

	# Current Function is to first filter all the questions of query type
	if split == "train":
		pct = float(args.pct) / 80
	else:
		pct = float(args.pct) / 100

	qa_ids = [ x for x in data if data[x]['types']['structural'] == 'query' ]
	filtered_image_ids = list(set([ data[x]['imageId'] for x in qa_ids ]) - set(invalid_img_ids))
	image_ids = []
	for img_id in filtered_image_ids:
		for obj in inp_sg[img_id]['objects']:
			if len(inp_sg[img_id]['objects'][obj]['relations']) > 0:
				image_ids.append(img_id)
				break

	qa_sz = len(qa_ids)
	img_sz = len(image_ids)
	sample_sz = int(pct*img_sz)
	
	print('After Type filtering, QA Size: {}, Img Size: {}'.format(qa_sz, img_sz))

	perm = np.random.permutation(img_sz).tolist()[:sample_sz]
	permute = []
	for idx in perm:
		permute.append(image_ids[idx])

	if split == "train":
		train_sz = int(0.8*sample_sz)
		train_ids = permute[:train_sz]
		val_ids = permute[train_sz+1:]
		
		write_data(data, train_ids, qa_ids, 'train', args)
		write_data(data, val_ids, qa_ids, 'val', args)

		save_scene_graphs(train_ids, inp_sg, 'train', 'train', args)
		save_scene_graphs(val_ids, inp_sg, 'train', 'val', args)

	else:
		
		write_data(data, permute, qa_ids, 'test', args)
		save_scene_graphs(permute, inp_sg, 'val', 'test', args)
		write_choices(qa_ids, 'test', args)

if __name__ == "__main__":

	args = get_args()

	if not os.path.exists(args.out_data_dir):
		os.makedirs(args.out_data_dir)
	
	filter_qa('train', args)
	filter_qa('val', args)
