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
	return parser.parse_args()

def save_scene_graphs(img_ids, inp_split, out_split, args):

	with open(os.path.join(args.inp_sg_data_dir, '{}_sceneGraphs.json'.format(inp_split)), 'r') as inp_sgf:
		inp_sg = json.load(inp_sgf)

	out_sg = {}
	for idx in img_ids:
		out_sg[idx] = inp_sg[idx]

	with open(os.path.join(args.out_data_dir, '{}_sceneGraphs.json'.format(out_split)), 'w') as out_sgf:
		json.dump(out_sg, out_sgf, indent=4)

def write_data(data, img_ids, qa_ids, split_name, args):

	new_qa_ids = set([ x for x in qa_ids if data[x]['imageId'] in img_ids ])

	new_data = {}
	for idx in new_qa_ids:
		new_data[idx] = data[idx]

	print('{} QA Size: {}, No. of Images: {}'.format(split_name, len(new_data), len(img_ids)))

	with open(os.path.join(args.out_data_dir, '{}_{}_data.json'.format(args.dataset, split_name)), 'w') as f:
		json.dump(new_data, f, indent=4)

def filter_qa(split, args):

	"""
	Implement the filtering criteria to filter the QA pairs here and return the list of images satisfying that criteria.
	"""
	
	print("Processing the {} split".format(split))

	data_path = os.path.join(args.inp_qa_data_dir, "{}_{}_questions.json".format(split, args.dataset))

	with open(data_path, 'r') as f:
		data = json.load(f)
		
	# Current Function is to first filter all the questions of query type
	if split == "train":
		pct = float(args.pct) / 80
	else:
		pct = float(args.pct) / 100

	qa_ids = [ x for x in data if data[x]['types']['structural'] == 'query' ]
	image_ids = list(set([ data[x]['imageId'] for x in qa_ids ]))

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

		save_scene_graphs(train_ids, 'train', 'train', args)
		save_scene_graphs(train_ids, 'train', 'val', args)

	else:
		
		write_data(data, permute, qa_ids, 'test', args)
		save_scene_graphs(permute, 'val', 'test', args)

if __name__ == "__main__":

	args = get_args()
	
	filter_qa('train', args)
	filter_qa('val', args)
