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
	parser.add_argument('--inp_data_dir', type=str, help="Path to the data directory where the QA json file is stored")
	parser.add_argument('--out_data_dir', type=str, help="Path of the directory where the filtered data will be stored")
	parser.add_argument('--pct', type=int, help="Percentage of original to be used")
	return parser.parse_args()

def write_data(data, ids, split_name, args):

	new_data = {}
	for idx in ids:
		new_data[idx] = data[idx]

	images = set([ new_data[x]['imageId'] for x in new_data])
	print('{} QA Size: {}, No. of Images: {}'.format(split_name, len(new_data), len(images)))

	with open(os.path.join(args.out_data_dir, '{}_{}_data.json'.format(args.dataset, split_name)), 'w') as f:
		json.dump(new_data, f)

def filter_qa(split, args):

	"""
	Implement the filtering criteria to filter the QA pairs here and return the list of images satisfying that criteria.
	"""
	
	print("Processing the {} split".format(split))

	data_path = os.path.join(args.inp_data_dir, "{}_{}_questions.json".format(split, args.dataset))

	with open(data_path, 'r') as f:
		data = json.load(f)
		
	# Current Function is to first filter all the questions of query type
	if split == "train":
		pct = float(args.pct) / 80
	else:
		pct = float(args.pct) / 100
		
	qa_ids = [x for x in data if data[x]['types']['structural'] == 'query']
	sz = len(qa_ids)
	print('After Type filtering, Size: {}'.format(sz))
	sample_sz = int(pct*sz)

	perm = np.random.permutation(sz).tolist()[:sample_sz]
	permute = []
	for idx in perm:
		permute.append(qa_ids[idx])

	if split == "train":
		train_sz = int(0.8*sample_sz)
		train_ids = permute[:train_sz]
		val_ids = permute[train_sz+1:]
		write_data(data, train_ids, 'train', args)
		write_data(data, val_ids, 'val', args)
	else:
		write_data(data, permute, 'test', args)

if __name__ == "__main__":

	args = get_args()
	filter_qa('train', args)
	filter_qa('val', args)