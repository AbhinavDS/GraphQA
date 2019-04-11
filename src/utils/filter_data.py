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
	parser.add_argument('--data', type=str, help="path to the data directory where the QA json file and image features are stored")
	parser.add_argument('--out_data', type=str, help="Path of the directory where the filtered data will be stored")

	return parser.parse_args()

def filter_img_feats(images, args):

	"""
	Filter the features based on the 
	@param images: list of image ids which need to kept in the final version of the dataset.
	"""

	feat_names = ['objects', 'spatial']

	img_sz = len(images)
	spec = {
		"spatial": {"features": (img_sz, 2048, 7, 7)},
		"objects": {"features": (img_sz, 100, 2048),
					"bboxes": (108077, 100, 4)}
	}

	for fname in feat_names:
		
		inp_file_path = os.path.join(args.data, "gqa_{}.h5".format(fname))
		out_file_path = os.path.join(args.out_data, "gqa_{}.h5".format(fname))
		inp_info_file_path = os.path.join(args.data, "gqa_{}_merged_info.json".format(fname))
		out_info_file_path = os.path.join(args.out_data, "gqa_{}_merged_info.json".format(fname))

		idx = 0

		info = {}
		with open(inp_info_file_path, 'r') as f:
			inp_info = json.load(f)

		# Throw away information for Image IDs that are not required
		for img_id in images:
			info[img_id] = inp_info[img_id]

		del inp_info

		with h5py.File(out_file_path) as out:
			datasets  = {}
			for dname in spec[fname]:
				datasets[dname] = out.create_dataset(dname, spec[fname][dname])

			with h5py.File(inp_file_path) as feats:
				for i in tqdm(range(img_sz)):
					
					for dname in spec[fname]:
						datasets[dname][idx] = feats[dname][info[images[i]]['index']]
					
					# Update the Index in the information object
					info[images[i]]['index'] = idx
					idx += 1

		print("Final Index", idx)
		with open(out_info_file_path, "w") as f:
			json.dump(info, f)

def filter_qa(split, args):

	"""
	Implement the filtering criteria to filter the QA pairs here and return the list of images satisfying that criteria.
	"""
	
	print("Processing the {} split".format(split))

	data_path = os.path.join(args.data, "{}_{}_data.json".format(args.dataset, split))

	with open(data_path, 'r') as f:
		data = json.load(f)
		
	# Current Function is to first filter all the questions of query type

	filtered_data = {'questions': []}

	filtered_data['questions'] = [x for x in data['questions'] if x['type'] == 'query']

	sample_sz = int(0.1*len(filtered_data['questions']))
	permut = np.random.permutation(len(filtered_data['questions'])).tolist()
	permut = permut[:sample_sz]

	sample_data = {'questions': []}
	for idx in permut:
		sample_data['questions'].append(filtered_data['questions'][idx])

	print("No of QA pairs: ", len(sample_data['questions']))
	
	images = [ x['imageId'] for x in sample_data['questions'] ] 
	
	print("No of Images: ", len(set(images)))

	with open(os.path.join(args.out_data, '{}_{}_data.json'.format(args.dataset, split)), 'w') as f:
		json.dump(sample_data, f)

	return list(set(images))

if __name__ == "__main__":

	args = get_args()
	train_images = filter_qa('train', args)
	val_images = filter_qa('val', args)

	images = train_images + val_images

	filter_img_feats(images, args)

