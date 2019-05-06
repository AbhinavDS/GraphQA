"""
Module that creates the files in format required by modules to prepare GQA dataset in VG format

TODO: 

1. Read all SG files to add all image files to the preprocessing
2. Compute Mean, Variance of the images in the train set
3. Incorporating Split Information in these combined files
"""

import os
import argparse
import json

splits = ['train', 'val', 'test']

def parse_args():

	"""
	Parse the commnad line arguments passed to the module
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', help="The path of the trainset ")

	parser.add_argument('--out_dir', help="Output directory where all generated files will be stored")

	return parser.parse_args()

def read_sg_data(args):

	"""
	Read the SG data file
	"""

	sg_data = {}
	for spl in splits:

		if spl == 'test':
			with open(os.path.join(args.data_dir, '..', 'test_set', 'gold', 'test_sceneGraphs.json'), 'r') as f:
				sg_data['test'] = json.load(f)
			continue

		with open(os.path.join(args.data_dir, 'gold', '{}_sceneGraphs.json').format(spl), 'r') as f:
			sg_data[spl] = json.load(f)

	return sg_data

def create_obj_data(sg_data_spl, args):

	"""
	Create a list of JSON objects where each object is for a image.
	For each image, we create a schema similar to for objects.json at: https://visualgenome.org/api/v0/api_readme
	TODO: Confirm the assumptions in the schema with one another
	"""

	obj_data = []
	obj_map = {}
	obj_list = set()

	for spl in sg_data_spl:

		sg_data = sg_data_spl[spl]
	
		for img_id in sg_data:

			img_dict = {
				'image_id': int(img_id),
				'split': spl,
				'objects': []
			}

			obj_map[int(img_id)] = {}

			for obj_id in sg_data[img_id]['objects']:
				
				obj = sg_data[img_id]['objects'][obj_id]

				obj_dict = {
					'object_id': int(obj_id),
					'x': obj['x'],
					'y': obj['y'],
					'w': obj['w'],
					'h': obj['h'],
					'names': [obj['name']]
				}

				if spl == "train":
					obj_list.add(obj['name'])

				img_dict['objects'].append(obj_dict)
				obj_map[int(img_id)][int(obj_id)] = obj_dict

			obj_data.append(img_dict)

	with open(os.path.join(args.out_dir, 'obj_data.json'), 'w') as f:
		json.dump(obj_data, f)

	with open(os.path.join(args.out_dir, 'obj_list.txt'), 'w') as f:
		for obj in obj_list:
			f.write("{}\n".format(obj))

	print(len(obj_list))

	return obj_map

def create_rel_data(sg_data_spl, obj_map, args):

	"""
	Create List of Relation data for each image in the dataset based on the schema of Visual Genome given at its weblink

	TODO: Confirm the assumptions made during conversion with one another
	"""

	rel_data = []
	rel_idx = 1
	pred_list = set()

	for spl in sg_data_spl:

		sg_data = sg_data_spl[spl]
	
		for img_id in sg_data:

			img_dict = {
				'image_id': int(img_id),
				'split': spl, 
				'relationships': []
			}

			for obj_id in sg_data[img_id]['objects']:

				obj = sg_data[img_id]['objects'][obj_id]

				for rel in obj['relations']:

					rel_dict = {
						'relationship_id': rel_idx,
						'predicate': rel['name'],
						'subject': obj_map[int(img_id)][int(obj_id)],
						'object': obj_map[int(img_id)][int(rel['object'])]
					}

					rel_idx += 1

					img_dict['relationships'].append(rel_dict)

					if spl == "train":
						pred_list.add(rel['name'])

			rel_data.append(img_dict)

	with open(os.path.join(args.out_dir, 'rel_data.json'), 'w') as f:
		json.dump(rel_data, f, indent = 4)

	with open(os.path.join(args.out_dir, 'pred_list.txt'), 'w') as f:
		for pred in pred_list:
			f.write("{}\n".format(pred))

	print(len(pred_list))

def create_img_data(sg_data_spl, args):

	"""
	Create an Image Data Json file which is a list of JSON objects. Each object has the following keys:
	1. width
	2. height
	3. image_id

	Not required:
	- coco_id
	- flickr_id
	- url
	"""

	img_data = []

	for spl in sg_data_spl:

		sg_data = sg_data_spl[spl]
		print(spl, len(sg_data))

		for img_id in sg_data:

			img_obj = {
				'image_id': int(img_id),
				'width': sg_data[img_id]['width'],
				'height': sg_data[img_id]['height'],
				'split': spl
			}

			img_data.append(img_obj)

	with open(os.path.join(args.out_dir, 'img_metadata.json'), 'w') as f:
		json.dump(img_data, f)

if __name__ == "__main__":

	args = parse_args()

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	sg_data = read_sg_data(args)
	create_img_data(sg_data, args)
	obj_map = create_obj_data(sg_data, args)
	create_rel_data(sg_data, obj_map, args)