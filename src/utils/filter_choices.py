"""
Filter the choices for the splits in the directory provided.
"""

import argparse
import json
import os

def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', required=True, help="Directory where the train and val set questions Ids are stored for a split")
	parser.add_argument('--src', required=True, help="Directory Path to choices file that will be used")

	return parser.parse_args()

def create_choices(src_choices, fl, args):

	split = fl.split('_')[1]
	print(fl, split)

	with open(os.path.join(args.data_dir, fl), 'r') as f:
		data = json.load(f)

		keys = list(data.keys())

	choices_data = {}
	for k in keys:
		choices_data[k] = src_choices[k]

	print('Writing Choices', split)
	with open(os.path.join(args.data_dir, '{}_choices.json'.format(split)), 'w') as f:
		json.dump(choices_data, f)

if __name__ == "__main__":

	args = parse_args()

	print('Reading Source Choices')
	with open(args.src, 'r') as f:
		src_choices = json.load(f)

	print('Initiating Filtering')
	filenames = ['balanced_train_data.json', 'balanced_val_data.json']

	for fl in filenames:
		if os.path.exists(os.path.join(args.data_dir, fl)):
			create_choices(src_choices, fl, args)