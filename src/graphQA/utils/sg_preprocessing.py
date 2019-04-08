import os
import argparse
import json
import numpy as np

import preprocess as preprocess_utils
import utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_relations_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=0, type=int)
parser.add_argument('--output_vocab_json', default='')


def main(args):
	if (args.output_vocab_json == ''):
		print('Must give output_vocab_json')
		return

	print('Loading scene graph data')
	with open(args.input_relations_json, 'r') as f:
		scenegraphs = json.load(f)

	# Either create the vocab
	if args.input_vocab_json == '' or args.expand_vocab == 1:
		print('Building relation vocab')
		relations = []
		for sg_key in scenegraphs.keys():
			sg = scenegraphs[sg_key]
			for obj_key in sg['objects'].keys():
				obj = sg['objects'][obj_key]
				for rel in obj['relations']:
					relations.append(rel['name'])
		relation_token_to_idx = preprocess_utils.build_vocab(
			relations, delim=None, min_token_count=args.unk_threshold,
			punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
		vocab = {'relation_token_to_idx': relation_token_to_idx}
	else:
		print ('Nothing to do! Input vocab already given; Expansion not given as parameter')
		return

	# Either create the vocab
	if args.input_vocab_json != '' and args.expand_vocab == 1:
		new_vocab = vocab
		with open(args.input_vocab_json, 'r') as f:
			vocab = json.load(f)
		if args.expand_vocab == 1:
			num_new_words = 0
			for word in new_vocab['relation_token_to_idx']:
				if word not in vocab['relation_token_to_idx']:
					print('Found new word %s' % word)
					idx = len(vocab['relation_token_to_idx'])
					vocab['relation_token_to_idx'][word] = idx
					num_new_words += 1
			print('Found %d new words' % num_new_words)

	utils.mkdirs(os.path.dirname(args.output_vocab_json))
	with open(args.output_vocab_json, 'w') as f:
		json.dump(vocab, f)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)