import os
import argparse
import json
import numpy as np
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stop_words

from . import preprocess as preprocess_utils
from . import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--input_relations_path', required=True)
parser.add_argument('--inp_word2vec_path', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=0, type=int)
parser.add_argument('--output_vocab_path', default='', required=True)
parser.add_argument('--meta_data_path', default='', required=True)
parser.add_argument('--out_word2vec_dir', default='')

def store_embedding(vocab, word2vec_path):

	word2vec = {}
	args.stop_words = set(args.stop_words)
	args.stop_words.add('near')
	args.stop_words.add('alongside')
	args.stop_words = frozenset(args.stop_words)
	for key in vocab.keys():
		word2vec[key] = []
		split_key =  key.strip().split()
		if len(split_key) > 1:
			longest_word = ''   
			for token in split_key:
				if(len(longest_word) < len(token)):
					longest_word = token
				if token in args.stop_words:
					pass
				else:
					word2vec[key].append(token)
			if len(word2vec[key]) == 0:
				word2vec[key] = [longest_word]
		else:
			word2vec[key] = split_key
		if (len(word2vec[key]) != 1):
			#print (key, word2vec[key],longest_word)
			word2vec[key] = longest_word
		else:
			word2vec[key] = word2vec[key][0]

	reversemap = {}
	for key in word2vec:
		if word2vec[key] in reversemap:
			reversemap[word2vec[key]].append(key)
		else:
			reversemap[word2vec[key]] = [key]
	
	# Load the input word2vec vectors and store the ones that are actually present in the vocabulary.
	embeddings = {}
	with open(args.inp_word2vec_path, 'r') as f:

		for line in f.readlines():
			line = line.replace('\r', '').replace('\n', '').split()
			word = line[0].lower()
			vector = line[1:]
			if word in reversemap:
				for key in reversemap[word]:
					embeddings[key] = vector

	missed = len(set(vocab) - set(embeddings.keys()))
	print('Missed {}/{} words in Pre-Trained embeddings'.format(missed, len(vocab)))
	
	with open(word2vec_path, 'w') as f:
		json.dump(embeddings, f)

def main(args):
	if (args.output_vocab_path == ''):
		print('Must give output_vocab_path')
		return

	if (args.meta_data_path == ''):
		print('Must give meta_data_path')
		return

	print('Loading scene graph data')
	with open(args.input_relations_path, 'r') as f:
		scenegraphs = json.load(f)

	max_num_objs = 0
	meta_vocab = None

	# Either create the vocab
	if args.input_vocab_json == '' or args.expand_vocab == 1:
		print('Building relation vocab')
		relations = []
		for sg_key in scenegraphs.keys():
			sg = scenegraphs[sg_key]
			num_objs = len(sg['objects'])
			max_num_objs = num_objs if max_num_objs < num_objs else max_num_objs;
			for obj_key in sg['objects'].keys():
				obj = sg['objects'][obj_key]
				for rel in obj['relations']:
					relations.append(rel['name'])
		relation_token_to_idx = preprocess_utils.build_vocab(
			relations, delim=None, min_token_count=args.unk_threshold,
			punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
		vocab = {'relation_token_to_idx': relation_token_to_idx}

		print('Building object names vocab')
		object_names = []
		for sg_key in scenegraphs.keys():
			sg = scenegraphs[sg_key]
			num_objs = len(sg['objects'])
			max_num_objs = num_objs if max_num_objs < num_objs else max_num_objs;
			for obj_key in sg['objects'].keys():
				obj = sg['objects'][obj_key]
				object_names.append(obj['name'])
		object_token_to_idx = preprocess_utils.build_vocab(
			object_names, delim=None, min_token_count=args.unk_threshold,
			punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
		vocab['object_token_to_idx'] = object_token_to_idx

	else:
		print ('Nothing to do! Input vocab already given; Expansion not given as parameter')
		return

	# Either load the vocab
	if args.input_vocab_json != '' and args.expand_vocab == 1:
		new_vocab = vocab
		with open(args.input_vocab_json, 'r') as f:
			vocab = json.load(f)
		if args.expand_vocab == 1:
			
			for vocab_type in new_vocab:
				num_new_words = 0
				for word in new_vocab[vocab_type]:
					if word not in vocab[vocab_type]:
						print('Found new word %s' % word)
						idx = len(vocab[vocab_type])
						vocab[vocab_type][word] = idx
						num_new_words += 1
				print('Found %d new words for %' % (num_new_words, vocab_type))

	utils.mkdirs(os.path.dirname(args.output_vocab_path))
	with open(args.output_vocab_path, 'w') as f:
		json.dump(vocab, f)

	# Check if meta file exists
	if os.path.exists(args.meta_data_path):
		meta_vocab = utils.load_vocab(args.meta_data_path)
		meta_old = meta_vocab.get('max_num_objs',0)
		max_num_objs = meta_old if max_num_objs < meta_old else max_num_objs;
	if meta_vocab is None:
		meta_vocab = {}
	meta_vocab['max_num_objs'] = max_num_objs
	utils.mkdirs(os.path.dirname(args.meta_data_path))
	with open(args.meta_data_path, 'w') as f:
		json.dump(meta_vocab, f)

	if args.inp_word2vec_path != "":
		store_embedding(vocab['relation_token_to_idx'], os.path.join(args.out_word2vec_dir, 'rel_glove.300d.json'))
		store_embedding(vocab['object_token_to_idx'], os.path.join(args.out_word2vec_dir, 'obj_name_glove.300d.json'))

if __name__ == '__main__':
	args = parser.parse_args()
	args.stop_words = stop_words
	main(args)