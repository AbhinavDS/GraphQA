import os
import argparse
import json
import numpy as np

from . import preprocess as preprocess_utils
from . import utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=0, type=int)
parser.add_argument('--output_vocab_json', default='')
parser.add_argument('--meta_vocab_json', default='')


def main(args):
	if (args.output_vocab_json == ''):
		print('Must give output_vocab_json')
		return
	
	if (args.meta_vocab_json == ''):
		print('Must give meta_vocab_json')
		return

	print('Loading QA data')
	with open(args.input_questions_json, 'r') as f:
		questions = json.load(f)

	max_ques_len = 0
	meta_vocab = None

	# Either create the vocab
	if args.input_vocab_json == '' or args.expand_vocab == 1:
		print('Building vocab')
		answer_token_to_idx = preprocess_utils.build_vocab((questions[key]['answer'] for key in questions.keys()), delim=None)
		question_token_to_idx = preprocess_utils.build_vocab((questions[key]['question'] for key in questions.keys()),min_token_count=args.unk_threshold, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
		question_lens = [len(preprocess_utils.tokenize(questions[key]['question'], punct_to_keep=[';', ','], punct_to_remove=['?', '.'])) for key in questions.keys()]
		max_ques_len = max(question_lens);
		vocab = {
			'question_token_to_idx': question_token_to_idx,
			'answer_token_to_idx': answer_token_to_idx,
			}
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
			for word in new_vocab['question_token_to_idx']:
				if word not in vocab['question_token_to_idx']:
					print('Found new word %s' % word)
					idx = len(vocab['question_token_to_idx'])
					vocab['question_token_to_idx'][word] = idx
					num_new_words += 1
			print('Found %d new words' % num_new_words)

	utils.mkdirs(os.path.dirname(args.output_vocab_json))
	with open(args.output_vocab_json, 'w') as f:
		json.dump(vocab, f)

	# Check if meta file exists
	if os.path.exists(args.meta_vocab_json):
		meta_vocab = utils.load_vocab(args.meta_vocab_json)
		meta_old = meta_vocab.get('max_ques_len',0)
		max_ques_len = meta_old if max_ques_len < meta_old else max_ques_len;
	if meta_vocab is None:
		meta_vocab = {}
	meta_vocab['max_ques_len'] = max_ques_len
	utils.mkdirs(os.path.dirname(args.meta_vocab_json))
	with open(args.meta_vocab_json, 'w') as f:
		json.dump(meta_vocab, f)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)