"""
Main module that controls the complete graph QA pipeline
"""

from .config_parser import parse_args
from .trainer import Trainer
from .data.dataset import GQADataset

if __name__ == "__main__":

	args = parse_args()
	args.gen_config()

	if args.mode == "train":
		train_dataset = GQADataset(args.qa_data_path['val'], args.sg_data_path['train'], args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path, args.meta_data_path)
		val_dataset = GQADataset(args.qa_data_path['val'], args.sg_data_path['train'], args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path, args.meta_data_path)

		args.set_config(train_dataset.get_data_config())
	
	elif args.mode == "eval":
		val_dataset = GQADataset(args.qa_data_path['test'], args.sg_data_path['val'], args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path, args.meta_data_path)
				
		args.set_config(val_dataset.get_data_config())
	
	else:
		raise("Please specify the correct training mode")
	
	if args.mode == "train":
		
		trainer = Trainer(args, train_dataset, val_dataset)
		trainer.train()
	
	elif args.mode == "eval":
		# Read the configs from the model
		pass
	
	else:
		raise("Specify correct operating Mode") 