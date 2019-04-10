"""
Main module that controls the complete graph QA pipeline
"""

from torch.utils.data import DataLoader

from .config_parser import parse_args
from .trainer import Trainer
from .data.dataset import GQADataset

if __name__ == "__main__":

	args = parse_args()
	args.gen_config()

	if args.mode == "train":
		train_dataset = GQADataset(args.qa_data_path, args.sc_data_path, args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path)
		val_dataset = GQADataset(args.val_qa_data_path, args.val_sc_data_path, args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path)

		args.set_config(train_dataset.get_data_config())
	
	elif args.mode == "eval":
		val_dataset = GQADataset(args.val_qa_data_path, args.val_sc_data_path, args.img_feat_data_path, args.img_info_path, args.word_vocab_path, args.rel_vocab_path)
		
		args.set_config(val_dataset.get_data_config())
	
	else:
		raise("Please specify the correct training mode")
	
	if args.mode == "train":
		train_loader = DataLoader(dataset = train_dataset, batch_size=args.bsz, shuffle=True, num_workers=1)

		val_loader = DataLoader(dataset=val_dataset, batch_size=args.bsz, shuffle=True, num_workers=1)
		trainer = Trainer(args, train_loader, val_loader)
		trainer.train()
	
	elif args.mode == "eval":
		val_loader = DataLoader(dataset=val_dataset, batch_size=args.bsz, shuffle=True, num_workers=1)
		# Read the configs from the model
		pass
	
	else:
		raise("Specify correct operating Mode") 