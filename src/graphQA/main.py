"""
Main module that controls the complete graph QA pipeline
"""

from .config_parser import parse_args
from .trainer import Trainer
from .mac_trainer import Trainer as MacTrainer
from .rl_trainer import Trainer as RLTrainer
from.evaluator import Evaluator
from .mac_evaluator import Evaluator as MacEvaluator
from .rl_evaluator import Evaluator as RLEvaluator
from .data.dataset import GQADataset

if __name__ == "__main__":

	args = parse_args()
	args.gen_config()

	if args.mode == "train":
		train_dataset = GQADataset(args, qa_data_key='train', sg_data_key='train')
		val_dataset = GQADataset(args, qa_data_key='val', sg_data_key='val')

		args.set_config(train_dataset.get_data_config())
		print(args)

		if args.use_mac:
			trainer = MacTrainer(args, train_dataset, val_dataset)
		elif args.use_rl:
			trainer = RLTrainer(args, train_dataset, val_dataset)
		else:
			trainer = Trainer(args, train_dataset, val_dataset)
		
		trainer.train()
	
	elif args.mode == "eval":
		val_dataset = GQADataset(args, qa_data_key='test', sg_data_key='test')
		args.set_config(val_dataset.get_data_config())
		
		if args.use_mac:
			evaluator = MacEvaluator(args, val_dataset)
		elif args.use_rl:
			evaluator = RLEvaluator(args, val_dataset)
		else:
			evaluator = Evaluator(args, val_dataset)
		
		evaluator.eval()
	
	else:
		raise("Please specify the correct training mode")