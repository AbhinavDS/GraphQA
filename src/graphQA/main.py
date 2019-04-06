"""
Main module that controls the complete graph QA pipeline
"""

from config_parser import parse_args

if __name__ == "__main__":

	args = parse_args()
	
	if args.mode == "train":
		pass	
		#trainer = Trainer(args)
		#trainer.train()

	elif args.mode == "eval":
		pass
	else:
		raise("Specify correct operating Mode") 