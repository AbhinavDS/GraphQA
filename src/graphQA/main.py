"""
Main module that controls the complete graph QA pipeline
"""

from config_parser import parse_args

if __name__ == "__main__":

	args = parse_args()
	args.gen_args()
	
	# Invoke the Data Loader and set the config variables with additional values
	args.set_args()

	if args.mode == "train":
		pass	
		trainer = Trainer(args, train_loader, val_loader)
		trainer.train()
	elif args.mode == "eval":
		pass
	else:
		raise("Specify correct operating Mode") 