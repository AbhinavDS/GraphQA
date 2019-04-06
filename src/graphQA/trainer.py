"""
Module that controls the training of the graphQA module
"""

from tensorboardX import SummaryWriter

class Trainer:

	def __init__(self, args):

		self.args = args

		if args.log:
			writer = SummaryWriter(args.log_dir)

	def train(self):

		