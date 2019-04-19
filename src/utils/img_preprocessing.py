import os
import argparse
import json
import h5py
import numpy as np

import torch
from torch import nn as nn

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from . import utils as utils

parser = argparse.ArgumentParser()
parser.add_argument('--traindir', required=True)
parser.add_argument('--output_image_feats_path', required=True)
parser.add_argument('--output_info_json_path', required=True)

parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--workers', default=1)

class ImgModel(nn.Module):
	def __init__(self, args):
		super(ImgModel, self).__init__()
		base_model = models.resnet101(pretrained=True)
		self.layers = nn.Sequential(*list(base_model.children())[:-2])
		self.layers.eval()

	def forward(self, input):
		return self.layers(input)


def main(args):
	
	f = h5py.File(args.output_image_feats_path, "w")

	# Data loading code
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	fixed_size = (640,480)
	resize = transforms.Resize(fixed_size, interpolation=2)
	gqa_dataset = datasets.ImageFolder(args.traindir, transforms.Compose([resize,transforms.ToTensor(),normalize,]))
	
	img_dict = {}
	for img_id, (img_path, _) in enumerate(gqa_dataset.imgs):
		_, filename = os.path.split(img_path)
		img_dict[filename[:-4]] = {'index' : img_id}
	
	#Save image_spatial_info
	utils.mkdirs(os.path.dirname(args.output_info_json_path))
	with open(args.output_info_json_path, 'w') as image_info_f:
		json.dump(img_dict, image_info_f)

	train_loader = torch.utils.data.DataLoader(gqa_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
	
	# Create dataset
	t = len(img_dict)
	dset = f.create_dataset("features", (t, 2048, fixed_size[0]//32,fixed_size[1]//32), maxshape=(None, 2048, fixed_size[0]//32,fixed_size[1]//32), chunks=(args.batch_size, 2048, fixed_size[0]//32,fixed_size[1]//32))
	
	# Model
	model = ImgModel(args).cuda()
	total_processed = 0
	for i, (input_var, target) in enumerate(train_loader):
		print ("Processed {}/{}".format(i*args.batch_size, t))
		output = model(input_var.cuda())
		bs, _, _, _ = output.size()
		output = output.data.cpu().numpy()
		dset[total_processed: total_processed+bs,:,:,:] = output
		total_processed += bs
		# save output
	f.close()

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)