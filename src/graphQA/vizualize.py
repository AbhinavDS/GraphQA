"""
Generates attention visualizations for adding in the paper
"""

import skimage
from skimage import io
from skimage import transform
import numpy as np
import os
import json

gt_file = "/scratch/cluster/ankgarg/gqa/test_data/test_set/balanced_test_data.json"
sg_file = "/scratch/cluster/ankgarg/gqa/test_data/test_set/gold/test_sceneGraphs.json"
preds_file = "/scratch/cluster/ankgarg/gqa/graphQA_res/5p_pred_cls_g1_rel_probs/test_preds.json"
resize = False
alpha = 0.002
viz_dir = "/scratch/cluster/ankgarg/gqa/viz/"

def read_image(img_id):

	basepath = "/scratch/cluster/abhinav/gnlp/gqa_data/images/gqa/"
	img_path = os.path.join(basepath, str(img_id) + ".jpg")

	img = io.imread(img_path)
	#img = skimage.color.rgba2rgb(img)
	print('Image Shape', img_id, img.shape)
	return img

def gen_attn(obj, gt_data, mode):
	
	qid = obj['questionId']
	img_id = gt_data[qid]['imageId']

	print(mode, qid, img_id, gt_data[qid]['question'], gt_data[qid]['answer'], obj['prediction'])
	img = read_image(img_id)

	attn_map = np.zeros((img.shape[0], img.shape[1]))
	cnt = np.zeros((img.shape[0], img.shape[1]))
	
	attentions = obj['attention']
	
	print(len(attentions), len(attentions[0]))

	w,h = img.shape[0], img.shape[1]
	for att in attentions:

		lx = int(att[0]*w)
		ly = int(att[1]*h)
		hx = int(att[2]*w)
		hy = int(att[3]*h)
		
		attn_map[lx:hx+1, ly:hy+1] += att[4]
		cnt[lx:hx+1, ly:hy+1] += 1

	cnt_mx = np.maximum(1, cnt)
	#print(cnt_mx)
	attn_map = np.divide(attn_map, cnt_mx)

	attn_map = attn_map.reshape((img.shape[0], img.shape[1], 1))

	mi, mx = np.min(attn_map), np.max(attn_map)

	attn_map = (attn_map - mi) / float(mx - mi)

	#print(attn_map)
	final_img = alpha*img + (1.0 - alpha)*attn_map
	out_img_path = os.path.join(viz_dir, '{}_{}_{}.png'.format(mode, qid, img_id))
	io.imsave(out_img_path, final_img)

def get_qids(gt_data, preds):

	sz = len(preds)
	correct = []
	incorrect = []

	for qobj in preds:

		qid = qobj['questionId']
		gt = gt_data[qid]['answer']
		pr = qobj['prediction']

		if gt == pr:
			correct.append(qobj)
		else:
			incorrect.append(qobj)

	print(len(correct), len(incorrect))

	# Get 2 random correct question ids

	# Get 2 random incorrect question ids

	return correct[:20], incorrect[:10]

def read_data():

	with open(gt_file, 'r') as f:
		gt_data = json.load(f)

	with open(sg_file, 'r') as f:
		sg_data = json.load(f)

	with open(preds_file, 'r') as f:
		preds = json.load(f)

	cr, incr = get_qids(gt_data, preds)
	
	for cri in cr:
		gen_attn(cri, gt_data, 'correct')

	for incri in incr:
		gen_attn(incri, gt_data, 'incorrect')

read_data()