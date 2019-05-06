import torch
import torch.nn as nn
from ..modules.git_attention import Attention, NewAttention
from ..modules.git_language_model import WordEmbedding, QuestionEmbedding
from ..modules.git_classifier import SimpleClassifier
from ..modules.git_fc import FCNet

from question_parse.models.encoder import Encoder as QuesEncoder
from roi_pooling.functions.roi_pooling import roi_pooling_2d
import utils.utils as utils

class BaseModel2(nn.Module):
	def __init__(self, args, word2vec=None, rel_word2vec=None):
		super(BaseModel2, self).__init__()
		self.w_emb = WordEmbedding(args.ques_vocab_sz, args.ques_word_vec_dim)
		self.q_emb = QuestionEmbedding(args.ques_word_vec_dim, args.n_attn, 1, False)
		self.v_att = Attention(args.n_img_feats, self.q_emb.num_hid, args.n_attn, args.device)
		self.q_net = FCNet([args.n_attn, args.n_attn])
		self.v_net = FCNet([args.n_img_feats, args.n_attn])
		self.classifier = SimpleClassifier(args.n_attn, 2 * args.n_attn, args.n_ans, 0.5)

		self.roi_output_size = (5,5)
		self.avg_layer = nn.AvgPool2d(self.roi_output_size)
		self.device = args.device
		self.bidirectional = args.bidirectional
		self.ques_encoder = QuesEncoder(args.ques_vocab_sz, args.max_ques_len, args.ques_word_vec_dim, args.n_ques_emb, args.n_ques_layers, input_dropout_p=args.drop_prob, dropout_p=args.drop_prob, bidirectional=args.bidirectional, variable_lengths=args.variable_lengths, word2vec=word2vec)

	
	def forward(self, img_feats, ques, objs, adj_mat, ques_lens, num_obj):
		# Obtain Object Features for the Image
		rois = utils.batch_roiproposals(objs, self.device)# Change this later
		obj_feats = roi_pooling_2d(img_feats, rois, self.roi_output_size)#.detach()
		v = self.avg_layer(obj_feats).view(objs.size(0), objs.size(1), -1)

		# torch.set_printoptions(profile="full")
		# print (num_obj)
		# torch.set_printoptions(profile="default")
		# Obtain Question Embedding
		ques_output, (ques_hidden, _) = self.ques_encoder(ques, ques_lens)
		
		if self.bidirectional:
			q_emb = torch.cat([ques_hidden[-2, :, :], ques_hidden[-1, :, :]], 1)
			q_emb = self.dropout_layer(self.ques_proj(ques_emb))
		else:
			q_emb = ques_hidden[-1, :, :]


		att = self.v_att(v, q_emb, num_obj)
		v_emb = (att * v).sum(1) # [batch, v_dim]

		q_repr = self.q_net(q_emb)
		v_repr = self.v_net(v_emb)
		joint_repr = q_repr * v_repr
		logits = self.classifier(joint_repr)
		return logits