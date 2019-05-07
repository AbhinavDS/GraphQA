import torch
from torch.autograd import Variable
from torch import nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal
import torch.nn.functional as F
from ..modules.mac_unit import MACUnit

def linear(in_dim, out_dim, bias=True):
	lin = nn.Linear(in_dim, out_dim, bias=bias)
	xavier_uniform_(lin.weight)
	if bias:
		lin.bias.data.zero_()

	return lin

class MACNetwork(nn.Module):
	def __init__(self, args, word2vec=None):
		super().__init__()

		self.n_vocab = args.ques_vocab_sz
		self.dim = 512
		self.embed_hidden = args.ques_word_vec_dim
		self.max_steps = args.mac_max_steps
		self.self_attention = args.mac_self_attention
		self.memory_gate = args.use_memory_gate
		self.classes = args.n_ans
		self.dropout = args.drop_prob

		self.conv = nn.Sequential(nn.Conv2d(2048, self.dim, 3, padding=1),
								nn.ELU(),
								nn.Conv2d(self.dim, self.dim, 3, padding=1),
								nn.ELU())

		if word2vec is not None:
			print('Using Pre-trained Word2vec')
			assert word2vec.size(0) == self.n_vocab
			assert word2vec.size(1) == self.embed_hidden
			self.embed = nn.Embedding(self.n_vocab, self.embed_hidden)
			self.embed.weight = nn.Parameter(word2vec)
		else:
			self.embed = nn.Embedding(self.n_vocab, self.embed_hidden)
		
		self.lstm = nn.LSTM(self.embed_hidden, self.dim,
						batch_first=True, bidirectional=True)
		self.lstm_proj = nn.Linear(self.dim * 2, self.dim)

		self.mac = MACUnit(self.dim, self.max_steps,
						self.self_attention, self.memory_gate, self.dropout)


		self.classifier = nn.Sequential(linear(self.dim * 3, self.dim),
										nn.ELU(),
										linear(self.dim, self.classes))

		self.reset()

	def reset(self):
		self.embed.weight.data.uniform_(0, 1)

		kaiming_uniform_(self.conv[0].weight)
		self.conv[0].bias.data.zero_()
		kaiming_uniform_(self.conv[2].weight)
		self.conv[2].bias.data.zero_()

		kaiming_uniform_(self.classifier[0].weight)

	def forward(self, image, question, objs, adj_mat, question_len, num_obj, obj_wrds):
		b_size = question.size(0)

		img = self.conv(image)
		img = img.view(b_size, self.dim, -1)

		embed = self.embed(question)
		embed = nn.utils.rnn.pack_padded_sequence(embed, question_len,
												batch_first=True)
		lstm_out, (h, _) = self.lstm(embed)
		lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out,
													batch_first=True)
		lstm_out = self.lstm_proj(lstm_out)
		h = h.permute(1, 0, 2).contiguous().view(b_size, -1)

		memory = self.mac(lstm_out, h, img)

		out = torch.cat([memory, h], 1)
		out = self.classifier(out)

		return out
