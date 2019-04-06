"""
Module that implements the gated tanh function that is repeatedly used in the attention and question answering module
"""

class GatedTanh(nn.Module):

	def __init__(self, n_inp, n_out)
		super(GatedTanh, self).__init__()

		t_layer = nn.Linear(n_out, n_inp)
		g_layer = nn.Linear(n_out, n_inp)

	def forward(self, inp):

		y = nn.Tanh(t_layer(inp))
		g = nn.Sigmoid(g_layer(inp))

		return torch.mul(y, g)


