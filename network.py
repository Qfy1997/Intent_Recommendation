import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable 
import math
import torch.nn.init as init

# def gelu2(x):
#     return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# def gelu3(x): # slower than gelu2
#     return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__(in_features, out_features, bias)
        # 使用自定义的权重初始化方法
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        self.bias.requires_grad = False
        for i in range(len(self.bias)):
            self.bias[i] = torch.tensor(0)
        print(self.bias)
        self.bias.requires_grad = True
        
        self.weight.requires_grad=False
        for i in range(len(self.weight)):
            self.weight[i] = torch.tensor(0)
        self.weight.requires_grad=True
        print(self.weight)
        # print(type(self.weight))


class NetWork(nn.Module):
	"""docstring for NetWork"""
	def __init__(self, config):
		super(NetWork, self).__init__()
		self.config = config
		self._generate_model_layer()

		# drop out prob
		self.keep_prob = 1

	def set_mode(self, mode='train'):
		self.keep_prob = 1 # if mode == 'train' else 1.

	def _generate_model_layer(self):
		# self._word_embed = nn.Parameter(Variable(torch.Tensor(self.config.vocab_size, self.config.embed_size_word), requires_grad=True))
		self._word_embed = nn.Parameter(torch.zeros(self.config.vocab_size,self.config.embed_size_word),requires_grad=True)
		# self.register_parameter('word_embed', nn.Parameter(self._word_embed))
		# lstms
		# h,c for lstm weights, w b for mlp weights, bias
		self.user_word_lstm, self.h_1, self.c_1, self.w_l1, self.b_l1, self.loss1 = self._rnn_lstm(5, self.config.batch_num, 64, 64)
		self.user_item_query_lstm, self.h_2, self.c_2, self.w_l2, self.b_l2, self.loss2 = self._rnn_lstm(5, self.config.batch_num, 64, 64)
		self.user_query_item_lstm, self.h_3, self.c_3, self.w_l3, self.b_l3, self.loss3 = self._rnn_lstm(5, self.config.batch_num, 64, 64)
		self.user_query_seq_lstm, self.h_4, self.c_4, self.w_l4, self.b_l4, self.loss4 = self._rnn_lstm(5, self.config.batch_num, 64, 64)
		# cnn
		self.query_item_query_cnn, self.conv_w1, self.loss5 = self._cnn(64)
		# self.query_item_query_linear = nn.Linear(12 * 2 * 1, 64)
		self.query_item_query_linear = CustomLinear(12 * 2 * 1, 64)
		self.query_item_query_relu = nn.ReLU()

		self.query_user_item_cnn, self.conv_w2, self.loss6 = self._cnn(64)
		# self.query_user_item_linear = nn.Linear(12 * 2 * 1, 64)
		self.query_user_item_linear = CustomLinear(12 * 2 * 1, 64)
		self.query_user_item_relu = nn.ReLU()

		# weights & bias
		# wide feats, mlp layers for wide feats, 81 for static feats length
		self.wide_feat_w, self.loss7 = self.get_weights_variables([64, 81], self.config.weight_decay)
		self.wide_feat_b = self.get_bias_variables(64)
		# concat query weights, 64 * 7 for concat feats len
		self.concat_query_w, self.loss8 = self.get_weights_variables([64, 64 * 7], self.config.weight_decay)
		self.concat_query_b = self.get_bias_variables(64)
		# concat_query_user_wide,  concat query and user wide infos, then len: 128
		self.concat_query_user_wide_w, self.loss9 = self.get_weights_variables([64, 128], self.config.weight_decay)
		self.concat_query_user_wide_b = self.get_bias_variables(64)
		# deep_wide_feat, the last layer mlp => predict val
		self.deep_wide_feat_w, self.loss10 = self.get_weights_variables([1, 64], self.config.weight_decay)
		self.deep_wide_feat_b = self.get_bias_variables(1)

		return

	@property
	def regular_loss(self):
		return self.get_regular_loss(self.h_1) + self.get_regular_loss(self.c_1) + self.get_regular_loss(self.w_l1) \
		     + self.get_regular_loss(self.h_2) + self.get_regular_loss(self.c_2) + self.get_regular_loss(self.w_l2) \
			 + self.get_regular_loss(self.h_3) + self.get_regular_loss(self.c_3) + self.get_regular_loss(self.w_l3) \
			 + self.get_regular_loss(self.h_4) + self.get_regular_loss(self.c_4) + self.get_regular_loss(self.w_l4) \
			 + self.get_regular_loss(self.conv_w1) + self.get_regular_loss(self.conv_w2) \
		     + self.get_regular_loss(self.wide_feat_w) + self.get_regular_loss(self.concat_query_w) \
		     + self.get_regular_loss(self.concat_query_user_wide_w) + self.get_regular_loss(self.deep_wide_feat_w)

	def get_regular_loss(self, params):
		return torch.sum(torch.pow(params, 2)) * self.config.weight_decay

	def get_weights_variables(self, shape, weight_decay, trainable=True):
		params = nn.Parameter(torch.zeros(*shape),requires_grad=True)
		if weight_decay == 0:
			regular_loss = 0.
		else:
			# xavier initializer, need params demension >= 2
			# nn.init.xavier_normal_(params, gain=1.)
			nn.init.constant_(params, 0.)
			# l2 regular loss for weights
			regular_loss = torch.sum(torch.pow(params, 2)) * weight_decay
		return params, regular_loss

	def get_bias_variables(self, size, trainable=True):
		params = nn.Parameter(torch.zeros(size,1),requires_grad=True)
		# constant initialize of
		nn.init.constant_(params, 0.)
		return params

	def _rnn_lstm(self, layers_num, batches_num, features_num, hidden_size):
		"""
		:param input: layers_num, batch_num, features_num
		:param hidden_size:
		:return: lstm out, loss of weights(l2 regular)
		input like: T * BatchSize * 64
		"""
		# LstmCell: features * hidden_size * layers_num
		lstm = nn.LSTM(features_num, hidden_size, layers_num, bias=True)
		# weights: num_layers * batch * hidden size
		h_0, regular_loss_h0 = self.get_weights_variables([layers_num, batches_num, hidden_size], self.config.weight_decay)
		c_0, regular_loss_c0 = self.get_weights_variables([layers_num, batches_num, hidden_size], self.config.weight_decay)

		w_l, regular_loss_wl = self.get_weights_variables([hidden_size, hidden_size], self.config.weight_decay)
		b_l = self.get_bias_variables(hidden_size)
		# hidden_layer = torch.tanh(torch.matmul(w_l, lstm_cell.T) + b_l)
		loss_total = regular_loss_h0 + regular_loss_c0 + regular_loss_wl
		# output: 64 * batch_num
		return lstm, h_0, c_0, w_l, b_l, loss_total

	def _cnn(self, features_num):
		"""
		:param input:  nums * batch_num * features_num
		:param hidden_size:
		:return:
		"""
		# x: batch * channels * h * w
		# x_input = input.view(-1, 1, nums, features_num)
		conv_w, regular_loss_w = self.get_weights_variables([12, 1, 2, features_num], self.config.weight_decay)
		conv_b, _ = self.get_weights_variables([12], 0)
		# default stride 1, padding valid
		conv = nn.Conv2d(1, 12, [2, features_num], stride=1)
		conv.weight.requires_grad = False
		for i in range(len(conv.weight)):
			conv.weight[i] = torch.tensor(0)
		conv.weight.requires_grad = True
		print("conv weight:")
		print(conv.weight)
		print(conv.weight.shape)

		conv.bias.requires_grad = False
		for i in range(len(conv.bias)):
			conv.bias[i]=torch.tensor(0)
		conv.bias.requires_grad = True
		print("conv bias:")
		print(conv.bias)
		print(conv.bias.shape)
		# conv.weight = nn.Parameter(conv_w)
		# conv.bias = nn.Parameter(conv_b)
		# batch * 12 * 4 * 1
		# pool = F.max_pool2d(conv_active, [2, 1], stride=2)
		# pool_flat = pool.view(-1, 12 * 2 * 1)
		# dense = nn.LeakyReLU()(nn.Linear(12 * 2 * 1, hidden_size)(pool_flat))
		return conv, conv_w, regular_loss_w

	def forward_model(self, x):
		# x:[m, batch_size];
		wide_feat = x[:81, :]; user_item_seq = x[81:276, :]; query_feat = x[276:292, :];
		user_query_seq = x[292:462, :]; query_item_query = x[462:562, :]; user_query_item = x[562:662, :];
		user_item_query = x[662:812, :]; query_user_item = x[812:, :];

		# query embedding, query terms 10 * batch
		query_terms, query_topcate, query_leafcate = torch.split(query_feat, [self.config.query_length, \
											  self.config.query_topcate_length, \
											  self.config.query_leafcate_length], 0)
		# look up in word embedding's dict [280000, 64] => 10 * batch * 64
		inputs_query_raw = self._word_embed[query_terms.reshape(-1).long()].view(self.config.query_length, -1, self.config.embed_size_word)
		input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
		# get query word to vec sum 64 * batchsize
		self.query_w2v_sum = torch.mean(inputs_query_raw[:int(input_num.item())], 0).T

		# user word embedding
		raw_word_embedding_list = torch.split(user_item_seq[-13 * 5:, :], [13] * 5, 0)
		step_embedding_list = []
		for raw_word_embed in raw_word_embedding_list:
			item_terms, item_topcate, item_leafcate, time_delta = torch.split(raw_word_embed, [self.config.user_item_term_length, 1, 1, 1], 0)
			step_embedding = self._word_embed[item_terms.reshape(-1).long()].view(self.config.user_item_term_length, -1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			# step_avg_embedding: batchsize * 64
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			# append a new axis in the first
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		# step_embedding vec: 5 * batchsize * 64
		step_embedding_vec = torch.cat(step_embedding_list, 0)
		lstm_cells, lstm_hiddens = self.user_word_lstm(step_embedding_vec, (self.h_1, self.c_1))
		self.user_item_term_lstm_output = torch.tanh(torch.matmul(self.w_l1, lstm_cells[-1].T) + self.b_l1)

		# user_item_query_embedding
		raw_user_item_query_embedding_list = torch.split(user_item_query[:10 * 5, :], [10] * 5, 0)
		step_embedding_list = []
		for raw_user_item_embed in raw_user_item_query_embedding_list:
			item_terms = raw_user_item_embed[:5]
			step_embedding = self._word_embed[item_terms.reshape(-1).long()].view(5,
																				   -1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		step_embedding_vec = torch.cat(step_embedding_list, 0)
		lstm_cells, lstm_hiddens = self.user_item_query_lstm(step_embedding_vec, (self.h_2, self.c_2))
		self.user_item_query_term_lstm_output = torch.tanh(torch.matmul(self.w_l2, lstm_cells[-1].T) + self.b_l2)

		# user_query_item embedding
		raw_user_query_item_embedding_list = torch.split(user_query_item[-10 * 5:, :], [10] * 5, 0)
		step_embedding_list = []
		for raw_user_item_embed in raw_user_query_item_embedding_list:
			item_terms = raw_user_item_embed[:5]
			step_embedding = self._word_embed[item_terms.reshape(-1).long()].view(5,
																				   -1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		step_embedding_vec = torch.cat(step_embedding_list, 0)
		lstm_cells, lstm_hiddens = self.user_query_item_lstm(step_embedding_vec, (self.h_3, self.c_3))
		self.user_query_item_term_lstm_output = torch.tanh(torch.matmul(self.w_l3, lstm_cells[-1].T) + self.b_l3)

		#  user_query_seq embed
		raw_user_query_seq_embedding_list = torch.split(user_query_seq[-17 * 5:, :], [17] * 5, 0)
		step_embedding_list = []
		for raw_user_item_embed in raw_user_query_seq_embedding_list[::-1]:
			query_terms, query_topcate, query_leafcate, time_delta = torch.split(raw_user_item_embed, [self.config.user_query_term_length, 3, 3, 1], 0)
			step_embedding = self._word_embed[query_terms.reshape(-1).long()].view(self.config.user_query_term_length,
																				   -1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		step_embedding_vec = torch.cat(step_embedding_list, 0)
		lstm_cells, lstm_hiddens = self.user_query_seq_lstm(step_embedding_vec, (self.h_4, self.c_4))
		self.user_query_term_lstm_output = torch.tanh(torch.matmul(self.w_l4, lstm_cells[-1].T) + self.b_l4)

		# query_item_query embed
		raw_query_item_query_embedding_list = torch.split(query_item_query[-10 * 5:, :], [10] * 5, 0)
		step_embedding_list = []
		for raw_user_item_embed in raw_query_item_query_embedding_list:
			query_terms = raw_user_item_embed[:5]
			step_embedding = self._word_embed[query_terms.reshape(-1).long()].view(5,
																					-1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		step_embedding_vec = torch.cat(step_embedding_list, 0)

		convd = self.query_item_query_cnn(step_embedding_vec.view(-1, 1, 5, self.config.embed_size_word))
		convd_active = F.relu(convd)
		pooled = F.max_pool2d(convd_active, [2, 1], stride=2)
		pool_flat = pooled.view(-1, 12 * 2 * 1)
		self.query_item_query_cnn_output = self.query_item_query_relu(self.query_item_query_linear(pool_flat)).T
		# self.query_item_query_cnn_output = gelu3(self.query_item_query_linear(pool_flat)).T

		# query_user_item embedd
		raw_query_user_item_embedding_list = torch.split(query_user_item[-10 * 5:, :], [10] * 5, 0)
		step_embedding_list = []
		for raw_user_item_embed in raw_query_user_item_embedding_list:
			item_terms = raw_user_item_embed[:5]
			step_embedding = self._word_embed[item_terms.reshape(-1).long()].view(5,
																					-1, self.config.embed_size_word)
			input_num = torch.sum(torch.sign(query_terms)) if torch.sum(torch.sign(query_terms)) > 1 else torch.tensor(1)
			step_avg_embedding = torch.mean(step_embedding[:int(input_num.item())], 0)
			step_embedding_list.append(step_avg_embedding.unsqueeze(0))
		step_embedding_vec = torch.cat(step_embedding_list, 0)

		convd = self.query_user_item_cnn(step_embedding_vec.view(-1, 1, 5, self.config.embed_size_word))
		convd_active = F.relu(convd)
		pooled = F.max_pool2d(convd_active, [2, 1], stride=2)
		pool_flat = pooled.view(-1, 12 * 2 * 1)
		self.query_user_item_cnn_output = self.query_user_item_relu(self.query_user_item_linear(pool_flat)).T
		# self.query_user_item_cnn_output = gelu3(self.query_user_item_linear(pool_flat)).T

		# wide connected, connect static params
		# wide feat hidden output: 64 * batch_size
		self.wide_hidden_layer1 = torch.tanh(F.dropout(torch.matmul(self.wide_feat_w, wide_feat.float()) + self.wide_feat_b, self.keep_prob))
		concat_seq = [self.user_item_term_lstm_output, self.user_query_term_lstm_output, self.query_w2v_sum,
					  self.user_item_query_term_lstm_output, self.user_query_item_term_lstm_output, self.query_item_query_cnn_output,
					  self.query_user_item_cnn_output]
		qu_term_concat = F.dropout(torch.cat(concat_seq, 0), self.keep_prob)
		# concat user query feats: 64 * batch_size
		self.qu_term_hidden_layer1 = torch.tanh(F.dropout(torch.matmul(self.concat_query_w, qu_term_concat) + self.concat_query_b, self.keep_prob))
		# concat feats and static datas: 128 * batch_size
		deep_wide_concat = torch.cat([self.qu_term_hidden_layer1, self.wide_hidden_layer1], 0)
		# mlp for wide concat data : 64 * batch_size
		self.dw_hidden_layer0 = torch.tanh(F.dropout(torch.matmul(self.concat_query_user_wide_w, deep_wide_concat) + self.concat_query_user_wide_b, self.keep_prob))
		self.dw_hidden_layer1 = torch.sigmoid(torch.matmul(self.deep_wide_feat_w, self.dw_hidden_layer0) + self.deep_wide_feat_b)
		self.predict_labels = self.dw_hidden_layer1.squeeze()
		# print(self.predict_labels.shape)
		return self.predict_labels

	def forward(self, x):
		return self.forward_model(x)
	

# model = NetWork(None)
# A, B = model._cnn(torch.empty(5, 512, 64), 64)
# print(A.shape, B)
# model(np.zeros([1000, 500]))