import os
import torch
import numpy as np
import utils


class Config(object):
	"""docstring for Config"""
	def __init__(self):
		super(Config, self).__init__()
		self.device = None
		self.train_data_path = None
		self.test_data_path = None
		self.save_dir = './train_log'
		self.log_dir = os.path.join(self.save_dir, 'log/')
		self.model_dir = os.path.join(self.save_dir, 'model/')
		dirs = [self.save_dir, self.log_dir, self.model_dir]
		utils.ensure_dirs(dirs)
	
	def initialize(self, args):
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.train_data_path = args.train_data
		self.test_data_path = args.test_data

		# 42, user statics params, 39, query statics params, 81
		self.wide_feat_list_size = 42 + 39
		# 13, item_terms(10) + item_topcate(1) + item_leafcate(1) + time_delta(1), 195
		self.user_item_seq_feat_size = 15 * 13
		# 16, query_lens(10) + query_topcate_len(3) + query_leafcate_len(3)
		self.query_feat_size = 10 + 3 + 3
		# 170 = N * querys
		self.user_query_seq_feat_size = 170
		# 100, query item to query term's avg
		self.query_item_query_feat_size = 100
		# 100, n * 10, 10: query => item(term ids) 
		self.user_query_item_feat_size = 100
		# 150, n * 10, 10: item => query(term ids)
		self.user_item_query_feat_size = 150
		# 100, item term ids
		self.query_user_item_feat_size = 100
		# model params 
		self.lr = 0.001
		self.beta1 = 0.9
		self.user_seq_length = 15
		self.user_item_term_length = 10 
		self.user_query_term_length = 10 
		self.query_length = 10
		self.query_topcate_length = 3 
		self.query_leafcate_length = 3 
		self.embed_size_word = 64 
		self.weight_decay = 0.00001
		self.vocab_size = 280000

		# training params, 15000, 5000
		self.train_epochs = 10
		self.batch_num = 500
		self.learning_rate = 1e-3
		self.num_workers = 0
		# train one step, we make a validation
		self.val_frequency = 1
		# after 10 epoch, save
		self.save_frequency = 10
		return

config = Config()



