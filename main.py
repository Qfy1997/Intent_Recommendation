#encoding=utf8
import torch
import os
import argparse
import numpy as np
import utils
from config import config
from network import NetWork
from dataset import get_data_loader
from agent import Agent
from collections import OrderedDict


torch.backends.cudnn.benchmark = True

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train_data', type=str, default='./train_data.txt', required=False)
	parser.add_argument('-te', '--test_data', type=str, default='./test_data.txt', required=False)
	parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
	args = parser.parse_args()

	config.initialize(args)
	train_loader = get_data_loader('train', config, config.batch_num, config.num_workers)
	test_loader = get_data_loader('test', config, config.batch_num, config.num_workers)
	test_loader = utils.cycle(test_loader)

	model = NetWork(config=config)
	print(model)
	model = model.to(config.device)

	model_agent = Agent(model, config)
	clock = model_agent.clock

	for e in range(config.train_epochs):
		for i, data in enumerate(train_loader):
			# train step, data: batch * features
			output, train_loss, train_acc = model_agent.train_model(data)
			print("epoch :",e,'loss', train_loss)

			if clock.step % config.val_frequency == 0:
				val_data = next(test_loader)
				model_agent.model.set_mode(mode='test')
				output, test_loss, test_acc = model_agent.eval_model(val_data)
				model_agent.model.set_mode(mode='train')
			
				print("train loss:",train_loss.item()," test loss:",test_loss.item()," train_acc:",train_acc," test_acc:",test_acc)
			clock.tick()

		model_agent.update_learning_rate()
		if clock.epoch % config.save_frequency == 0:
			model_agent.save_model()

		model_agent.save_model('latest.pth.tar')

		clock.tock()


if __name__ == '__main__':
	main()