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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tr', '--train_data', type=str, default='./train_data.txt', required=False)
    parser.add_argument('-te', '--test_data', type=str, default='./test_data.txt', required=False)
    parser.add_argument('-g', '--gpu_ids', type=int, default=0, required=False, help="specify gpu ids")
    args = parser.parse_args()
    config.initialize(args)
    model = NetWork(config=config)
    model = model.to(config.device)
    print(model)
    model_agent = Agent(model,config)
    model_agent.load_model(10)
    test_loader = get_data_loader('test',config,config.batch_num,config.num_workers)
    total_correct_num = 0
    for i,data in enumerate(test_loader):
        output, train_loss, train_acc = model_agent.forward(data)
        predicts_cpu = output.detach().numpy(); labels_cpu=data['labels'].detach().numpy();
        correct_num=sum((predicts_cpu>0.5)==labels_cpu)
        total_correct_num+=correct_num
    print(total_correct_num)
    print("test:",total_correct_num/5000)
    # epoch10 model test: 0.9064