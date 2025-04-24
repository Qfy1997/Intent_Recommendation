from utils import TrainClock
import os
import torch
import torch.optim as optim
import torch.nn as nn


class Agent(object):
    def __init__(self, model, config):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.model = model
        self.clock = TrainClock()
        self.device = config.device
        # set loss func
        self.total_loss = nn.BCELoss(reduction='mean')

        # set optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optim, 0.99)
        # for params in self.model.parameters():
        #     print(params.shape)

        # train & val log data
        # self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
        # self.test_tb = SummaryWriter(os.path.join(self.log_dir, 'test.events'))
        self.gpu_support = True if torch.cuda.is_available() else False
        # self.word_embed = self.model._word_embed

    def save_model(self, name=None):
        if name is None:
            save_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(self.clock.epoch))
        else:
            save_path = os.path.join(self.model_dir, 'model_epoch{}.pth'.format(name))

        torch.save(self.model.cpu().state_dict(), save_path)
        self.model.to(self.device)

    def load_model(self, epoch):
        load_path = os.path.join(self.model_dir, "model_epoch{}.pth".format(epoch))
        state_dict = torch.load(load_path)
        self.model.load_state_dict(state_dict)

    def update_model(self, loss):
        # print((self.word_embed == self.model._word_embed).all())
        # self.word_embed = self.model._word_embed
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_learning_rate(self):
        self.scheduler.step(self.clock.epoch)

    def forward(self, x):
        inputs = x['datas'].T; labels = x['labels'];
        if self.gpu_support:
            inputs = inputs.cuda()
            labels = labels.cuda()

        predicts = self.model(inputs)
        # loss
        loss_cross = self.total_loss(predicts, labels.float())
        loss_total = loss_cross + self.model.regular_loss
        # auc value
        predicts_cpu = predicts.cpu().detach().numpy(); labels_cpu = labels.cpu().detach().numpy();
        correct_num = sum((predicts_cpu > 0.5) == labels_cpu)
        acc = correct_num / len(predicts_cpu)
        return predicts, loss_total.cpu(), acc

    def train_model(self, x):
        self.model.train()
        output, loss, acc = self.forward(x)
        self.update_model(loss)
        return output, loss, acc

    def eval_model(self, x):
        self.model.eval()

        with torch.no_grad():
            output, loss, acc = self.forward(x)

        return output, loss, acc
