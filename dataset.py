import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import numpy as np
import codecs

class EmbeddingDataSet(Dataset):
    def __init__(self, phase, config):
        assert phase in ['train', 'test']
        if phase == 'train':
            self.data_path = config.train_data_path
        else:
            self.data_path = config.test_data_path
        self.load_data(self.data_path)

    def load_data(self, path):
        self.features_list = []
        self.label_list = []
        with codecs.open(path, mode='r', encoding='utf8') as f:
            for line in f.readlines():
                line_items = line.split()
                feature, label = line_items[0], line_items[1]
                self.features_list.append(feature)
                self.label_list.append(float(label))
        self.data_len = len(self.features_list)
        # print(self.data_len)

    def __getitem__(self, item):
        index = item
        feature_list = []
        feature = self.features_list[index]
        features_split = feature.split(',')
        for item in features_split[:81]:
            feature_list.append(float(item))
        for item in features_split[81:]:
            feature_list.append(int(item))
        return {'datas': torch.from_numpy(np.array(feature_list)), 'labels': self.label_list[index]}

    def __len__(self):
        return self.data_len


def get_data_loader(type, config, batch_size=512, num_workers=4):
    dataset = EmbeddingDataSet(type, config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, worker_init_fn=lambda _: np.random.seed())
    return dataloader