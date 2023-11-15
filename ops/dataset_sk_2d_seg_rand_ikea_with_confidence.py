#
import torch.utils.data as data
import torch.nn as nn
import os
import os.path
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from numpy.random import randint
ntu_skeleton_bone_pairs = ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16))
index_start_point = [x[0] for x in ntu_skeleton_bone_pairs]
index_end_point = [x[1] for x in ntu_skeleton_bone_pairs]

class MSSTdata(data.Dataset):
    def __init__(self, dataroot, modality='joint', test_mode=False, seg=8):
        datapath = os.path.join(dataroot, 'train_dict.pkl')
        if test_mode:
            datapath = os.path.join(dataroot, 'val_dict.pkl')
        self.modality = modality
        self.seg = seg
        self.test_mode = test_mode
        self.all_data = pickle.load(open(datapath, 'rb'))
        self.all_keys = list(self.all_data.keys())
        print('样本数量：',len(self.all_keys))

    def sample(self, couple_skeleton, length):
        avg_interval = length / self.seg
        if self.test_mode or avg_interval <= 1:
            indexs = np.multiply(list(range(self.seg)), avg_interval) + avg_interval / 2
            indexs = list(map(int, indexs))
            sampled_data = torch.tensor(np.array(couple_skeleton[:, :, indexs, :], dtype=np.float32))
            return sampled_data
        else:
            indexs = np.multiply(list(range(self.seg)), avg_interval) + randint(round(avg_interval), size=self.seg)
            indexs = list(map(int, indexs))
            sampled_data = torch.tensor(np.array(couple_skeleton[:, :, indexs, :], dtype=np.float32))
            return sampled_data

    def sample_motion(self, couple_skeleton, length):
        tem_seg = self.seg + 1
        avg_interval = length / tem_seg
        if self.test_mode or avg_interval <= 1:
            indexs = np.multiply(list(range(tem_seg)), avg_interval) + avg_interval / 2
            indexs = list(map(int, indexs))
            sampled_data = torch.tensor(np.array(couple_skeleton[:, :, indexs, :], dtype=np.float32))
            return sampled_data[:, :, 1:, :] - sampled_data[:, :, :-1, :]
        else:
            indexs = np.multiply(list(range(tem_seg)), avg_interval) + randint(round(avg_interval), size=tem_seg)
            indexs = list(map(int, indexs))
            sampled_data = torch.tensor(np.array(couple_skeleton[:, :, indexs, :], dtype=np.float32))
            return sampled_data[:, :, 1:, :] - sampled_data[:, :, :-1, :]
    def get_bone(self, sk_data):
        M, C, T, V = sk_data.shape
        bone_data = np.zeros([M, C, T, len(ntu_skeleton_bone_pairs)])
        # xy = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        # print(xy.shape) #(1, 2, 36, 16)
        bone_data[:, 0:2, :, :] = sk_data[:, 0:2, :, index_start_point] - sk_data[:, 0:2, :, index_end_point]
        conf = sk_data[:, 2, :, index_start_point] + sk_data[:, 2, :, index_end_point]
        # print(conf.shape)#(16, 1, 27) v 到前面去了
        conf = np.transpose(conf, (1, 2, 0))#
        bone_data[:, 2, :, :] = conf
        return bone_data

    def __getitem__(self, index):
        key = self.all_keys[index] #
        sk_data = self.all_data[key]##
        T, M, V, C = sk_data.shape

        sk_data = np.transpose(sk_data, (1, 3, 0, 2))  # T,M,V,C 变成 Mx3XTx17,为了适应后续的变化，
        if 'bone' in self.modality:
            sk_data = self.get_bone(sk_data)
        if 'motion' in self.modality:
            process_data = self.sample_motion(sk_data, T)
        else:
            process_data = self.sample(sk_data, T)
        label = int(key.split('_')[-1])

        return process_data, label

    def __len__(self):
        return len(self.all_keys)

if __name__ == '__main__':

    datapath = 'data'
    train_dataloader = DataLoader(MSSTdata(datapath, test_mode=False), batch_size=4, shuffle=True, num_workers=2)
    for step, (buffer, label) in enumerate(train_dataloader):
        print("label: ", label)

