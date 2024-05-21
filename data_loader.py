from cProfile import label
from decimal import localcontext
from functools import total_ordering
from platform import node
from random import gauss
from tkinter import N
from wsgiref.validate import InputWrapper
import torch
import torch.utils.data as data
import os
import pickle
import json
import numpy as np
from torch.utils.data import DataLoader
import math
import random
import torch.nn as nn
from layers import *

def load_json(path):
    f = open(path, )
    dataset = json.load(f)
    return dataset

def read_pkl(path):
    with open(path, 'rb') as f:
        array = pickle.load(f)
    return array

class pcle_dataset(data.Dataset):
    def __init__(self,
                mode,
                num_channels,
                dis_range,
                ord_regre,
                cls_regre,
                discrete_value,
                use_interp,
                MixUp=False,
		        norm=True):
        super(pcle_dataset, self).__init__()
        self.root_path = os.path.dirname(os.getcwd())
        self.mode = mode
        if mode[:5] == "train":
            self.train = True
        else:
            self.train = False
        self.num_channels = num_channels
        self.dis_range = dis_range
        self.ord_regre = ord_regre
        self.cls_regre = cls_regre
        self.use_interp = use_interp
        self.discrete_value = discrete_value
        self.norm = norm
        self.mixup = MixUp
        json_path = ('./{}_dataset.json').format(mode)
        self.data_list = load_json(json_path)
        self.random_range = 5
        self.height = 288
        self.width = 384
        self.e_mask = Grad_Energy()


    def normlization(self, data, I_max, I_min):
        if I_max == None:
            I_max = torch.max(data)
            I_min = torch.min(data)

        data_norm = (data - I_min) / (I_max - I_min)

        return data_norm

    
    def random_sign(self):
        return 1 if random.random() < 0.5 else -1

        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        inputs = {}
        data_dict = self.data_list[str(idx)]
        frame_split = data_dict['frame'].split('/')
        frame_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], frame_split[-1])
        video_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], 'video.npy')
        BM_path = os.path.join(self.root_path, frame_split[-3], frame_split[-2], 'BM.npy')
        video = torch.from_numpy(np.load(video_path))
        BM_score = torch.from_numpy(np.load(BM_path))
        num_frames, height, width = video.shape
        temp = torch.zeros(161, height, width)
        temp_BM = torch.zeros(161)
        temp[:num_frames, :, :] = video
        temp_BM[:num_frames] = BM_score
        inputs['optimal'] = torch.tensor(data_dict['optimal_index'], dtype=torch.long)
        inputs['video'] = temp
        inputs['BM'] = temp_BM
        inputs['index'] = torch.tensor(data_dict['index'], dtype=torch.long)
        inputs['V_max'] = torch.max(video)
        inputs['V_min'] = -400
        inputs['num_frames'] = num_frames
        inputs['origin'] = torch.from_numpy(read_pkl(frame_path)).type(torch.float32).unsqueeze(0)
        
        mixup = (self.train and self.mixup)
        

        if mixup:
            frame_current = torch.from_numpy(read_pkl(frame_path)).type(torch.float32).unsqueeze(0)
            random_ratio = torch.tensor(np.random.beta(0.2, 0.2) * self.random_sign())
            sign = torch.sign(random_ratio)
            ratio = torch.abs(random_ratio)
            distance = data_dict['distance']
            if sign > 0:
                if inputs["index"] != 0:
                    index = inputs["index"] - 1
                    inputs['distance'] = (distance + random_ratio * self.random_range).clone().squeeze(0)
                else:
                    index = inputs["index"] + 1
                    inputs['distance'] = (distance - random_ratio * self.random_range).clone().squeeze(0)
                
            elif sign < 0:
                if inputs["index"] != (num_frames - 1):
                    index = inputs["index"] + 1
                    inputs['distance'] = (distance + random_ratio * self.random_range).clone().squeeze(0)
                else:
                    index = inputs["index"] - 1
                    inputs['distance'] = (distance - random_ratio * self.random_range).clone().squeeze(0)
            frame_mix = temp[index, :, :].unsqueeze(0)
            im_pcle = (1 - ratio) * frame_current + ratio * frame_mix
        else:
            im_pcle = torch.from_numpy(read_pkl(frame_path)).type(torch.float32).unsqueeze(0)
            distance = data_dict['distance']
            inputs['distance'] = torch.tensor(distance, dtype=torch.float32)

        if self.ord_regre:
            n_disc = self.discrete_value.shape[0]
            idx_0 = n_disc // 2 
            if inputs['distance'] > 0:
                labels_t = ((self.discrete_value < inputs['distance']) * (self.discrete_value >= 0)) * 1
                n_ord = labels_t.sum()
                ratio = (inputs['distance'] - self.discrete_value[idx_0 + n_ord - 1]) / (self.discrete_value[idx_0 + n_ord] - self.discrete_value[idx_0 + n_ord - 1])
            elif inputs['distance'] == 0:
                labels_t = (self.discrete_value < inputs['distance']) * 1
                ratio = 0
            else:
                labels_t = ((self.discrete_value > inputs['distance']) * (self.discrete_value <= 0)) * 1
                n_ord = labels_t.sum()
                ratio = (inputs['distance'] - self.discrete_value[idx_0 - n_ord + 1]) / (self.discrete_value[idx_0 - n_ord] - self.discrete_value[idx_0 - n_ord + 1])
            labels_pos = labels_t[1:-1].float()
            labels_neg = 1 - labels_pos
            inputs['ord_label'] = torch.cat((labels_pos, labels_neg), dim=0)
            inputs['ratio'] = float(ratio)
        elif self.cls_regre:
            cls_target = torch.zeros(5, dtype=torch.float32)
            if inputs['distance'] < -35:
                cls_target[0] = (inputs['distance'] + 35) / (-400 + 35)
                cls_target[1] = (-400 - inputs['distance']) / (-400 + 35)
                # ratio = (inputs['distance'] + 35) / (-400 + 35)
                inputs['cls_target'] = cls_target
                # inputs['ratio'] = ratio
            elif -35 <= inputs['distance'] < 0:
                cls_target[1] = (inputs['distance']) / (-35)
                cls_target[2] = (-35 - inputs['distance']) / (-35)
                # ratio = (inputs['distance']) / (-35)
                inputs['cls_target'] = cls_target
                # inputs['ratio'] = ratio
            elif 0 <= inputs['distance'] <= 35:
                cls_target[2] = (35 - inputs['distance']) / (35)
                cls_target[3] = (inputs['distance']) / (35)
                # ratio = (inputs['distance']) / 35
                inputs['cls_target'] = cls_target
                # inputs['ratio'] = ratio
            else:
                cls_target[3] = (400 - inputs['distance']) / (400 - 35)
                cls_target[4] = (inputs['distance'] - 35) / (400 - 35)
                # ratio = (inputs['distance'] - 35) / (400 - 35)
                inputs['cls_target'] = cls_target
                # inputs['ratio'] = ratio

        if distance > 0:
            inputs['D_gt'] = torch.tensor(1, dtype=torch.float)
        elif distance < 0:
            inputs['D_gt'] = torch.tensor(-1, dtype=torch.float)
        else:
            inputs['D_gt'] = torch.tensor(0, dtype=torch.float)
        if self.norm == True:
            ipt_ch1 = self.normlization(im_pcle, None, None)
            ipt_ch2 = self.normlization(im_pcle, I_max=8191, I_min=-400)
            inputs['frame'] = torch.cat((ipt_ch1, ipt_ch2), dim=0)

        else:
            inputs['frame'] = im_pcle
        return inputs