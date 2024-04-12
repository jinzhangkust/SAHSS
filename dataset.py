"""
Author: Dr. Jin Zhang 
E-mail: j.zhang@kust.edu.cn
Created on 2023.08.02
"""

import torch
import torchvision
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
import random
import os


class TailingSensorSet(Dataset):
    def __init__(self, train_mode, clip_mode):
        self.root = '/home/ps/datasets/FrothData/Data4FrothGrade'
        csv_file = '/home/ps/datasets/FrothData/ImgReagent4Tail.csv'
        hf_file = '/home/ps/datasets/FrothData/StaticHFSeqSensing.csv'
        self.clip_mode = clip_mode
        self.frames = 7

        self.df = pd.read_csv(csv_file)
        clip = self.df.iloc[:, 0].values
        reagent = self.df.iloc[:, 1:4].values
        tailing = self.df.iloc[:, 4:7].values

        self.hf_df = pd.read_csv(hf_file)
        self.hf_data = self.hf_df.iloc[:, 6:17].values
        hf_clip = self.hf_df.iloc[:, 17].values
        self.hf_clip = []
        for idx in range(len(hf_clip)):
            self.hf_clip.append(hf_clip[idx].split('/')[5] + '/' + hf_clip[idx].split('/')[6])

        mean_tailing = [tailing[:, 0].mean(), tailing[:, 1].mean(), tailing[:,2 ].mean()]
        std_tailing = [tailing[:, 0].std(), tailing[:, 1].std(), tailing[:, 2].std()]
        print(f"mean_tailing: {mean_tailing} \n std_tailing: {std_tailing}")
        tailing = (tailing - mean_tailing) / std_tailing
        mean_reagent = [reagent[:, 0].mean(), reagent[:, 1].mean(), reagent[:, 2].mean()]
        std_reagent = [reagent[:, 0].std(), reagent[:, 1].std(), reagent[:, 2].std()]
        reagent = (reagent - mean_reagent) / std_reagent
        
        index = np.random.RandomState(seed=42).permutation(len(self.df))
        self.tailing = tailing[index, :]
        self.reagent = reagent[index, :]
        self.clip = clip[index]

        transform = None
        if transform is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5561, 0.5706, 0.5491],
                                                          std=[0.1833, 0.1916, 0.2061])
            if train_mode == "train":
                self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(256),
                torchvision.transforms.ToTensor(),
                normalize])
            else:
                self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(256),
                torchvision.transforms.ToTensor(),
                normalize])
                
        transform_clip = None
        if transform_clip is None:
            normalize = torchvision.transforms.Normalize(mean=[0.5429, 0.5580, 0.5357],
                                                          std=[0.1841, 0.1923, 0.2079])
            self.transform_clip = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize])
        
    def denormalize4img(self, x_hat):
        mean = [0.5561, 0.5706, 0.5491]
        std = [0.1833, 0.1916, 0.2061]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x = x_hat * std + mean
        return x

    def __len__(self):
        return len(self.clip)

    def __getitem__(self, idx):
        truth = torch.tensor(self.tailing[idx, :], dtype=torch.float32)
        reagent = torch.tensor(self.reagent[idx, :], dtype=torch.float32)
        clip = self.clip[idx]

        index = self.hf_clip.index(clip)
        hf_data = torch.tensor(self.hf_data[index, :], dtype=torch.float32)
        
        time_stamp = clip[:14] #print('time_stamp_1: {}'.format(time_stamp))

        if self.clip_mode == "seq":
            img_list = torch.FloatTensor(3, self.frames, 400, 400)  # [channels, frames, height, width]
            for i in range(1, self.frames + 1):
                file_name = "{}_{}.jpg".format(time_stamp, i)
                full_img_path = os.path.join(self.root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i - 1, :, :] = self.transform_clip(img).float()
            top = np.random.randint(0, 144)
            left = np.random.randint(0, 144)
            images = img_list[:, :, top: top + 256, left: left + 256]
        elif self.clip_mode == "test":
            img_list = torch.FloatTensor(3, self.frames, 400, 400)  # [channels, frames, height, width]
            for i in range(1, self.frames + 1):
                file_name = "{}_{}.jpg".format(time_stamp, i)
                full_img_path = os.path.join(self.root, clip, file_name)
                img = Image.open(full_img_path).convert("RGB")
                img_list[:, i - 1, :, :] = self.transform_clip(img).float()
            top = np.random.randint(0, 144)
            left = np.random.randint(0, 144)
            seq = img_list[:, :, 72: 72 + 256, 72: 72 + 256]

            file_name = "{}_{}.jpg".format(time_stamp, random.randint(1, 10))
            full_img_path = os.path.join(self.root, clip, file_name)
            single = Image.open(full_img_path).convert("RGB")
            single = self.transforms(single)
            images = (single, seq)
        else:
            file_name = "{}_{}.jpg".format(time_stamp, 6) #random.randint(1, 10))   #fixed as 6 for testing
            full_img_path = os.path.join(self.root, clip, file_name)
            images = Image.open(full_img_path).convert("RGB")
            images = self.transforms(images)
        
        return reagent, images, hf_data, truth
