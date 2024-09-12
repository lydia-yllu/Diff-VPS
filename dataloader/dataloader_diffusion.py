from __future__ import division

import os
import numpy as np
import cv2
import torch
import pickle
import random
import imageio
from PIL import Image
import skimage.morphology as sm
from torch.utils.data import Dataset
from dataloader.augmentation_diffusion import *
import pdb

class TrainDataset(Dataset):

	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess',
				 transform=None, timeclips=4):

		samples_list_file = os.path.join(file_path_root, 'timeclips_{}/train.txt'.format(timeclips))
		f = open(samples_list_file, "r")
		self.samples_list = f.readlines()
		self.transform = transform
		self.timeclips = timeclips
  
	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):
		sample_line = self.samples_list[idx].strip()
		seq_path_line, frame_path, mask_path, clas, bbox = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		clas = int(clas)
		bbox = bbox.split(" ")
		for i in range(len(bbox)):
			bbox[i] = int(bbox[i])
		seq = []
		for i in range(self.timeclips):
			img = Image.open(seq_path[i]).convert('RGB')
			seq.append(img)
		mask = Image.open(mask_path).convert('L')  # mask
		frame = Image.open(frame_path).convert('RGB')  # H W C
		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'reconstruct_gt': frame, 'clas': clas, 'bbox' : bbox}
		if self.transform is not None:
			sample = self.transform(sample)
		return sample


class ValidDataset(Dataset):

	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess', samples_dataset='TestEasyDataset_Seen',
            	transform=None, timeclips=4):

		samples_list_file = os.path.join(file_path_root, 'timeclips_{}/{}.txt'.format(timeclips, samples_dataset))
		f = open(samples_list_file, "r")
		self.samples_list = f.readlines()
		self.transform = transform
		self.timeclips = timeclips
  
	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):

		sample_line = self.samples_list[idx].strip()
		seq_path_line, frame_path, mask_path = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		
		seq = []
		for i in range(self.timeclips):
			img = Image.open(seq_path[i]).convert('RGB')
			seq.append(img)

		mask = Image.open(mask_path).convert('L')  # mask
		frame = Image.open(frame_path).convert('RGB')  # H W C

		sample = {'seq': seq, 'frame': frame, 'mask': mask}
		if self.transform is not None:
			sample = self.transform(sample)
		return sample


class TestDataset(Dataset):

	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess', dataset='TestEasyDataset_Seen',
            	transform=None, timeclips=4):

		samples_list_file = os.path.join(file_path_root, 'timeclips_4/{}.txt'.format(dataset))
		f = open(samples_list_file, "r")
		self.samples_list = f.readlines()
		self.transform = transform
		self.timeclips = timeclips
  
	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):
		
		sample_line = self.samples_list[idx].strip()
		seq_path_line, frame_path, mask_path = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		
		seq = []

		for i in range(self.timeclips):
			img = Image.open(seq_path[i]).convert('RGB')
			seq.append(img)
		
		mask = Image.open(mask_path).convert('L')  # mask
		frame = Image.open(frame_path).convert('RGB')  # H W C

		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'frame_path': frame_path}
		if self.transform is not None:
			sample = self.transform(sample)

		return sample


# class TestDatasetEU(Dataset):

# 	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess', 
#             	transform=None, timeclips=4):

# 		samples_list_file = os.path.join(file_path_root, 'timeclips_{}/TestEasyDataset_Unseen.txt'.format(timeclips))
# 		f = open(samples_list_file, "r")
# 		self.samples_list = f.readlines()
# 		self.transform = transform
# 		self.timeclips = timeclips
  
# 	def __len__(self):
# 		return len(self.samples_list)

# 	def __getitem__(self, idx):

# 		sample_line = self.samples_list[idx].strip()
# 		seq_path_line, frame_path, mask_path = sample_line.split(';')
# 		seq_path = seq_path_line.split(',')
		
# 		seq = []
# 		for i in range(self.timeclips):
# 			img = Image.open(seq_path[i]).convert('RGB')
# 			seq.append(img)

# 		mask = Image.open(mask_path).convert('L')  # mask
# 		frame = Image.open(frame_path).convert('RGB')  # H W C

# 		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'frame_path': frame_path}
# 		if self.transform is not None:
# 			sample = self.transform(sample)
# 		return sample


# class TestDatasetHS(Dataset):

# 	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess',
#             	transform=None, timeclips=4):

# 		samples_list_file = os.path.join(file_path_root, 'timeclips_{}/TestHardDataset_Seen.txt'.format(timeclips))
# 		f = open(samples_list_file, "r")
# 		self.samples_list = f.readlines()
# 		self.transform = transform
# 		self.timeclips = timeclips
  
# 	def __len__(self):
# 		return len(self.samples_list)

# 	def __getitem__(self, idx):

# 		sample_line = self.samples_list[idx].strip()
# 		seq_path_line, frame_path, mask_path = sample_line.split(';')
# 		seq_path = seq_path_line.split(',')
		
# 		seq = []
# 		for i in range(self.timeclips):
# 			img = Image.open(seq_path[i]).convert('RGB')
# 			seq.append(img)

# 		mask = Image.open(mask_path).convert('L')  # mask
# 		frame = Image.open(frame_path).convert('RGB')  # H W C

# 		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'frame_path': frame_path}
# 		if self.transform is not None:
# 			sample = self.transform(sample)
# 		return sample


# class TestDatasetHU(Dataset):

# 	def __init__(self, file_path_root='Diff_VPS/dataloader/preprocess',
#             	transform=None, timeclips=4):

# 		samples_list_file = os.path.join(file_path_root, 'timeclips_{}/TestHardDataset_Unseen.txt'.format(timeclips))
# 		f = open(samples_list_file, "r")
# 		self.samples_list = f.readlines()
# 		self.transform = transform
# 		self.timeclips = timeclips
  
# 	def __len__(self):
# 		return len(self.samples_list)

# 	def __getitem__(self, idx):

# 		sample_line = self.samples_list[idx].strip()
# 		seq_path_line, frame_path, mask_path = sample_line.split(';')
# 		seq_path = seq_path_line.split(',')
		
# 		seq = []
# 		for i in range(self.timeclips):
# 			img = Image.open(seq_path[i]).convert('RGB')
# 			seq.append(img)

# 		mask = Image.open(mask_path).convert('L')  # mask
# 		frame = Image.open(frame_path).convert('RGB')  # H W C

# 		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'frame_path': frame_path}
# 		if self.transform is not None:
# 			sample = self.transform(sample)
# 		return sample


def get_video_dataset():
    statistics = torch.load("Diff_VPS/dataloader/statistics.pth")
    trsf_main = Compose_train([
        Resize_train(384, 448),
        Random_crop_Resize_train(7),
        Random_horizontal_flip_train(0.5),
        toTensor_train(),
        Normalize_train(statistics["mean"], statistics["std"])
    ])
    train_loader = TrainDataset(transform=trsf_main)

    return train_loader