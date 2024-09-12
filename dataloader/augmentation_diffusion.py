import random
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor as torchtotensor  # 把PIL.Image或ndarray从 (H x W x C)形状转换为 (C x H x W) 的tensor
import torch.nn.functional as F
import pdb

# Transforms
class Compose_train(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            seq, frame, mask, reconstruct_gt, clas, bbox = sample['seq'], sample['frame'], sample['mask'], \
										 sample['reconstruct_gt'], sample['clas'], sample['bbox']
            seq, frame, mask, reconstruct_gt, clas, bbox = t(seq, frame, mask, reconstruct_gt, clas, bbox) 
            sample['seq'] = seq
            sample['frame'] = frame
            sample['mask'] = mask
            sample['reconstruct_gt'] = reconstruct_gt
            sample['clas'] = clas
            sample['bbox'] = bbox
        return sample


# 随机裁剪
class Random_crop_Resize_train(object):
    def _randomCrop(self, seq, frame, mask, reconstruct_gt, clas, bbox, x, y):
        width, height = frame.size  # "width": 1158/1240, "height": 1080
        region = [x, y, width - x, height - y]
        for i in range(len(seq)):
            seq[i] = seq[i].crop(region)
        frame, mask, reconstruct_gt = frame.crop(region), mask.crop(region), reconstruct_gt.crop(region)
        old_width, old_height = frame.size
        bbox[0] = max(bbox[0] - x, 0)
        bbox[1] = max(bbox[1] - y, 0)
        
        for i in range(len(seq)):
            seq[i] = seq[i].resize((width, height), Image.BILINEAR)
        frame = frame.resize((width, height), Image.BILINEAR)
        mask = mask.resize((width, height), Image.NEAREST)
        reconstruct_gt = reconstruct_gt.resize((width, height), Image.BILINEAR)
        bbox[0] = bbox[0] / old_width * width  # 要回到原来的size
        bbox[1] = bbox[1] / old_height * height
        bbox[2] = bbox[2] / old_width * width
        bbox[3] = bbox[3] / old_height * height
        return seq, frame, mask, reconstruct_gt, clas, bbox

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        # x,y设定裁剪的边界
        # 不确定这里的list是否可行
        crop_seq, crop_frames, crop_masks, crop_reconstruct_gts, crop_clases, crop_bboxs = [], [], [], [], [], []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        seq, frame, mask, reconstruct_gt, clas, bbox = self._randomCrop(seq, frame, mask, reconstruct_gt, clas, bbox, x, y)

        return seq, frame, mask, reconstruct_gt, clas, bbox


# 随机水平翻转
class Random_horizontal_flip_train(object):
    def _horizontal_flip(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        for i in range(len(seq)):
            seq[i] = seq[i].transpose(Image.FLIP_LEFT_RIGHT)
        frame = frame.transpose(Image.FLIP_LEFT_RIGHT)  
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        reconstruct_gt = reconstruct_gt.transpose(Image.FLIP_LEFT_RIGHT)
        width, _ = frame.size
        bbox[0] = width - bbox[0] - bbox[2]
        return seq, frame, mask, reconstruct_gt, clas, bbox

    def __init__(self, prob):
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        if random.random() < self.prob:
            flip_seq, flip_frames, flip_masks, flip_reconstruct_gts, flip_clases, flip_bboxs = [], [], [], [], [], []
            seq, frame, mask, reconstruct_gt, clas, bbox = self._horizontal_flip(seq, frame, mask, reconstruct_gt, clas, bbox)

            return seq, frame, mask, reconstruct_gt, clas, bbox
        else:
            return seq, frame, mask, reconstruct_gt, clas, bbox


def box_xywh_to_cxcywh(x):
    [x0, y0, w, h] = x
    b = [x0 + w / 2, y0 + h / 2, w, h]
    return torch.Tensor(b)


class Resize_train(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        old_width, old_height = frame.size  # [1058, 1008]
        for i in range(len(seq)):
            seq[i] = seq[i].resize((self.width, self.height), Image.BILINEAR)
            seq[i] = np.array(seq[i], dtype=np.float32)
        frame = frame.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        reconstruct_gt = reconstruct_gt.resize((self.width, self.height), Image.BILINEAR)
        bbox = box_xywh_to_cxcywh(bbox)
        bbox[0] = bbox[0] / old_width
        bbox[1] = bbox[1] / old_height
        bbox[2] = bbox[2] / old_width
        bbox[3] = bbox[3] / old_height
        return seq, frame, mask, reconstruct_gt, clas, bbox


# Convert Img to Tensors
class toTensor_train(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        # 将seq变为shape = H * W * 4C
        seq = np.stack(seq)
        seq = seq.transpose((0, 3, 1, 2))
        seq = torch.from_numpy(seq)
        frame, mask = self.totensor(frame), self.totensor(mask)
        reconstruct_gt = frame
        # bbox = torch.tensor(bbox)
        return seq, frame, mask, reconstruct_gt, clas, bbox


# 通道归一化
class Normalize_train(object):
    def __init__(self, mean, std):
        self.mean, self.std = torch.from_numpy(mean), torch.from_numpy(std)

    def __call__(self, seq, frame, mask, reconstruct_gt, clas, bbox):
        self.mean = self.mean.view(1, 3, 1, 1)
        self.std = self.std.view(1, 3, 1, 1)
        seq = seq / 255
        seq[:, 0:3, :, :] -= self.mean
        seq[:, 0:3, :, :] /= self.std

        self.mean = self.mean.view(3, 1, 1)
        self.std = self.std.view(3, 1, 1)
        frame[0:3, :, :] -= self.mean
        frame[0:3, :, :] /= self.std
        reconstruct_gt = frame
        return seq, frame, mask, reconstruct_gt, clas, bbox

    
# Validation
class Compose_valid(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            seq, frame, mask = sample['seq'], sample['frame'], sample['mask']
            seq, frame, mask = t(seq, frame, mask) 
            sample['seq'] = seq
            sample['frame'] = frame
            sample['mask'] = mask
        return sample


class Resize_valid(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, seq, frame, mask):
        old_width, old_height = frame.size
        for i in range(len(seq)):
            seq[i] = seq[i].resize((self.width, self.height), Image.BILINEAR)
        frame = frame.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        return seq, frame, mask


# 通道归一化
class Normalize_valid(object):
    def __init__(self, mean, std):
        self.mean, self.std = torch.from_numpy(mean), torch.from_numpy(std)

    def __call__(self, seq, frame, mask):
        self.mean = self.mean.view(1, 3, 1, 1)
        self.std = self.std.view(1, 3, 1, 1)
        seq = seq / 255
        seq[:, 0:3, :, :] -= self.mean
        seq[:, 0:3, :, :] /= self.std
        
        self.mean = self.mean.view(3, 1, 1)
        self.std = self.std.view(3, 1, 1)
        frame[0:3, :, :] -= self.mean
        frame[0:3, :, :] /= self.std
        return seq, frame, mask


# Convert Img to Tensors
class toTensor_valid(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, seq, frame, mask):
        for i in range(len(seq)):
            seq[i] = np.array(seq[i], dtype=np.float32)
        # 将seq变为shape = H * W * 4C
        seq = np.stack(seq)
        seq = seq.transpose((0, 3, 1, 2))
        seq = torch.from_numpy(seq)
        frame, mask = self.totensor(frame), self.totensor(mask)
        return seq, frame, mask
    
    
# Test
class Compose_test(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            seq, frame, mask, frame_path = sample['seq'], sample['frame'], sample['mask'], sample['frame_path']
            seq, frame, mask, frame_path = t(seq, frame, mask, frame_path) 
            sample['seq'] = seq
            sample['frame'] = frame
            sample['mask'] = mask
            sample['frame_path'] = frame_path
        return sample


class Resize_test(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, seq, frame, mask, frame_path):
        old_width, old_height = frame.size
        for i in range(len(seq)):
            seq[i] = seq[i].resize((self.width, self.height), Image.BILINEAR)
        frame = frame.resize((self.width, self.height), Image.BILINEAR)
        mask = mask.resize((self.width, self.height), Image.NEAREST)
        return seq, frame, mask, frame_path


# 通道归一化
class Normalize_test(object):
    def __init__(self, mean, std):
        self.mean, self.std = torch.from_numpy(mean), torch.from_numpy(std)

    def __call__(self, seq, frame, mask, frame_path):
        
        self.mean = self.mean.view(1, 3, 1, 1)
        self.std = self.std.view(1, 3, 1, 1)
        seq = seq / 255
        seq[:, 0:3, :, :] -= self.mean
        seq[:, 0:3, :, :] /= self.std
        
        self.mean = self.mean.view(3, 1, 1)
        self.std = self.std.view(3, 1, 1)
        frame[0:3, :, :] -= self.mean
        frame[0:3, :, :] /= self.std

        return seq, frame, mask, frame_path


# Convert Img to Tensors
class toTensor_test(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, seq, frame, mask, frame_path):
        for i in range(len(seq)):
            seq[i] = np.array(seq[i], dtype=np.float32)
        # 将seq变为shape = H * W * 4C
        seq = np.stack(seq)
        seq = seq.transpose((0, 3, 1, 2))
        seq = torch.from_numpy(seq)
        frame, mask = self.totensor(frame), self.totensor(mask)
        return seq, frame, mask, frame_path
    
    
from torch.utils.data import Dataset
class TrainDataset(Dataset):

	def __init__(self, samples_list_file='Diff_VPS/txt/SegFormer/PNS_train.txt',
				 transform=None, num_frame=4):

		f = open(samples_list_file, "r")
		self.samples_list = f.readlines()
		self.transform = transform
		self.num_frame = num_frame
  
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
		for i in range(self.num_frame):
			img = Image.open(seq_path[i]).convert('RGB')
			seq.append(img)
		mask = Image.open(mask_path).convert('L')  # mask
		frame = Image.open(frame_path).convert('RGB')  # H W C
		sample = {'seq': seq, 'frame': frame, 'mask': mask, 'reconstruct_gt': frame, 'clas': clas, 'bbox' : bbox}
		if self.transform is not None:
			sample = self.transform(sample)
		return sample


class ValidDataset(Dataset):

	def __init__(self, samples_list_file='Diff_VPS/txt/SegFormer/TestEasyDataset_Seen.txt', 
            	transform=None, num_frame=4):

		f = open(samples_list_file, "r")
		self.samples_list = f.readlines()
		self.transform = transform
		self.num_frame = num_frame
  
	def __len__(self):
		return len(self.samples_list)

	def __getitem__(self, idx):

		sample_line = self.samples_list[idx].strip()
		seq_path_line, frame_path, mask_path = sample_line.split(';')
		seq_path = seq_path_line.split(',')
		
		seq = []
		for i in range(self.num_frame):
			img = Image.open(seq_path[i]).convert('RGB')
			seq.append(img)

		mask = Image.open(mask_path).convert('L')  # mask
		frame = Image.open(frame_path).convert('RGB')  # H W C

		sample = {'seq': seq, 'frame': frame, 'mask': mask}
		if self.transform is not None:
			sample = self.transform(sample)
		return sample
    

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
def bbox_to_rect(bbox, color):  
    return plt.Rectangle(
        xy=((bbox[0] - bbox[2] / 2)*256 , (bbox[1] - bbox[3] / 2)*256), width=bbox[2]*256, height=bbox[3]*256,
        fill=False, edgecolor=color, linewidth=2)


if __name__ == "__main__":

    statistics = torch.load("Diff_VPS/dataloader/statistics.pth")

    train_transforms = Compose_train([
            Resize_train(256, 256),
            toTensor_train(),
            Normalize_train(statistics["mean"], statistics["std"])
        ])
    valid_transforms = Compose_valid([
            Resize_valid(256, 256),
            toTensor_valid(),
            Normalize_valid(statistics["mean"], statistics["std"])
        ])
    train_dataset = TrainDataset(transform=train_transforms, num_frame=4)
    valid_dataset = ValidDataset(samples_list_file='Diff_VPS/txt/SegFormer/TestEasyDataset_Seen.txt', transform=valid_transforms, num_frame=4)
    sample = train_dataset[200]
    img = sample['frame']
    mean = torch.from_numpy(statistics["mean"]).view(3, 1, 1)
    std = torch.from_numpy(statistics["std"]).view(3, 1, 1)
    img[0:3, :, :] *= std
    img[0:3, :, :] += mean
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    bbox = sample['bbox'].numpy().tolist()
    fig = plt.imshow(img)
    fig.axes.add_patch(bbox_to_rect(bbox, 'red'))
    plt.savefig("Diff_VPS/for_check/img_bbox3.jpg")
    img.save("Diff_VPS/for_check/img3.jpg")
    
    from pandas.core.frame import DataFrame
    clas = []
    for i in range(len(train_dataset)):
        clas.append(train_dataset[i]['clas'])
    Clas = {'Clas': clas}
    Clas = DataFrame(Clas)
    print(Clas)