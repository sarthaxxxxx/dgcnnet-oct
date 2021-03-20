import os 
import cv2
import csv
import random
import numpy as np
from torch.utils.data import Dataset


class Cityscapes(Dataset):
    r"""Dataset prep for Cityscapes. 
    Performs mean and variance normalization. 
    Added features : padding, mirroring (randomly done).
    """
    def __init__(self, 
                listdir,
                mean = (128, 128, 128),  
                var = (1, 1, 1), 
                crop_size = (321, 321), 
                scale = True, mirror = True,
                ignore_label = 255, 
                rgb = False,
                max_iters = None):
        self.mean = mean
        self.var = var
        self.crop_height, self.crop_width = crop_size
        self.scale = scale
        self.mirror = mirror
        self.list = listdir
        self.rgb = rgb
  
        self.files = []
        self.img_locs = [files.strip().split() for files in open(self.list)]

        if max_iters is not None: # for distributed training
            self.img_locs = self.img_locs * int(np.ceil(float(max_iters) / len(self.img_locs)))
            
        for _img_path, _gt_path in self.img_locs:
            self.files.append({'image' : _img_path,
            'gt' : _gt_path
            })  

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}

    def __getitem__(self, index):
        image = cv2.imread(self.files[index]['image'], cv2.IMREAD_COLOR)
        gt = self.id2trainid(cv2.imread(self.files[index]['gt'], cv2.IMREAD_GRAYSCALE))

        if self.scale:
            image, gt = self.gen_scale_label(image, gt)

        image = np.array(image).astype(np.float32)

        if self.rgb:
            image = (image[:,:,::-1]) / 255 #BGR to RGB

        image = (image - self.mean) / self.var

        height, width = gt.shape
        pad_height = max(self.crop_height - height, 0)
        pad_width = max(self.crop_width - width, 0)

        if pad_height > 0 or pad_width > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_height, 0, 
                                           pad_width, cv2.BORDER_CONSTANT, 
                                           value = (0.0, 0.0, 0.0))
            pad_gt = cv2.copyMakeBorder(gt, 0, pad_height, 0,
                                        pad_width, cv2.BORDER_CONSTANT, 
                                        value = (0.0, 0.0, 0.0))
        else:
            pad_image, pad_gt = image, gt

        height, width = pad_gt.shape
        h_off = random.randint(0, height - self.crop_height)
        w_off = random.randint(0, width - self.crop_width)

        image = np.array(image[h_off : h_off + self.crop_height, w_off : w_off + self.crop_width]).astype(np.float32)
        gt = np.array(gt[h_off : h_off + self.crop_height, w_off : w_off + self.crop_width]).astype(np.float32)

        image = image.transpose(2, 0, 1) #C, H, W

        if self.mirror:
            flip = np.random.choice(2)*2 - 1
            image = image[:, :, ::flip]
            gt = gt[:, ::flip]

        return image.copy(), gt.copy()
        
    def __len__(self):
        return len(self.files)

    def id2trainid(self, label, reverse = False):
        label_copy = label.copy()
        if reverse:
            for value, key in self.id_to_trainid.items():
                label_copy[label == key] = value
        else:
            for key, value in self.id_to_trainid.items():
                label_copy[label == key] = value 
        return label_copy

    def gen_scale_label(self, image, gt):
        f_scale = 0.7 + random.randint(0, 14) / 10.0
        image = cv2.resize(image, dsize = None, fx = f_scale, fy = f_scale, interpolation = cv2.INTER_LINEAR)
        gt = cv2.resize(gt, dsize = None, fx = f_scale, fy = f_scale, interpolation = cv2.INTER_NEAREST)
        return image, gt



if __name__=='__main__':
    dataset = Cityscapes()
    dataset.__getitem__(10)
