import math

import numpy as np
import torch
import numpy
import cv2
import torch.utils.data as data
import os
import scipy.io as scio
from PIL import Image
import matplotlib.pyplot as plt


# This is used for domain adaptation
# self 这里为一个可以互通的桥梁
class vip_DA(data.Dataset):
    def __init__(self, root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database"):
        self.train_root = os.path.join(root, "train")
        self.valid_root = os.path.join(root, "valid")
        self.path_U = []
        self.path_C = []
        for _, folders, _ in os.walk(self.train_root):
            folders = sorted(folders)
            folders = folders[:30]
            for folder in folders:
                dir_ = os.path.join(self.train_root, folder)
                dir_ = os.path.join(dir_, "IR", "uncover")
                for _, _, files in os.walk(dir_):
                    files = sorted(files)
                    for file in files:
                        #files = sorted(files)
                        path_U = os.path.join(dir_, file)
                        self.path_U.append(path_U)
                    break
            break
        for _, folders, _ in os.walk(self.valid_root):
            folders = sorted(folders)
            folders1 = folders[:5]
            for folder in folders1:
                dir_valid = os.path.join(self.valid_root, folder)
                dir_valid = os.path.join(dir_valid, "IR", "cover1")
                for _, _, files in os.walk(dir_valid):
                    files = sorted(files)
                    for file in files:
                        path_C = os.path.join(dir_valid, file)
                        self.path_C.append(path_C)
                    break
            break
        for _, folders, _ in os.walk(self.valid_root):
            folders = sorted(folders)
            folders2 = folders[-5:]
            for folder in folders2:
                dir_valid = os.path.join(self.valid_root, folder)
                dir_valid = os.path.join(dir_valid, "IR", "cover2")
                for _, _, files in os.walk(dir_valid):
                    files = sorted(files)
                    for file in files:
                        path_C = os.path.join(dir_valid, file)
                        self.path_C.append(path_C)
                    break
            break





    def __getitem__(self, item):  # 这里的item为指针
        path_U = self.path_U[item]
        path_C = self.path_C[item]
        print(path_U)
        print(path_C)
        #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #image_U = cv2.imread(path_U, cv2.IMREAD_COLOR)
        #image_C = cv2.imread(path_C, cv2.IMREAD_COLOR)
        image_U = cv2.imread(path_U, cv2.IMREAD_GRAYSCALE)
        image_C = cv2.imread(path_C, cv2.IMREAD_GRAYSCALE)
        image_U = image_U / 255.  # Normalize
        image_C = image_C / 255.  # Normalize
        top = 0
        bottom = 0
        left = 0
        right = 8
        value = [0, 0, 0]
        borderType = cv2.BORDER_CONSTANT
        image_U = cv2.copyMakeBorder(image_U, top, bottom, left, right, borderType, None, value)
        image_C = cv2.copyMakeBorder(image_C, top, bottom, left, right, borderType, None, value)
        #print(self.path)
        #print(self.path_labels)
        #image = image[np.newaxis, :, :]  # 1 x w x h 增加一个维度，灰图
        #print(np.shape(image))
        # 注，有关image维度，如果为4维，a*b*c*d, a为batch size，意为每次导入几张图片，
        # b 为 1，则灰度图像，为3，则为彩图 (channel)
        # 剩下两个是长度和宽度的数据
        #image_U = image_U.transpose(2, 0, 1)
        #image_C = image_C.transpose(2, 0, 1)
        #print(np.shape(image))
        image_U = torch.from_numpy(image_U).float()  # numpy -> tensor
        image_C = torch.from_numpy(image_C).float()  # numpy -> tensor

        return image_U, image_C

    def __len__(self):
        return len(self.path_U)
    # def adjust_target_weight(self, joint, target_weight, tmp_size):
    #     # feat_stride = self.image_size / self.heatmap_size
    #     mu_x = joint[0] # 40
    #     mu_y = joint[1] # 32
    #
    #     # Check that any part of the gaussian is in-bounds
    #     ul = [int(40 - tmp_size), int(32 - tmp_size)]
    #     br = [int(40 + tmp_size + 1), int(32 + tmp_size + 1)]
    #     if ul[0] >= 32 or ul[1] >= 40 \
    #             or br[0] < 0 or br[1] < 0:
    #         # If not, just return the image as is
    #         target_weight = 0
    #
    #     return target_weight


if __name__ == '__main__':
    VIP_Data = vip_DA(root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database")
    vip_loader = data.DataLoader(VIP_Data, batch_size=1, num_workers=0, shuffle=False)

    for i, (image_U, image_C) in enumerate(vip_loader):
         print("one cycle starts")
         image_U = image_U.numpy()
         #print(np.shape(image))
         #image = image.reshape(160, 128)
         image_C = image_C.numpy()
         #print(np.shape(targets))
         #sum_targets = sum_targets.numpy()
         image_U = image_U[0]
         image_C = image_C[0]
         #print(np.shape(targets))
         #sum_targets = sum_targets.reshape(40, 32)
         plt.subplot(221)
         plt.imshow(image_U)
         plt.subplot(222)
         #points = plt.plot(x, y, 'ro')
         plt.imshow(image_C)
         #plt.imshow(new_targets)
         plt.show()
         #print(path)
         if i == 2:
             break



