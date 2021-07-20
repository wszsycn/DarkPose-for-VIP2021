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




class vip(data.Dataset):
    def __init__(self, root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database", mode="train"):
        self.root = os.path.join(root, mode)
        self.path = []
        self.labels = []
        self.path_labels = []
        if (mode == "train"):
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders = folders[:30]
                for folder in folders:
                    dir_ = os.path.join(self.root, folder)
                    dir_ = os.path.join(dir_, "IR", "uncover")
                    for _, _, files in os.walk(dir_):
                        files = sorted(files)
                        for file in files:
                            #files = sorted(files)
                            path = os.path.join(dir_, file)
                            self.path.append(path)
                        break
                break
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders = folders[:30]
                for folder in folders:
                    dir_labels = os.path.join(self.root, folder)
                    #dir_ = os.path.join(dir_, "IR", "uncover")
                    for _, _, files in os.walk(dir_labels):
                        files = sorted(files)
                        for file in files:
                            if (file == 'joints_gt_IR.mat'):
                                path_labels = os.path.join(dir_labels, file)
                                matdata = scio.loadmat(path_labels)
                                labels = matdata["joints_gt"]
                                labels = labels / 4
                                #print(labels)
                                #print(np.shape(labels))
                                self.labels.append(labels)
                                self.path_labels.append(path_labels)
                        break
                break
        if (mode == "valid"):
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders1 = folders[:5]
                for folder in folders1:
                    dir_ = os.path.join(self.root, folder)
                    dir_ = os.path.join(dir_, "IR", "cover1")
                    for _, _, files in os.walk(dir_):
                        files = sorted(files)
                        for file in files:
                            path = os.path.join(dir_, file)
                            self.path.append(path)
                        break
                break
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders1 = folders[:5]
                for folder in folders1:
                    dir_labels = os.path.join(self.root, folder)
                    #dir_ = os.path.join(dir_, "IR", "uncover")
                    for _, _, files in os.walk(dir_labels):
                        files = sorted(files)
                        for file in files:
                            if (file == 'joints_gt_IR.mat'):
                                path_labels = os.path.join(dir_labels, file)
                                matdata = scio.loadmat(path_labels)
                                labels = matdata["joints_gt"]
                                labels = labels / 4
                                self.labels.append(labels)
                        break
                break
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders2 = folders[-5:]
                for folder in folders2:
                    dir_ = os.path.join(self.root, folder)
                    dir_ = os.path.join(dir_, "IR", "cover2")
                    for _, _, files in os.walk(dir_):
                        files = sorted(files)
                        for file in files:
                            path = os.path.join(dir_, file)
                            self.path.append(path)
                        break
                break
            for _, folders, _ in os.walk(self.root):
                folders = sorted(folders)
                folders2 = folders[-5:]
                for folder in folders2:
                    dir_labels = os.path.join(self.root, folder)
                    #dir_ = os.path.join(dir_, "IR", "uncover")
                    for _, _, files in os.walk(dir_labels):
                        files = sorted(files)
                        for file in files:
                            if (file == 'joints_gt_IR.mat'):
                                path_labels = os.path.join(dir_labels, file)
                                matdata = scio.loadmat(path_labels)
                                labels = matdata["joints_gt"]
                                labels = labels / 4
                                self.labels.append(labels)
                        break
                break





    def __getitem__(self, item):
        path = self.path[item]
        #image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = image / 255.  # Normalize
        top = 0
        bottom = 0
        left = 0
        right = 8
        value = [0, 0, 0]
        borderType = cv2.BORDER_CONSTANT
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType, None, value)
        #print(self.path)
        #print(self.path_labels)
        #image = image[np.newaxis, :, :]
        #print(np.shape(image))
        image = image.transpose(2, 0, 1)
        #print(np.shape(image))
        image = torch.from_numpy(image).float()  # numpy -> tensor
        targets = self.labels[math.floor(item/45)]
        targets = targets[0:2, :, :] # Take the first two columns 2 * 14 * 45
        #test = targets.transpose(2, 0, 1)
        #print(test)
        a = np.zeros((45, 14, 32, 40)) # 45 pictures in one folder， 14 feature points， 32 * 40 pixels, x = 32, y = 40
        targets_weights = np.ones((14, 1), dtype=np.float32)
        for num_pictures in range(0,45):
            for num_points in range(0,14):

                x = np.arange(0, 32, 1, np.float32)
                y = np.arange(0, 40, 1, np.float32)
                y = y[:, np.newaxis]
                mu_x = targets[0][num_points][num_pictures]
                mu_y = targets[1][num_points][num_pictures]
                Gaussian = np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * 2 ** 2))
                Gaussian = Gaussian.transpose(1, 0)
                a[num_pictures][num_points] = Gaussian
                #a[num_pictures][num_points][int(round(mu_x))][int(round(mu_y))] = 1
                #targets_weights = np.multiply(targets_weights, self.joints_weight)
        #print(np.shape(a)) # 45, 14, 32, 40
        a = a.transpose(0, 1, 3, 2)
        #print(np.shape(a)) # 45, 14, 40, 32
        targets = a[item % 45]
        #print(path)
        #print(np.shape(targets))
        #sum_targets = np.zeros((40, 32))
        #targets = targets[np.newaxis]
        targets_weights = torch.from_numpy(targets_weights).float()
        targets = torch.from_numpy(targets).float()
        #print(np.shape(targets))
        #return image, targets, path
        return image, targets, targets_weights

    def __len__(self):
        return len(self.path)
    def adjust_target_weight(self, joint, target_weight, tmp_size):
        # feat_stride = self.image_size / self.heatmap_size
        mu_x = joint[0] # 40
        mu_y = joint[1] # 32

        # Check that any part of the gaussian is in-bounds
        ul = [int(40 - tmp_size), int(32 - tmp_size)]
        br = [int(40 + tmp_size + 1), int(32 + tmp_size + 1)]
        if ul[0] >= 32 or ul[1] >= 40 \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            target_weight = 0

        return target_weight


if __name__ == '__main__':
    VIP_Data = vip(root="/home/vip2021/NewDarkDataset/SLP_VIPCup_database", mode="train")
    vip_loader = data.DataLoader(VIP_Data, batch_size=1, num_workers=0, shuffle=True)

    for i, (image, targets, targets_weights) in enumerate(vip_loader):
         print("one cycle starts")
         image = image.numpy()
         #print(np.shape(image))
         #image = image.reshape(160, 128)
         targets = targets.numpy()
         #print(np.shape(targets))
         #sum_targets = sum_targets.numpy()
         image = image[0]
         targets = targets[0]
         #print(np.shape(targets))
         #sum_targets = sum_targets.reshape(40, 32)
         #plt.subplot(221)
         #plt.imshow(image)
         #plt.subplot(222)
         #points = plt.plot(x, y, 'ro')
         #plt.imshow(sum_targets)
         new_targets = np.zeros((160, 128))
         for num in range(0,14):
             # plt.subplot(4, 7, num+15)
             new_targets = new_targets + targets[num]
             #plt.imshow(targets[num])
         #print(np.shape(new_targets))
         new_targets = new_targets + image
         plt.imshow(new_targets)
         plt.show()
         #print(path)
         if i == 2:
             break



