from PIL import Image
import numpy as np
import scipy.io as scio
from torchvision import transforms
import h5py
import torch.utils.data


LABEL_DIR = '/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/lab.mat'
TXT_DIR = '/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/txt.mat'
IMG_DIR = '/home/chen/PycharmProjects/pythonProject/MGCH/Datasets/mirflickr/img.mat'

label_set = scio.loadmat(LABEL_DIR)
label_set = np.array(label_set['lab'], dtype=float)
txt_set = scio.loadmat(TXT_DIR)
txt_set = np.array(txt_set['txt'], dtype=float)
img_set = scio.loadmat(IMG_DIR)
img_set = np.array(img_set['img'], dtype=float)

first = True
for label in range(label_set.shape[1]):
    index = np.where(label_set[:, label] == 1)[0]

    N = index.shape[0]
    perm = np.random.permutation(N)
    index = index[perm]

    if first:
        test_index = index[:160]
        train_index = index[160:160 + 400]
        first = False
    else:
        ind = np.array([i for i in list(index) if i not in (list(train_index) + list(test_index))])
        test_index = np.concatenate((test_index, ind[:80]))
        train_index = np.concatenate((train_index, ind[80:80 + 200]))

database_index = np.array([i for i in list(range(label_set.shape[0])) if i not in list(test_index)])

if train_index.shape[0] < 5000:
    pick = np.array([i for i in list(database_index) if i not in list(train_index)])
    N = pick.shape[0]
    perm = np.random.permutation(N)
    pick = pick[perm]
    res = 5000 - train_index.shape[0]
    train_index = np.concatenate((train_index, pick[:res]))

indexTest = test_index
indexDatabase = database_index
indexTrain = train_index



label_feat_len = label_set.shape[1]

class MIRFlickr(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None, train=True, database=False):
        self.transform = transform
        self.target_transform = target_transform

        if train:
            self.train_labels = label_set[indexTrain]
            self.train_index = indexTrain
            self.txt = txt_set[indexTrain]
            self.img = img_set[indexTrain]
        elif database:
            self.train_labels = label_set[indexDatabase]
            self.train_index = indexDatabase
            self.txt = txt_set[indexDatabase]
            self.img = img_set[indexDatabase]
        else:
            self.train_labels = label_set[indexTest]
            self.train_index = indexTest
            self.txt = txt_set[indexTest]
            self.img = img_set[indexTest]
    def __getitem__(self, index):

        img = self.img[index]
        txt = self.txt[index]
        target = self.train_labels[index]

        return img, txt, target, index

    def __len__(self):
        return len(self.train_labels)