from __future__ import print_function
from PIL import Image
import os
import os.path
import sys


if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
import numpy as np

import torch
from torchvision import datasets, transforms

from utils import *


class iCIFAR10(datasets.CIFAR10):

    def __init__(self, root, classes, task_num, train, transform=None, target_transform=None, download=True):

        super(iCIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform, download=True)

        self.train = train  # training set or   set
        if not isinstance(classes, list):
            classes = [classes]

        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels


        if not self.train:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:

                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            test_data = []
            test_labels = []
            test_tt = []  # task module labels
            test_td = []  # disctiminator labels
            for i in range(len(self.test_data)):
                if self.test_labels[i] in classes:
                    test_data.append(self.test_data[i])
                    test_labels.append(self.test_labels[i])
                    # self.class_indices[self.class_mapping[self.test_labels[i]]].append(i)

            self.test_data = np.array(test_data)
            self.test_labels = test_labels


    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img)
        except:
            pass

        try:
            if self.transform is not None:
                img = self.transform(img)
        except:
            pass
        try:
            if self.target_transform is not None:
                target = self.target_transform(target)
        except:
            pass

        return img, target




    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }



class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()

        self.seed = args.seed
        self.batch_size=args.batch_size
        self.pc_valid=args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim

        self.num_tasks = int(args.ntasks)
        self.num_classes = 100

        self.num_samples = args.samples


        self.inputsize = [3,32,32]
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]

        self.transformation = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean, std)])

        self.taskcla = [[t, int(self.num_classes/self.num_tasks)] for t in range(self.num_tasks)]

        self.indices = {}
        self.dataloaders = {}
        self.idx={}

        self.num_workers = args.workers
        self.pin_memory = False ## TODO(bdevnani3): Change back if necessary

        np.random.seed(self.seed)
        if args.task_ids == None:
            task_ids = np.split(np.random.permutation(self.num_classes),self.num_tasks)
            self.task_ids = [list(arr) for arr in task_ids]
        else:
            self.task_ids = args.task_ids

        # self.task_ids_tasksize_1 = [[x] for x in range(self.num_classes)]

        self.train_set = {}
        self.test_set = {}
        self.train_split = {}
        # self.test_set_tasksize_1 = {}

    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()

        self.train_set[task_id] = iCIFAR100(root=self.root, classes=self.task_ids[task_id], task_num=task_id, train=True, download=True, transform=self.transformation)
        self.test_set[task_id] = iCIFAR100(root=self.root, classes=self.task_ids[task_id], task_num=task_id, train=False,
                                     download=True, transform=self.transformation)
        # for i in self.task_ids_tasksize_1:
        #     self.test_set_tasksize_1[i] = iCIFAR100(root=self.root, classes=self.task_ids_tasksize_1[i], task_num=task_id, train=False,
        #                                 download=True, transform=self.transformation)

        split = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        train_split, valid_split = torch.utils.data.random_split(self.train_set[task_id], [len(self.train_set[task_id]) - split, split])

        self.train_split[task_id] = train_split
        train_loader = torch.utils.data.DataLoader(train_split, batch_size=self.batch_size, num_workers=self.num_workers,
                                                   pin_memory=self.pin_memory,shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_split, batch_size=int(self.batch_size * self.pc_valid),
                                                   num_workers=self.num_workers, pin_memory=self.pin_memory,shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_set[task_id], batch_size=self.batch_size, num_workers=self.num_workers,
                                                  pin_memory=self.pin_memory,shuffle=True)
        # for i in self.task_ids_tasksize_1:
        #     test_loader_tasksize_1 = torch.utils.data.DataLoader(self.test_set_tasksize_1[i], batch_size=self.batch_size, num_workers=self.num_workers,
        #                                             pin_memory=self.pin_memory,shuffle=True)
        #     self.dataloaders[i]['test_tasksize_1'] = test_loader_tasksize_1

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader

        self.dataloaders[task_id]['name'] = 'CIFAR100-{}-{}'.format(task_id,self.task_ids[task_id])

        self.dataloaders[task_id]['task_labels'] = self.task_ids[task_id]
        self.dataloaders[task_id]['train_labels'] = self.train_set[task_id].train_labels
        self.dataloaders[task_id]['test_labels'] = self.test_set[task_id].test_labels
        # for i in self.task_ids_tasksize_1:
        #     self.dataloaders[i]['test_tasksize_1'] = self.test_set_tasksize_1[i].test_labels

        print ("Training set size:   {} images of {}x{}".format(len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Validation set size: {} images of {}x{}".format(len(valid_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Train+Val  set size: {} images of {}x{}".format(len(valid_loader.dataset)+len(train_loader.dataset),self.inputsize[1],self.inputsize[1]))
        print ("Test set size:       {} images of {}x{}".format(len(test_loader.dataset),self.inputsize[1],self.inputsize[1]))

        return self.dataloaders
