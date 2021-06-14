from torchvision.datasets import CIFAR100
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
import os
import torch
import numpy as np


class ClipExptDataset:
    def __init__(self, num_workers, batch_size):
        self.name = "Not Defined"

        self.train_loader = None
        self.test_loader = None
        self.validate_loader = None

        self.clip_train_loader = None
        self.clip_test_loader = None
        self.clip_validate_loader = None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.valid_size = 0.1  # 10% of training data to be used as validation split
        self.classes = None

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomRotation(45),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Where the datasets are saved, override if location changes
        self.root = os.path.expanduser("/nethome/bdevnani3/raid/data/")

    def get_train_loaders(self, transform_fn):
        raise NotImplementedError

    def get_test_loader(self, transform_fn):
        raise NotImplementedError


##################################
############ DATASETS ############
##################################


class Cifar100(ClipExptDataset):
    def __init__(self, num_workers, batch_size):

        super().__init__(num_workers, batch_size)
        self.name = "CIFAR100"
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )  # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )

        valid_dataset = CIFAR100(
            root=self.root,
            train=True,
            download=True,
            transform=transform_fn,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers,
        )

        # Little hacky - need to improve
        if self.classes == None:
            self.classes = train_dataset.classes

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = CIFAR100(
            root=self.root,
            train=False,
            download=True,
            transform=transform_fn,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if self.classes == None:
            self.classes = test_dataset.classes

        return test_loader


##################################


class Flowers102(ClipExptDataset):
    def __init__(self, num_workers, batch_size):
        super().__init__(num_workers, batch_size)
        self.name = "Flowers102"

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "flower_data/train", transform=transform_fn
        )

        valid_dataset = datasets.ImageFolder(
            self.root + "flower_data/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if self.classes == None:
            self.classes = train_dataset.classes

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "flower_data/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if self.classes == None:
            self.classes = test_dataset.classes

        return test_loader


##################################


class OxfordPets(ClipExptDataset):
    def __init__(self, num_workers, batch_size):
        super().__init__(num_workers, batch_size)
        self.name = "OxfordPets"

    def get_train_loaders(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.train_transform

        train_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/train", transform=transform_fn
        )

        valid_dataset = datasets.ImageFolder(
            self.root + "oxford_pets/valid", transform=transform_fn
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        if self.classes == None:
            self.classes = train_dataset.classes

        return train_loader, valid_loader

    def get_test_loader(self, transform_fn=None):

        if transform_fn is None:
            transform_fn = self.test_transform

        test_dataset = datasets.ImageFolder(
            self.root + "oxfordpets_data/test", transform=transform_fn
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        if self.classes == None:
            self.classes = test_dataset.classes

        return test_loader
