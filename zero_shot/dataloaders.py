import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets

import os

#TODO(Bhavika): This can be made more object oriented

class Cifar10:    
    def __init__(self, root=".", download=True):

        self.data_path = os.path.join(root, "data")

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.train_dataset = datasets.CIFAR10(
            root=self.data_path,
            download=download,
            train=True,
            transform=transform_train,
        )

        self.class_names = self.train_dataset.classes

        self.test_dataset = datasets.CIFAR10(
            root=self.data_path,
            download=download,
            train=False,
            transform=transform_test,
        )
    
    def init_dataloader(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class Cifar100:    
    def __init__(self, root=".", download=True):

        self.data_path = os.path.join(root, "data")

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )

        self.train_dataset = datasets.CIFAR100(
            root=self.data_path,
            download=download,
            train=True,
            transform=transform_train,
        )

        self.class_names = self.train_dataset.classes

        self.test_dataset = datasets.CIFAR100(
            root=self.data_path,
            download=download,
            train=False,
            transform=transform_test,
        )

    def init_dataloader(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)