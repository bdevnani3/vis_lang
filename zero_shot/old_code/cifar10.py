import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets

from utils import init_root
import os

from base import Base


class Cifar10(Base):
    def __init__(self, root_path, variant_name="cifar10_base", epochs=200):
        super(Cifar10, self).__init__(
            root_path=root_path, variant_name=variant_name, epochs=epochs
        )

    def init_datasets(self):

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
            **self.dataset_args,
            train=True,
            transform=transform_train,
        )

        self.class_names = self.train_dataset.classes

        self.test_dataset = datasets.CIFAR10(
            **self.dataset_args,
            train=False,
            transform=transform_test,
        )


if __name__ == "__main__":

    root_path = init_root()
    cifar10 = Cifar10(root_path=root_path)
    cifar10.init_datasets()
    cifar10.init_dataloaders()
    cifar10.set_up_model_architecture(10)
    cifar10.init_model_helpers(nn.CrossEntropyLoss)
    cifar10.train_model()
