import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets


import os

from base import Base


class Cifar100(Base):
    def __init__(self, root_path, variant_name="cifar100_base", epochs=200):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)

    def init_datasets(self):

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
            **self.dataset_args,
            train=True,
            transform=transform_train,
        )

        self.class_names = self.train_dataset.classes

        self.test_dataset = datasets.CIFAR100(
            **self.dataset_args,
            train=False,
            transform=transform_test,
        )


if __name__ == "__main__":

    if os.path.exists("/nethome/bdevnani3/raid"):
        root_path = "/nethome/bdevnani3/raid"
    else:
        root_path = "."

    variant = Cifar100(root_path=root_path, epochs=300)

    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(100)
    variant.init_model_helpers(nn.CrossEntropyLoss)
    variant.train_model()
