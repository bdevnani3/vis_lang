import numpy as np
import torch.nn as nn
from torch.utils.data import Subset


from cifar10_emb_removeclasses import Cifar10EmbRemoveClasses
from cifar100 import Cifar100

import os


class Cifar100Emb(Cifar10EmbRemoveClasses, Cifar100):
    def __init__(self, root_path, variant_name="cifar100_emb", epochs=200):
        super().__init__(
            root_path=root_path,
            variant_name=variant_name,
            epochs=epochs,
            classes_to_remove=[],
        )

    def init_datasets(self):
        Cifar100.init_datasets(self)

        # Detect missing classes
        for c in self.train_dataset.classes:
            if c not in self.word_vectors:
                self.classes_to_remove.append(c)


if __name__ == "__main__":

    if os.path.exists("/nethome/bdevnani3/raid"):
        root_path = "/nethome/bdevnani3/raid"
    else:
        root_path = "."

    variant = Cifar100Emb(root_path=root_path, epochs=300)

    variant.init_datasets()
    variant.remove_classes()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
