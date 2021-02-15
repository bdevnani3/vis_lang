import torch.nn as nn

from cifar10_emb_removeclasses import Cifar10EmbRemoveClasses
from cifar100 import Cifar100
from utils import init_root


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

    root_path = init_root()
    variant = Cifar100Emb(root_path=root_path, epochs=300)

    variant.init_word_vectors()
    variant.init_datasets()
    variant.remove_classes()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
