from cifar10_emb import Cifar10Emb
from torch.utils.data import Subset
import numpy as np
import torch.nn as nn

from utils import init_root


class Cifar10EmbRemoveClasses(Cifar10Emb):
    def __init__(
        self,
        root_path,
        classes_to_remove=None,
        variant_name="cifar10_emb_removeclasses",
        epochs=200,
    ):
        super(Cifar10EmbRemoveClasses, self).__init__(
            root_path=root_path, variant_name=variant_name, epochs=epochs
        )
        self.classes_to_remove = classes_to_remove

    def remove_classes(self):

        new_datasets = []
        for dataset in [self.train_dataset, self.test_dataset]:
            for _class in self.classes_to_remove:
                assert (
                    _class in dataset.classes
                ), f"{_class} not in the selected Dataset"

            mask = np.ones((len(dataset)), dtype=bool)

            for c in self.classes_to_remove:
                # Get index for class
                idx = dataset.class_to_idx[c]

                # Just a hack to ensure class idx's don't have to be moved around.
                # Since this vector will never be used, it doesn't really matter
                # what the values are.
                self.word_vectors[c] = np.zeros(300)

                # get indices where targets == idx
                mask = np.logical_and(mask, np.array(dataset.targets) != idx)

            # Use mask on dataset
            mask_indices = np.where(mask > 0)[0]
            new_datasets.append(Subset(dataset, mask_indices))

        self.train_dataset = new_datasets[0]
        self.test_dataset = new_datasets[1]


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar10EmbRemoveClasses(root_path=root_path, classes_to_remove=["cat"])

    variant.init_datasets()
    variant.remove_classes()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
