import torch
import torch.nn as nn
from cifar10_emb_removeclasses import Cifar10EmbRemoveClasses

from utils import init_root

mode_criterion_mapping = {
    "l2": nn.CrossEntropyLoss,
    "cossim": lambda x, y: 1 - torch.cosine_similarity(x, y),
    "dot": lambda x, y: 1 - torch.dot(x, y),
}

class Cifar10EmbLossVariants(Cifar10EmbRemoveClasses):
    def __init__(
        self,
        root_path,
        classes_to_remove=None,
        variant_name="cifar10_emb_loss",
        epochs=200,
        mode="l2"
    ):
        super().__init__(
            root_path=root_path, variant_name=f"{variant_name}_{mode}", epochs=epochs, classes_to_remove=classes_to_remove
        )
        self.similarity_mode = mode
        self.criterion = mode_criterion_mapping[self.similarity_mode]


if __name__ == "__main__":

    # This iteration loop's modes can be altered as needed before calling this file
    for m in ["l2", "cossim", "dot"]:
        print("MODE:", m)
        root_path = init_root()
        variant = Cifar10EmbLossVariants(root_path=root_path, mode=m)
        variant.init_datasets()
        variant.init_dataloaders()
        variant.set_up_model_architecture(300)
        variant.init_model_helpers()
        variant.init_word_lookup()
        variant.train_model()


