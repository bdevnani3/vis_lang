import torch.nn as nn

from cifar100_emb import Cifar100Emb
from utils import init_root


class Cifar100EmbRemoveClasses(Cifar100Emb):
    def __init__(
        self,
        root_path,
        classes_to_remove=None,
        variant_name="cifar100_emb_removeclasses",
        epochs=200,
    ):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)
        self.classes_to_remove.extend(classes_to_remove if classes_to_remove else [])


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar100EmbRemoveClasses(
        root_path=root_path, epochs=300, classes_to_remove=["camel"]
    )
    variant.init_word_vectors()
    variant.init_datasets()
    variant.remove_classes()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
