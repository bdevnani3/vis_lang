import torch
import torch.nn as nn
from cifar10_emb_removeclasses import Cifar10EmbRemoveClasses

from utils import init_root


class Cifar10EmbLossDot(Cifar10EmbRemoveClasses):
    def __init__(
        self,
        root_path,
        classes_to_remove=None,
        variant_name="cifar10_emb_loss_dot",
        epochs=200,
    ):
        super().__init__(
            root_path=root_path,
            variant_name=variant_name,
            epochs=epochs,
            classes_to_remove=classes_to_remove,
        )
        self.similarity_mode = "dot"

    def calc_loss(self, outputs, labels):
        loss = torch.mean(self.criterion(outputs, self.model.word_lookup[labels]))
        return torch.norm(outputs) * torch.norm(labels) * (1.0 - loss)


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar10EmbLossDot(root_path=root_path)
    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.CosineSimilarity)

    variant.init_word_lookup()
    variant.train_model()
