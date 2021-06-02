import torch.nn as nn

from cifar10_emb_bert import Cifar10EmbBert
from cifar100 import Cifar100
from utils import init_root


class Cifar100EmbBert(Cifar10EmbBert, Cifar100):
    def __init__(self, root_path, variant_name="cifar100_emb_bert_clip", epochs=200):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)

    def init_datasets(self):
        Cifar100.init_datasets(self)


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar100EmbBert(root_path=root_path, epochs=300)

    variant.init_bert_model()
    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(768)
    variant.init_model_helpers(nn.CrossEntropyLoss)
    variant.init_word_lookup()
    variant.train_model()
