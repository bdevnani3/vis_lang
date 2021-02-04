import gensim.downloader
import torch
import numpy as np
import torch.nn as nn

from base import get_device
from cifar10 import Cifar10


class Cifar10Emb(Cifar10):
    def __init__(self, root_path, variant_name="cifar10_emb", epochs=200):
        super(Cifar10Emb, self).__init__(
            root_path=root_path, variant_name=variant_name, epochs=epochs
        )

    def find_closest_words(
        self, word_lookup: torch.Tensor, x: torch.Tensor, mode: str = "l2"
    ) -> torch.Tensor:
        """
        Given a size [N, c] lookup table (N classes, c channels per vector) and a set of [M, c] vectors to look up,
        returns a size [M] vector of indices from 0 to N-1 containing the closest vector in the lookup for that input.

        Modes:
            l2     - Computes pairwise L2 distance and chooses the lowest one.
            cossim - Computs pairwise cosine similarity, and chooses the most similar. (Not implemented)
        """
        N, c = word_lookup.shape
        M, c2 = x.shape

        assert (
            c == c2
        ), "The lookup should have the same number of channels as the input."

        if mode == "l2":
            return (
                ((word_lookup[None, :, :] - x[:, None, :]) ** 2)
                .sum(dim=-1)
                .argmin(dim=-1)
            )
        else:
            raise NotImplementedError

    def init_word_lookup(self):
        # We only need to lazily initialize this once. Don't reinitialize it if it's already been initialized.
        word_vectors = gensim.downloader.load(name="word2vec-google-news-300")

        # Note: we store the word lookup in the model, not the datset because
        #   1.) The word lookup should be on the same device as the model
        #   2.) If using multiple GPUs, the model will get duplicated to each device, but the dataset won't
        #   3.) The word model (i.e., textual feature encoder) is a property of the model not the dataset
        self.model.word_lookup = torch.from_numpy(
            np.stack([word_vectors[_class] for _class in self.class_names])
        ).to(get_device())

    def num_correct_preds(self, outputs, labels):
        return (
            (self.find_closest_words(self.model.word_lookup, outputs) == labels)
            .sum()
            .item()
        )

    def calc_loss(self, outputs, labels):
        return self.criterion(outputs, self.model.word_lookup[labels])


if __name__ == "__main__":

    root_path = "/nethome/bdevnani3/raid"
    variant = Cifar10Emb(root_path=root_path)

    variant.load_data()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
