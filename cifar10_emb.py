import gensim.downloader
import torch
import numpy as np
import torch.nn as nn

from base import get_device
from cifar10 import Cifar10
from utils import init_root


class Cifar10Emb(Cifar10):
    def __init__(self, root_path, variant_name="cifar10_emb", epochs=200):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)

        # The similarity mode to pass into find_closest_words
        self.similarity_mode = "l2"

    def find_closest_words(
        self, word_lookup: torch.Tensor, x: torch.Tensor, mode: str = "l2"
    ) -> torch.Tensor:
        """
        Given a size [N, c] lookup table (N classes, c channels per vector) and a set of [M, c] vectors to look up,
        returns a size [M] vector of indices from 0 to N-1 containing the closest vector in the lookup for that input.

        Modes:
            l2     - Computes pairwise L2 distance and chooses the lowest one.
            cossim - Computes pairwise cosine similarity, and chooses the most similar.
            dot    - Computes pairwise dot product similarity, and choses the most similar
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
        elif mode == "cossim":
            # Note: we don't need to divide by the length of x here, because it's the same for the argmax.
            # Also, it's imporant that we can get away with that for numerical stability.
            return ((x @ word_lookup.t()) / word_lookup.norm(dim=-1)[None, :]).argmax(
                dim=-1
            )
        elif mode == "dot":
            return (x @ word_lookup.t()).argmax(dim=-1)
        else:
            raise NotImplementedError

    def init_word_vectors(self):
        self.word_vectors = gensim.downloader.load(name="word2vec-google-news-300")

    def init_word_lookup(self):

        # Note: we store the word lookup in the model, not the dataset because
        #   1.) The word lookup should be on the same device as the model
        #   2.) If using multiple GPUs, the model will get duplicated to each device, but the dataset won't
        #   3.) The word model (i.e., textual feature encoder) is a property of the model not the dataset
        self.model.word_lookup = torch.from_numpy(
            np.stack([self.word_vectors[_class] for _class in self.class_names])
        ).to(get_device())

    def num_correct_preds(self, outputs, labels):
        return (
            (
                self.find_closest_words(
                    self.model.word_lookup, outputs, mode=self.similarity_mode
                )
                == labels
            )
            .sum()
            .item()
        )

    def calc_loss(self, outputs, labels):
        return self.criterion(outputs, self.model.word_lookup[labels])


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar10Emb(root_path=root_path)

    variant.init_word_vectors()
    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.MSELoss)
    variant.init_word_lookup()
    variant.train_model()
