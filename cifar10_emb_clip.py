import gensim.downloader
import torch
import numpy as np
import torch.nn as nn

from base import get_device
from cifar10_emb import Cifar10Emb
from utils import init_root, get_device


class Cifar10EmbClip(Cifar10Emb):
    def __init__(self, root_path, variant_name="cifar10_emb_clip", epochs=200):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)

        # The similarity mode to pass into find_closest_words
        self.similarity_mode = "cossim"

    def set_up_model_architecture(self, num_features_in_last_layer: int):
        super().set_up_model_architecture(num_features_in_last_layer)

        # CLIP has this so we do too
        self.model.logit_scale = nn.Parameter(torch.ones([], device=get_device()))

    def init_word_lookup(self):
        super().init_word_lookup()

        self.model.word_lookup = self.model.word_lookup / self.model.word_lookup.norm(
            dim=-1, keepdim=True
        )

    def calc_loss(self, outputs, labels):
        """ Most of this is copied line-for-line from https://github.com/openai/CLIP/blob/main/clip/model.py#L353 """

        image_features = outputs
        text_features = self.model.word_lookup

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Note: I prenormalize the text features in init
        # text_features  = text_features  / text_features .norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return self.criterion(logits_per_image, labels)


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar10EmbClip(root_path=root_path)

    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(300)
    variant.init_model_helpers(nn.CrossEntropyLoss)
    variant.init_word_lookup()
    variant.train_model()
