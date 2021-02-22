from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch.nn as nn

from base import get_device
from cifar10_emb_clip import Cifar10EmbClip
from utils import init_root


class Cifar10EmbBert(Cifar10EmbClip):
    def __init__(self, root_path, variant_name="cifar10_emb_bert_clip", epochs=200):
        super().__init__(root_path=root_path, variant_name=variant_name, epochs=epochs)

        # The similarity mode to pass into find_closest_words
        self.similarity_mode = "cossim"

    def init_bert_model(self, model="stsb-bert-base"):
        print(f"Initializing {model}...")
        self.transformer_model = SentenceTransformer(model)

    def init_word_lookup(self):
        word_vectors = [
            self.transformer_model.encode(_class) for _class in self.class_names
        ]
        self.model.word_lookup = torch.from_numpy(np.stack(word_vectors)).to(
            get_device()
        )
        self.model.word_lookup = self.model.word_lookup / self.model.word_lookup.norm(
            dim=-1, keepdim=True
        )


if __name__ == "__main__":

    root_path = init_root()
    variant = Cifar10EmbBert(root_path=root_path)

    variant.init_bert_model()
    variant.init_datasets()
    variant.init_dataloaders()
    variant.set_up_model_architecture(768)
    variant.init_model_helpers(nn.CrossEntropyLoss)
    variant.init_word_lookup()
    variant.train_model()
