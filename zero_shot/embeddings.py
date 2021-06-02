import gensim
from utils import get_device
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class Embedding:
    
    def init_word_vectors(self):
        return NotImplementedError

    def get_word_lookup_table(self):
        return NotImplementedError


class Word2Vec(Embedding):

    def init_word_vectors(self):
        print(f"Initializing word2vec-google-news-300 ...")
        self.word_vectors = gensim.downloader.load(name="word2vec-google-news-300")
        
    def get_word_lookup_table(self,class_names):

        # Note: we store the word lookup in the model, not the dataset because
        #   1.) The word lookup should be on the same device as the model
        #   2.) If using multiple GPUs, the model will get duplicated to each device, but the dataset won't
        #   3.) The word model (i.e., textual feature encoder) is a property of the model not the dataset
        cn = class_names
        for i in range(len(cn)):
            if cn[i] == 'aquarium_fish':
                cn[i] = 'fish'
            if cn[i] == "sweet_pepper":
                cn[i] = 'bell_pepper'
        return torch.from_numpy(
            np.stack([self.word_vectors[_class] for _class in class_names])
        ).to(get_device())


class Bert(Embedding):

    def init_bert_model(self, model="stsb-bert-base"):
        print(f"Initializing {model}...")
        self.transformer_model = SentenceTransformer(model)

    def get_word_lookup_table(self, class_names):
        word_vectors = [
            self.transformer_model.encode(_class) for _class in class_names
        ]
        word_lookup = torch.from_numpy(np.stack(word_vectors)).to(
            get_device()
        )
        word_lookup = word_lookup / word_lookup.norm(
            dim=-1, keepdim=True
        )
        return word_lookup