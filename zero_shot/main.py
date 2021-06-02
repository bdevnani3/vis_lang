import argparse
from omegaconf import OmegaConf
from dataloaders import *
import utils
from base import *
import torch.nn as nn
from embeddings import *
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

def main(args, train=True):
    root = utils.init_root()

    expt = Base(
        root_path=root, 
        project_path="zero_shot", 
        variant_name=args.experiment_name, 
        epochs=args.epochs,
        loss_name=args.loss)

    # Load the data
    if args.data == "cifar10":
        dataset = Cifar10(root=root)
    elif args.data == "cifar100":
        dataset = Cifar100(root=root)
        
    dataset.init_dataloader()
    expt.train_loader = dataset.train_loader
    expt.test_loader = dataset.test_loader
    expt.class_names = dataset.class_names

    # Set up model architecture
    if args.embedding == "default":
        expt.set_up_model_architecture(10)
    elif args.embedding == "w2v":
        expt.set_up_model_architecture(300)
    elif args.embedding == "bert":
        expt.set_up_model_architecture(768)

    # Initialize model parameters
    if args.criterion == "ce":
        expt.init_model_helpers(nn.CrossEntropyLoss)
    elif args.criterion == "mse":
        expt.init_model_helpers(nn.MSELoss)
    elif args.criterion == "cosine_loss":
        expt.init_model_helpers()
        expt.criterion = cosine_loss
    elif args.criterion == "cossim":
        expt.init_model_helpers(nn.CosineSimilarity)

    # Initialize embedding parameters
    if args.embedding == "w2v":
        emb = Word2Vec()
        emb.init_word_vectors()
        expt.model.word_lookup = emb.get_word_lookup_table(expt.class_names)
    elif args.embedding == "bert":
        emb = Bert()
        emb.init_bert_model()
        expt.model.word_lookup = emb.get_word_lookup_table(expt.class_names)

    # Initializations for embedding losses
    if args.loss == "clip":
        expt.similarity_mode = "cossim"
        expt.model.logit_scale = nn.Parameter(torch.ones([], device=get_device()))
        expt.model.word_lookup = expt.model.word_lookup / expt.model.word_lookup.norm(
            dim=-1, keepdim=True
        )
    elif args.loss == "dot":
        expt.similarity_mode = "dot"
    elif args.loss == "cosine":
        expt.similarity_mode = "cossim"
    elif args.loss == "fuzzy_mse" or args.loss == "default":
        expt.similarity_mode = "l2"

    if train:
        expt.train_model()
    
    return expt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
    parser.add_argument('--config',  type=str, default='./configs/config_cifar100_experimental.yml')
    flags =  parser.parse_args()
    args = OmegaConf.load(flags.config)
    main(args)