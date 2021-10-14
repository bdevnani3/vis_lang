import random
from tqdm import tqdm

import argparse
from omegaconf import OmegaConf

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from datasets import *

import json

import torch.nn as nn
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

import clip

clip_model, clip_preprocess = clip.load("ViT-B/32", device)
results_path = "results"
ITERS = 5

prompt = None

def batch(iterable1,iterable2, n=1):
    l = len(iterable1)
    for ndx in range(0, l, n):
        yield (iterable1[ndx:min(ndx + n, l)], iterable2[ndx:min(ndx + n, l)])

def num_correct_preds(outputs, labels):
    _, predicted = outputs.max(1)
    return predicted.eq(labels).sum().item()

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def run_expts(args):

    results = {}
    results["params"] = {}

    dataset = args.data
    dataset_obj = None
    if dataset == "cifar100":
        dataset_obj = Cifar100(args.num_workers, args.batch_size)
    elif dataset == "flowers102":
        dataset_obj = Flowers102(args.num_workers, args.batch_size)
    elif dataset == "oxfordpets":
        dataset_obj = OxfordPets(args.num_workers, args.batch_size)
    elif dataset == "smallflowers102":
        dataset_obj = SmallFlowers102(args.num_workers, args.batch_size)
    elif dataset == "food101":
        dataset_obj = Food101(args.num_workers, args.batch_size)
    results["params"]["data"] = str(args.data)
    results["params"]["batch_size"] = int(args.batch_size)

    assert dataset_obj is not None, "Please select a valid dataset"

    global clip_preprocess

    if args.few_shot_sweep == True:
        class_range = [1, 2, 4, 6, 8, 16, 32, -1]
    else:
        class_range = [-1]

    prompt = None

    with open(args.phrases_file) as f:
        templates = [line for line in f]
        if len(templates) == 1:
            prompt = templates[0]
    
    assert prompt, "Currently there is no logic for ensembling text phrases, please use a file with only one phrase"

    global phrase

    phrase = prompt

    global ITERS
    global results_path

    results = {e.__name__:{c:[] for c in class_range} for e in ALL_EXPTS}
    for n_classes in reversed(class_range):
        for it in range(ITERS):
            train_loader, _ = dataset_obj.get_train_loaders(transform_fn=clip_preprocess)
            train_features, train_labels = get_clip_image_features(train_loader)

            # Normalize training features
            train_features = train_features/train_features.norm(dim=-1, keepdim=True)

            # Name for all the individual classes
            classes = dataset_obj.classes

            test_loader = dataset_obj.get_test_loader(transform_fn=clip_preprocess)

            for expt in ALL_EXPTS:
                if expt.__name__ == random_image_embedding:
                    if it > 0:
                        continue
                print("-------------------------------------")
                print(f"Running {expt.__name__}; iteration {it}; n_classes {n_classes}")
                print("-------------------------------------")
                out = expt(train_features, train_labels, test_loader, classes)
                results[expt.__name__][n_classes].append(out)
                print(out)

    with open(f"{results_path}/{dataset}/raw_embedding_generation_all.json", "w") as outfile:
        json.dump(results, outfile)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Embeddings Experiments")
    parser.add_argument("--config", type=str, default="./configs/dummy.yml")
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)
    results = run_expts(args)
    results["args"] = vars(args)
    