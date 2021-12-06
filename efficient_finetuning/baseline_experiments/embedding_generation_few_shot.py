import os
import clip
import torch
from torchvision import transforms, models

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
from omegaconf import OmegaConf

import json

from datasets import *
from clip_model_comparison import *

results_path = "results"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

clip_model, clip_preprocess = clip.load("ViT-B/32", device)

import torch.nn as nn
import torch.optim as optim

def get_clip_features(dataset):
    all_features = []
    all_labels = []

    global clip_model

    with torch.no_grad():
        for images, labels in tqdm(dataset):
            features = clip_model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

###### Generating weights ##########

def cross_entropy(train_loader, test_loader, classes):
    len_classes = len(classes)

    train_features, train_labels = get_clip_features(train_loader)
    test_features, test_labels = get_clip_features(test_loader)

    classifier = LogisticRegression(C=1, max_iter=1000, n_jobs=6)
    classifier.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())
    zeroshot_weights = torch.from_numpy(classifier.coef_.T).to(torch.float16)
    return zeroshot_weights

def clip_loss(train_loader, test_loader, classes):
    len_classes = len(classes)

    train_features, train_labels = get_clip_features(train_loader)
    test_features, test_labels = get_clip_features(test_loader)

    train_features = train_features / train_features.norm(dim=-1, keepdim=True)
    test_features = test_features / test_features.norm(dim=-1, keepdim=True)

    classifier = LogisticRegression(C=1, max_iter=1000, n_jobs=6)
    classifier.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())
    zeroshot_weights = torch.from_numpy(classifier.coef_.T).to(torch.float16)
    return zeroshot_weights

# def hybrid_loss(train_loader, test_loader):
#     """Concatenating text and image encoding, basically using a linear combination""""
#     len_classes = len(classes)

#     train_features, train_labels = get_clip_features(train_loader)
#     test_features, test_labels = get_clip_features(test_loader)

#     train_features = train_features / train_features.norm(dim=-1, keepdim=True)
#     test_features = test_features / test_features.norm(dim=-1, keepdim=True)

#     classifier = LogisticRegression(C=1, max_iter=1000, n_jobs=6)
#     zeroshot_weights = torch.from_numpy(classifier.coef_.T).to(torch.float16)
#     return zeroshot_weights

###################

def final_accuracy(test_loader, zeroshot_weights):

    global clip_model

    def accuracy(output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [
            float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
            for k in topk
        ]

    # lazy load
    if clip_model == None:
        clip_model, clip_preprocess = clip.load(clip_model_name, device)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features.to(device) @ zeroshot_weights.to(device)

            # measure accuracy
            acc1, _ = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            n += images.size(0)

    top1 = (top1 / n) * 100
    return top1

def run_expts(args):
    """
    Currently supports clip_zero_shot, clip_linear_probe, resnet_linear_probe expts
    """

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

    global clip_model, clip_preprocess

    if clip_preprocess is None or clip_model is None:
        clip_model, clip_preprocess = clip.load(args.clip_model, device)

    final = {"cross_entropy": {}, "clip": {}}
    for i in [1,2,4,8,16,32,-1]:
        tot_ce = 0
        print("-------------------------------------")
        print(f"Running Prompt Generation expts on {dataset_obj.name} with {i} elements per class")
        print("-------------------------------------")
        test_loader = dataset_obj.get_test_loader(transform_fn=clip_preprocess)
        try:
            for _ in range(5):
                train_loader, _ = dataset_obj.get_train_loaders(transform_fn=clip_preprocess, num_elements_per_class=i)
                ce = cross_entropy(train_loader,test_loader,dataset_obj.classes)
                out = final_accuracy(test_loader,ce)
                if i in final["cross_entropy"]:
                    final["cross_entropy"][i] +=out
                else:
                    final["cross_entropy"][i] =out

                ce = clip_loss(train_loader,test_loader,dataset_obj.classes)
                out = final_accuracy(test_loader,ce)
                if i in final["clip"]:
                    final["clip"][i] +=out
                else:
                    final["clip"][i] =out


                # h = hybrid_loss(train_loader,test_loader)
                # final_accuracy = final_accuracy(test_loader,h)
                # if i in final["hybrid"]:
                #     final["hybrid"][i] +=final_accuracy
                # else:
                #     final["hybrid"][i] =0

            final["cross_entropy"][i]/=5
            final["clip"][i]/=5
            print(final)
            # final["hybrid"][i]=/10
        except Error:
            pass

    with open(f"{results_path}/{dataset}_embedding_generation_few_shot.json", "w") as outfile:
        json.dump(final, outfile)

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip Experiments")
    parser.add_argument("--config", type=str, default="./configs/dummy.yml")
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)
    run_expts(args)