# Data efficieny experiments

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

results_path = "results"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = None, None
resnet_model = None

def clip_zero_shot(
    loader,
    classes,
    clip_model_name="ViT-B/32",
    phrase_file="configs/phrases/default.txt",
):

    global clip_model, clip_preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [
                    template.format(classname) for template in templates
                ]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = clip_model.encode_text(
                    texts
                )  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

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

    templates = []
    with open(phrase_file) as f:
        templates = [line for line in f]

    zeroshot_weights = zeroshot_classifier(classes, templates)

    with torch.no_grad():
        top1, top5, n = 0.0, 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(loader)):
            images = images.cuda()
            target = target.cuda()

            # predict
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ zeroshot_weights

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n) * 100
    top5 = (top5 / n) * 100

    return top1, top5


def get_clip_features(dataset):
    all_features = []
    all_labels = []

    global clip_model

    with torch.no_grad():
        for images, labels in tqdm(dataset):
            features = clip_model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def clip_linear_probe(
    train_dataset, test_dataset, classes, clip_model_name="ViT-B/32", C=1, max_iter=1000
):

    global clip_model, clip_preprocess

    train_features, train_labels = get_clip_features(train_dataset)
    test_features, test_labels = get_clip_features(test_dataset)

    classifier = LogisticRegression(C=C, max_iter=max_iter, n_jobs=6)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.0

    return accuracy


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

    final = {}
    test_loader = dataset_obj.get_test_loader(transform_fn=clip_preprocess)
    for i in [1,2,4,8,16,32, -1]:
        try:
            tot = []
            print("-------------------------------------")
            print(f"Running Clip Linear Probe on {dataset_obj.name} with {i} element per class")
            print("-------------------------------------")
            for _ in range(3):
                train_loader, _ = dataset_obj.get_train_loaders(transform_fn=clip_preprocess, num_elements_per_class=i)
                clp = clip_linear_probe(train_loader, test_loader, dataset_obj.classes)
                tot.append(clp/100)
            final[i] = tot
            print(f"Clip Linear Probe Acc: {final[i]}")
        except:
            continue
    print(final)

    with open(f"{results_path}/{dataset}_n_classes_expt.json", "w") as outfile:
        json.dump(final, outfile)

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clip Experiments")
    parser.add_argument("--config", type=str, default="./configs/dummy.yml")
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)
    run_expts(args)