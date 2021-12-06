import argparse
import clip
import json
import numpy as np
import torch

from tqdm import tqdm

from datasets2 import *

"""
Splits the train data into two parts, one to generate prototypes and other to evaluate on.
"""

# Global Variables
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device)
# phrase = "This is a photo of a {}."

# Helpers for embedding generation


def get_clip_text_features(classes):
    """Given a list of classes, generates clip embeddings per class"""
    # Assumes the positions are in accordance to the label numbers
    embedding_per_class = {}

    global clip_model
    global phrase

    with torch.no_grad():
        for i, _class in enumerate(classes):
            _class = _class.replace("_", " ")
            text = clip.tokenize(phrase.format(_class)).cuda()
            class_embeddings = clip_model.encode_text(text)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            embedding_per_class[i] = class_embeddings
    return embedding_per_class


def get_clip_image_features(data_loader):
    """Given a dataloader object, generate two torch arrays of encoded images and corresponding labels"""
    all_features = []
    all_labels = []

    global clip_model

    with torch.no_grad():
        for images, labels in data_loader:
            features = clip_model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = torch.nn.Parameter(torch.tensor([0.0]).cuda())
        self.sig = nn.Sigmoid()
        self.sig_lam = self.sig(self.lam)

    def forward(self, image_prototype, text_prototpe, images):
        sig_lam = self.sig(self.lam)
        self.sig_lam = sig_lam
        dot = sig_lam * (images @ image_prototype.T) + (1 - sig_lam) * (
            images @ text_prototpe.T
        )
        return dot

    def string(self):
        return f"Lambda: {self.sig(self.lam).item():7.3f}"


def generate_prototypes_per_model(train_features, train_labels, classes):

    text_clip_features = get_clip_text_features(classes)
    text_embeds = []
    image_embeds = []

    for c in range(len(classes)):
        ind = (c == train_labels).nonzero(as_tuple=True)[0]
        image_embs = torch.mean(train_features[ind], dim=0)
        text_embs = text_clip_features[c].squeeze()
        text_embeds.append(text_embs)
        image_embeds.append(image_embs)

    image_embeds = torch.stack(image_embeds).squeeze(1)
    text_embeds = torch.stack(text_embeds).squeeze(1)

    return image_embeds, text_embeds


# pass only test or valid loader here
def evaluate(data_loader, image_embeds, text_embeds, lam):
    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            dot = lam * (image_features @ image_embeds.T) + (1 - lam) * (
                image_features @ text_embeds.T
            )
            top1 += torch.sum(torch.argmax(dot, axis=1) == target).item()
            n += len(target)
    return top1 / n


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process inputs for finetuning expt.")
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--dataset", type=str, default="cifar100")
    flags = parser.parse_args()

    num_epochs = flags.num_epochs

    batch_size= 50
    num_workers = 12

    global phrase

    # initialize dataset
    if flags.dataset == "cifar100":
        dataset_obj = Cifar100(num_workers, batch_size)
        phrase = "This is a photo of a {}."
    elif flags.dataset == "flowers102":
        dataset_obj = Flowers102(num_workers, batch_size)
        phrase = "This is a {} with petals."
    elif flags.dataset == "oxfordpets":
        dataset_obj = OxfordPets(num_workers, batch_size)
        phrase = "This is a photo of a {}."
    elif flags.dataset == "food101":
        phrase = "This is a photo of a {}, a food item."
        dataset_obj = Food101(num_workers, batch_size)

    shots = [1, 2, 4, 8, 16]
    lams = np.linspace(0.0, 1.0, 20)

    data_dict = {}

    for shot in shots:
        data_dict[shot] = {"best_lambda": -1, "best_lambda_test_accuracy":-1}
        for iter in range(flags.num_iters):
            data_dict[shot][iter] = {
                "train_accuracy": {},
                "train_loss": {},
                "lambdas": {},
                "valid_accuracy": {},
                "test_accuracy": {},
                "manually_found_lambdas": {},
            }

    for iter in range(flags.num_iters):

        for shot in shots[::-1]:
            best_lambda = -1
            best_acc = -1
            print(
                    f"############### Iter {iter} || Shot {shot} ###############"
                )

            if shot > 8 and flags.dataset == "flowers102":
                continue

            train_loader, valid_loader = dataset_obj.get_train_loaders(
                transform_fn=clip_preprocess, num_elements_per_class=shot
            )
            test_loader = dataset_obj.get_test_loader(transform_fn=clip_preprocess)
            classes = dataset_obj.classes

            train_features_unnorm, train_labels = get_clip_image_features(train_loader)
            train_features = train_features_unnorm / train_features_unnorm.norm(
                dim=-1, keepdim=True
            )

            image_embeds, text_embeds = generate_prototypes_per_model(
                train_features, train_labels, classes
            )

            # Truth

            # best_lam = -1
            # best_acc = -1
            # for l in lams:
            #     acc = evaluate(test_loader, image_embeds, text_embeds, torch.tensor([l]).cuda())
            #     data_dict[shot][iter]["manually_found_lambdas"][l] = acc
            #     print(l,acc)
            #     if acc > best_acc:
            #         best_acc = acc
            #         best_lam = l

            # print(f"True best Lambda @ {best_lam}")

            # Model parameters
            model = Lambda()
            criterion = nn.CrossEntropyLoss(reduction="sum")
            learning_rate = 0.5
            optimizer = torch.optim.SGD(
                model.parameters(), lr=learning_rate, weight_decay=1e-5
            )
            num_epochs = 500

            # Train

            for epoch in range(num_epochs):

                total_loss = 0.0
                correct = 0.0
                total = 0


                for i, (images, target) in enumerate(valid_loader):
                    model.train()
                    images = images.cuda()
                    target = target.cuda()
                    image_features_unnorm = clip_model.encode_image(images)
                    image_features = image_features_unnorm / image_features_unnorm.norm(
                        dim=-1, keepdim=True
                    )

                    optimizer.zero_grad()
                    outputs = model(image_embeds, text_embeds, image_features)
                    loss = criterion(outputs, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total += len(target)
                    correct += torch.sum(torch.argmax(outputs, axis=1) == target).item()

                epoch_acc = correct/total
                epoch_loss = loss/total
                data_dict[shot][iter]["train_accuracy"][epoch] = epoch_acc
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_lambda = model.sig_lam.item()
                    data_dict[shot]["best_lambda"] = best_lambda
                    model.eval()
                    test_acc = evaluate(
                        test_loader, image_embeds, text_embeds, model.sig_lam.item()
                    )
                    data_dict[shot]["best_lambda_test_accuracy"] = test_acc
                data_dict[shot][iter]["train_loss"][epoch] = total_loss
                data_dict[shot][iter]["lambdas"][epoch] = model.sig_lam.item()

                print(
                    f"Epoch {epoch} || Loss: {total_loss/total} || Accuracy: {correct/total} || {model.string()} || Best Lambda: {best_lambda}"
                )

                # if epoch % 10 == 0:
                    
                #     model.eval()
                #     valid_acc = evaluate(
                #         valid_loader, image_embeds, text_embeds, model.sig_lam.item()
                #     )
                #     print("Valid Acc: ", valid_acc, "Lam:", model.sig_lam.item())
                #     data_dict[shot][iter]["valid_accuracy"][epoch] = valid_acc

                if epoch % 20 == 0:
                    
                    model.eval()
                    test_acc = evaluate(
                        test_loader, image_embeds, text_embeds, model.sig_lam.item()
                    )
                    print("Test Acc: ", test_acc, "Lam:", model.sig_lam.item())
                    data_dict[shot][iter]["test_accuracy"][epoch] = test_acc
        

            with open(f"/nethome/bdevnani3/vis_lang/efficient_finetuning/prompt_engineer/experiments/results/finetune_lambda_split_{flags.dataset}.json", "w") as outfile:
                json.dump(data_dict, outfile)        

            
