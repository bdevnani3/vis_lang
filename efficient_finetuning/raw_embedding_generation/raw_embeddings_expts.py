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

from sklearn.linear_model import LogisticRegression
from datasets import *

import json

from scipy.special import expit


# torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

import clip

clip_model, clip_preprocess = clip.load("ViT-B/32", device)
results_path = "results"
ITERS = 3

prompt = None

def get_clip_image_features(data_loader):
    """Given a dataloader object, generate two torch arrays of encoded images and corresponding labels"""
    all_features = []
    all_labels = []

    global clip_model

    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            features = clip_model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)

def get_clip_text_features(classes):
    """Given a dataloader object, generate two torch arrays of encoded images and corresponding labels"""
    # Assumes the positions are in accordance to the label numbers
    embedding_per_class = {}
    
    global clip_model

    global phrase

    with torch.no_grad():
        for i,_class in enumerate(classes):
            _class = _class.replace("_", " ")
            text = clip.tokenize(phrase.format(_class)).cuda() 
            class_embeddings = clip_model.encode_text(
                    text
                )
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            embedding_per_class[i] = class_embeddings
    return embedding_per_class

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

def get_stats(l):
    return np.mean(l), np.var(l)

################################################
####### Embedding Calculation Variations #######
################################################


# Random Image Embedding
def random_image_embedding(train_features, train_labels, test_loader, classes):

    train_features = train_features/train_features.norm(dim=-1, keepdim=True)

    embeddings = []

    global clip_model

    for c in range(len(classes)):
        # Choose any random index
        ind = random.choice(train_labels)
        embeddings.append(train_features[ind].cpu().numpy())

    S = torch.from_numpy(np.array(embeddings).T).to(torch.float16)

    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            # Normalize features at test time
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features.to(device) @ S.to(device)

            # measure accuracy
            acc1 = accuracy(logits, target, topk=(1,))
            top1 += acc1[0]
            n += images.size(0)
    return top1/n

# Random Image Embedding from a specific class
def random_image_embedding_from_class(train_features, train_labels, test_loader, classes):

    train_features = train_features/train_features.norm(dim=-1, keepdim=True)

    embeddings = []

    global clip_model

    for c in range(len(classes)):
        # Choose any random index
        ind = random.choice((c == train_labels).nonzero(as_tuple=False).tolist())
        embeddings.append(train_features[ind].cpu().numpy())

    S = torch.from_numpy(np.array(embeddings).squeeze(1).T).to(torch.float16)

    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            # Normalize features at test time
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features.to(device) @ S.to(device)

            # measure accuracy
            acc1 = accuracy(logits, target, topk=(1,))
            top1 += acc1[0]
            n += images.size(0)
    return top1/n


# Nearest Image Embedding
def nearest_image_embedding(train_features, train_labels, test_loader, classes):

    train_features = train_features/train_features.norm(dim=-1, keepdim=True)

    embeddings = []

    global clip_model

    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, targets) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            targets = targets.cuda()
            image_features = clip_model.encode_image(images)
            # Normalize features at test time
            image_features /= image_features.norm(dim=-1, keepdim=True)

            for image_feature, target in zip(image_features,targets):

                # Dot product image feature with train features, find the maximum dot product(closest vector)
                out = (image_feature @ train_features.t()).argmax(dim=-1)

                closest = out.cpu().numpy()
                prediction = train_labels[closest].cpu().numpy()

                if target.cpu().numpy() == prediction:
                    top1 +=1
                n+=1
    return top1/n

# def image_and_text_embedding(train_features, train_labels, test_loader, classes, a, b):
#     # a -> importance given to the image embedding
#     # b -> importance given to the text embedding

#     # Currently, this version takes an average of Clip-Zero shot & average image embedding

#     text_clip_features = get_clip_text_features(classes)

#     text_embeds = []
#     image_embeds = []
#     embeddings = []
#     for c in range(len(classes)):
#         ind = (c == train_labels).nonzero(as_tuple=True)[0]

#         # replace this with linear probe
#         image_embs = torch.mean(train_features[ind],dim=0)

#         text_embs = text_clip_features[c].squeeze()
#         text_embeds.append(text_embs)
#         image_embeds.append(image_embs)
#         avg_emb = (a*image_embs + b*text_embs)
#         embeddings.append(avg_emb)
    
#     zeroshot_weights = torch.stack(embeddings).squeeze(1)
#     image_embeds = torch.stack(image_embeds).squeeze(1)
#     image_embeds = torch.from_numpy(get_clip_linear_probe_weights(train_features, train_labels)).to(torch.float16).cuda()
#     print(image_embeds.shape)
#     text_embeds = torch.stack(text_embeds).squeeze(1)

#     with torch.no_grad():
#         top1, n = 0.0, 0.0
#         for i, (images, target) in enumerate(tqdm(test_loader)):
#             images = images.cuda()
#             target = target.cuda()
#             image_features = clip_model.encode_image(images)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             # predict
#             for image_feature,tar in zip(image_features, target):
#                 texts_per_img = []
#                 imgs_per_img = []
#                 for c in range(len(classes)):
#                     text_embs = text_clip_features[c].squeeze()
#                     texts_per_img.append(text_embs)
#                     imgs_per_img.append(image_feature)
#                 texts_per_img = torch.stack(texts_per_img)
#                 imgs_per_img = torch.stack(imgs_per_img)
#                 dot = a*(imgs_per_img @ image_embeds.T) + b*(imgs_per_img @ text_embeds.T) + 0*(texts_per_img @ text_embeds.T) + 0*(texts_per_img @ image_embeds.T)
#                 out = np.unravel_index(torch.argmax(dot, axis=None).cpu().numpy(), dot.shape)[1]
#                 if tar.cpu().numpy() == out:
#                     top1 +=1
#                 n+=1

#     return top1/n

def image_and_text_embedding(train_features, train_labels, test_loader, classes, a, b):
    # a -> importance given to the image embedding
    # b -> importance given to the text embedding

    # Currently, this version takes an average of Clip-Zero shot & Linear probe

    text_clip_features = get_clip_text_features(classes)

    text_embeds = []
    for c in range(len(classes)):
        text_embs = text_clip_features[c].squeeze()
        text_embeds.append(text_embs)

    text_embeds = torch.stack(text_embeds).squeeze(1)
    classifier = get_clip_linear_probe_classifier(train_features, train_labels)
    sftmx = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            image_features_norm = image_features/image_features.norm(dim=-1, keepdim=True)
            # predict
            for image_feature, image_feature_norm ,tar in zip(image_features, image_features_norm, target):
                texts_per_img = []
                imgs_per_img = []
                imgs_n_per_img = []
                for c in range(len(classes)):
                    text_embs = text_clip_features[c].squeeze()
                    texts_per_img.append(text_embs)
                    imgs_per_img.append(image_feature)
                    imgs_n_per_img.append(image_feature_norm)
                texts_per_img = torch.stack(texts_per_img)
                imgs_per_img = torch.stack(imgs_per_img)
                imgs_n_per_img = torch.stack(imgs_n_per_img)
                linear_probe = torch.from_numpy(classifier.predict_proba(imgs_per_img.cpu().numpy())).to(torch.float16).cuda()
                dot = a*(linear_probe) + b*sftmx((imgs_n_per_img @ text_embeds.T)*100) 
                out = np.unravel_index(torch.argmax(dot, axis=None).cpu().numpy(), dot.shape)[1]
                if tar.cpu().numpy() == out:
                    top1 +=1
                n+=1

    return top1/n


# Average Image Embedding
def average_image_embedding(train_features, train_labels, test_loader, classes):
    train_features = train_features/train_features.norm(dim=-1, keepdim=True)

    a, b = 1, 0
    text_clip_features = get_clip_text_features(classes)

    text_embeds = []
    image_embeds = []

    for c in range(len(classes)):
        ind = (c == train_labels).nonzero(as_tuple=True)[0]

        # replace this with linear probe
        image_embs = torch.mean(train_features[ind],dim=0)

        text_embs = text_clip_features[c].squeeze()
        text_embeds.append(text_embs)
        image_embeds.append(image_embs)

    image_embeds = torch.stack(image_embeds).squeeze(1)
    text_embeds = torch.stack(text_embeds).squeeze(1)

    with torch.no_grad():
        top1, n = 0.0, 0.0
        for i, (images, target) in enumerate(tqdm(test_loader)):
            images = images.cuda()
            target = target.cuda()
            image_features = clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # predict
            for image_feature,tar in zip(image_features, target):
                texts_per_img = []
                imgs_per_img = []
                for c in range(len(classes)):
                    text_embs = text_clip_features[c].squeeze()
                    texts_per_img.append(text_embs)
                    imgs_per_img.append(image_feature)
                texts_per_img = torch.stack(texts_per_img)
                imgs_per_img = torch.stack(imgs_per_img)
                dot = a*(imgs_per_img @ image_embeds.T) + b*(imgs_per_img @ text_embeds.T) + 0*(texts_per_img @ text_embeds.T) + 0*(texts_per_img @ image_embeds.T)
                out = np.unravel_index(torch.argmax(dot, axis=None).cpu().numpy(), dot.shape)[1]
                if tar.cpu().numpy() == out:
                    top1 +=1
                n+=1

    return top1/n

# Weighted Image + Text Embedding
def weighted_image_and_text_embedding(train_features, train_labels, test_loader, classes):
    lams = np.linspace(0.0, 1.0, 20)
    out = {l:-1 for l in lams}
    for lam in lams:
        print("---------- Lam:", lam, "----------")
        res = image_and_text_embedding(train_features, train_labels, test_loader, classes, a=lam,b=1-lam)
        out[lam] = res
    return out 


# Average Image + Text Embedding (Weighted with 0.5 as both the weights)
def weighted_image_and_text_embedding_0_5(train_features, train_labels, test_loader, classes):
    return image_and_text_embedding(train_features, train_labels, test_loader, classes, a=0.5,b=0.5)


# Text Embedding (Clip Zero Shot)
def text_embedding(train_features, train_labels, test_loader, classes):
    return image_and_text_embedding(train_features, train_labels, test_loader, classes,a=0, b=1)


def clip_linear_probe(
    train_dataset, test_dataset, C=1
):

    global clip_model, clip_preprocess

    train_features, train_labels = get_clip_image_features(train_dataset)
    test_features, test_labels = get_clip_image_features(test_dataset)
    # train_features /= train_features.norm(dim=-1, keepdim=True)
    # test_features /= test_features.norm(dim=-1, keepdim=True)

    classifier = LogisticRegression(C=C, max_iter=1000, n_jobs=6)
    classifier.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())
    # predictions_proba = classifier.predict_proba(test_features.cpu().numpy())
    test_labels = test_labels.cpu().numpy()

    # out = np.argmax(predictions_proba, axis=1)
    # accuracy1 = np.mean((test_labels == out).astype(np.float)) * 100.0
    # print(accuracy1)

    predictions = classifier.predict(test_features.cpu().numpy())
    accuracy2 = np.mean((test_labels == predictions).astype(np.float))
    print(accuracy2)

    return accuracy2


def get_clip_linear_probe_classifier(
    train_features, train_labels, C=1
):

    classifier = LogisticRegression(C=C, max_iter=1000, n_jobs=6)
    classifier.fit(train_features.cpu().numpy(), train_labels.cpu().numpy())

    return classifier

###################################
####### Experiment Executer #######
###################################


# Please update when we write a new experiment
ALL_EXPTS = [ , random_image_embedding_from_class, nearest_image_embedding, average_image_embedding, weighted_image_and_text_embedding, weighted_image_and_text_embedding_0_5, clip_linear_probe]
# ALL_EXPTS = [average_image_embedding, text_embedding, clip_linear_probe, weighted_image_and_text_embedding_0_5, weighted_image_and_text_embedding, nearest_image_embedding]
# ALL_EXPTS = [weighted_image_and_text_embedding]

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
        class_range = [1, 2, 4, 8, 16, 32]
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

    test_loader = dataset_obj.get_test_loader(transform_fn=clip_preprocess)

    for n_classes in reversed(class_range):
        if n_classes == -1:
            itrs = 1
        else:
            itrs = 5
        for it in range(itrs):
            try:
                train_loader, _ = dataset_obj.get_train_loaders(transform_fn=clip_preprocess,num_elements_per_class=n_classes)
                train_features, train_labels = get_clip_image_features(train_loader)
            except:
                continue

            # Normalize training features
            # train_features = train_features/train_features.norm(dim=-1, keepdim=True)

            # Name for all the individual classes
            classes = dataset_obj.classes

            for expt in ALL_EXPTS:
                if expt.__name__ == "random_image_embedding":
                    if it > 0:
                        continue
                print("-------------------------------------")
                print(f"Running {expt.__name__}; iteration {it}; n_classes {n_classes}")
                print("-------------------------------------")
                if expt.__name__ == "clip_linear_probe":
                    out = clip_linear_probe(train_loader, test_loader)
                else:
                    out = expt(train_features, train_labels, test_loader, classes)
                results[expt.__name__][n_classes].append(out)
                print(out)


    with open(f"{results_path}/{dataset}/all.json", "w") as outfile:
        json.dump(results, outfile)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raw Embeddings Experiments")
    parser.add_argument("--config", type=str, default="./configs/dummy.yml")
    flags = parser.parse_args()
    args = OmegaConf.load(flags.config)
    results = run_expts(args)
    results["args"] = vars(args)
    