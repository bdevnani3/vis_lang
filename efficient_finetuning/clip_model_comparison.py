
import os
import clip
import torch
from torchvision.datasets import CIFAR100
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

import json

results_path="results"
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = None, None
resnet_model = None


def clip_zero_shot(dataset, classes, phrase="a photo of a {}"):

    global clip_model, clip_preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # lazy load
    if clip_model == None:
        clip_model, clip_preprocess = clip.load('ViT-B/32', device)

    text_inputs = torch.cat([clip.tokenize(phrase.format(c)) for c in dataset.classes]).to(device)

    top_1_accuracy = 0
    top_5_accuracy = 0

    len_classes = len(classes)

    per_class_accuracy_top1 = { k:[0,0, dataset.classes[k]] for k in range(len_classes)} # correct, total, class_name
    per_class_accuracy_top5 = { k:[0,0, dataset.classes[k]] for k in range(len_classes)} 

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    for i in tqdm(range(len(dataset))):

        # Prepare the inputs
        image, class_id = dataset[i]
        image_input = clip_preprocess(image).unsqueeze(0).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)

        if indices[0] == class_id:
            top_1_accuracy +=1
            per_class_accuracy_top1[class_id][0] +=1

        if class_id in indices:
            top_5_accuracy +=1
            per_class_accuracy_top5[class_id][0] +=1

        per_class_accuracy_top1[class_id][1] +=1
        per_class_accuracy_top5[class_id][1] +=1

    return top_1_accuracy/float(len(dataset)), top_5_accuracy/float(len(dataset)), per_class_accuracy_top1, per_class_accuracy_top5


def get_resnet_features(dataset):
    all_features = []
    all_labels = []

    global resnet_model

    for inps, labels in tqdm(DataLoader(dataset, batch_size=100, num_workers=4)):
        with torch.no_grad():
            features = resnet_model(inps)

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def get_clip_features(dataset):
    all_features = []
    all_labels = []
    
    global clip_model

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100, num_workers=4)):
            features = clip_model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def resnet_linear_probe(train_dataset, test_dataset):
    # Use Resnet 50 trained on imagnet to extract features, train linear probe on top.

    global resnet_model

    len_classes = len(train_dataset.classes)

    per_class_accuracy_top1 = { k:[0,0, test_dataset.classes[k]] for k in range(len_classes)}

    if resnet_model == None:
        resnet_model = torch.hub.load('pytorch/vision:v0.8.2', 'resnet50', pretrained=True)

    train_features, train_labels = get_resnet_features(train_dataset)
    test_features, test_labels = get_resnet_features(test_dataset)

    classifier = LogisticRegression(random_state=1, C=0.5, max_iter=5000, verbose=1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.

    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            per_class_accuracy_top1[test_labels[i]][0] +=1
        per_class_accuracy_top1[test_labels[i]][1] +=1

    return accuracy, per_class_accuracy_top1


def clip_linear_probe(train_dataset, test_dataset):
    # Use Resnet 50 trained on imagnet to extract features, train linear probe on top.

    global clip_model, clip_preprocess

    len_classes = len(train_dataset.classes)

    per_class_accuracy_top1 = { k:[0,0, test_dataset.classes[k]] for k in range(len_classes)}
    
    # lazy load
    if clip_model == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load('ViT-B/32', device)

    train_features, train_labels = get_clip_features(train_dataset)
    test_features, test_labels = get_clip_features(test_dataset)

    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.

    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            per_class_accuracy_top1[test_labels[i]][0] +=1
        per_class_accuracy_top1[test_labels[i]][1] +=1

    return accuracy, per_class_accuracy_top1


###########
# Datasets
###########

def cifar100_expt(expt_name="base"):
    print("CIFAR")
    root = os.path.expanduser("/nethome/bdevnani3/raid/data/")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
        ]
    )

    # lazy load
    global clip_preprocess
    if clip_preprocess == None:
        _, clip_preprocess = clip.load('ViT-B/32', device)

    train = CIFAR100(root, download=True, train=True, transform=transform)
    test = CIFAR100(root, download=True, train=False, transform=transform)
    train_clip = CIFAR100(root, download=True, train=True, transform=clip_preprocess)
    test_clip = CIFAR100(root, download=True, train=False, transform=clip_preprocess)

    print(" Clip Zero Shot")
    czs = clip_zero_shot(test_clip, test.classes)
    print(f"Clip Zero Shot Acc (1%): {czs[0]}")
    print(f"Clip Zero Shot Acc (5%): {czs[1]}")

    print(" Resnet Linear Probe")
    rlp = resnet_linear_probe(train,test)
    print(f"Resnet + Linear Probe Acc (1%): {rlp[0]}")

    print(" Clip Linear Probe")
    clp = clip_linear_probe(train_clip,test_clip)
    print(f"CLIP + Linear Probe Acc (1%): {clp[0]}")

    print(f"""
    CIFAR100 Results:
    Clip Zero Shot Acc (1%): {czs[0]}
    Clip Zero Shot Acc (5%): {czs[1]}
    Resnet + Linear Probe Acc (1%): {rlp[0]}
    CLIP + Linear Probe Acc (1%): {clp[0]}
    """)
    with open(f"{results_path}/cifar100/{expt_name}-accuracies", "w") as outfile: 
        outfile.write(f"""
            CIFAR100 Results:
            Clip Zero Shot Acc (1%): {czs[0]}
            Clip Zero Shot Acc (5%): {czs[1]}
            Resnet + Linear Probe Acc (1%): {rlp[0]}
            CLIP + Linear Probe Acc (1%): {clp[0]}
            """)


    with open(f"{results_path}/cifar100/{expt_name}-czs_per_class_acc_1.json", "w") as outfile: 
        json.dump(czs[2], outfile)

    with open(f"{results_path}/cifar100/{expt_name}-czs_per_class_acc_5.json", "w") as outfile: 
        json.dump(czs[3], outfile)

def flowers_expt(expt_name="base"):
    print("FLOWERS")

    root = datasets.ImageFolder("/nethome/bdevnani3/raid/data/flower_data")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]
            ),
        ]
    )

    train = datasets.ImageFolder("/nethome/bdevnani3/raid/data/flower_data/train", transform=transform)
    test = datasets.ImageFolder("/nethome/bdevnani3/raid/data/flower_data/test", transform=transform)
    train_clip = datasets.ImageFolder("/nethome/bdevnani3/raid/data/flower_data/train")
    test_clip = datasets.ImageFolder("/nethome/bdevnani3/raid/data/flower_data/test")

    print(" Clip Zero Shot")
    czs = clip_zero_shot(test_clip, test.classes, phrase="a photo of a {}, a kind of flower.")
    print(f"Clip Zero Shot Acc (1%): {czs[0]}")
    print(f"Clip Zero Shot Acc (5%): {czs[1]}")

    print(" Resnet Linear Probe")
    rlp = resnet_linear_probe(train,test)
    print(f"Resnet + Linear Probe Acc (1%): {rlp[0]}")

    print(" Clip Linear Probe")
    clp = clip_linear_probe(train_clip,test_clip)
    print(f"CLIP + Linear Probe Acc (1%): {clp[0]}")

    print(f"""
    Flower Dataset Results:
    Clip Zero Shot Acc (1%): {czs[0]}
    Clip Zero Shot Acc (5%): {czs[1]}
    Resnet + Linear Probe Acc (1%): {rlp[0]}
    CLIP + Linear Probe Acc (1%): {clp[0]}
    """)
    with open(f"{results_path}/cifar100/{expt_name}-accuracies", "w") as outfile: 
        outfile.write(f"""
            Flower Dataset Results:
            Clip Zero Shot Acc (1%): {czs[0]}
            Clip Zero Shot Acc (5%): {czs[1]}
            Resnet + Linear Probe Acc (1%): {rlp[0]}
            CLIP + Linear Probe Acc (1%): {clp[0]}
            """)


    with open(f"{results_path}/flower/{expt_name}-czs_per_class_acc_1.json", "w") as outfile: 
        json.dump(czs[2], outfile)

    with open(f"{results_path}/flower/{expt_name}-czs_per_class_acc_5.json", "w") as outfile: 
        json.dump(czs[3], outfile)    


if __name__ == "__main__":
    # flowers_expt()
    cifar100_expt()