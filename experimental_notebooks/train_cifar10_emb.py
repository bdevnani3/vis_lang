__author__ = "bhavika"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import gensim.downloader

import models

import os
import time

EPOCHS = 200

# This is so that we can run it on other machines
if os.path.exists("/nethome/bdevnani3/raid"):
    ROOT_PATH = "/nethome/bdevnani3/raid"
else:
    ROOT_PATH = ".."

class_names = None  # initialized once in load_data_cifar10


def make_dirs(path: str):
    """ Why is this not how the standard library works? """
    path = os.path.split(path)[0]
    if path != "":
        os.makedirs(path, exist_ok=True)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_cifar10(train=True):
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)
                ),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

    dataset = datasets.CIFAR10(
        root=os.path.join(ROOT_PATH, "data"),
        train=train,
        download=True,
        transform=transform,
    )

    global class_names
    if class_names is None:
        class_names = dataset.classes

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=8
    )
    return dataloader


def init_word_lookup(model):
    # We only need to lazily initialize this once. Don't reinitialize it if it's already been initialized.
    word_vectors = gensim.downloader.load(name="word2vec-google-news-300")

    # Note: we store the word lookup in the model, not the datset because
    #   1.) The word lookup should be on the same device as the model
    #   2.) If using multiple GPUs, the model will get duplicated to each device, but the dataset won't
    #   3.) The word model (i.e., textual feature encoder) is a property of the model not the dataset
    model.word_lookup = torch.from_numpy(
        np.stack([word_vectors[_class] for _class in class_names])
    ).to(get_device())


def set_up_model():

    model = models.resnet18()
    model.linear = nn.Linear(in_features=512, out_features=300)

    init_word_lookup(model)

    criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    return model, criterion, optimizer, scheduler


def find_closest_words(
    word_lookup: torch.Tensor, x: torch.Tensor, mode: str = "l2"
) -> torch.Tensor:
    """
    Given a size [N, c] lookup table (N classes, c channels per vector) and a set of [M, c] vectors to look up,
    returns a size [M] vector of indices from 0 to N-1 containing the closest vector in the lookup for that input.

    Modes:
        l2     - Computes pairwise L2 distance and chooses the lowest one.
        cossim - Computs pairwise cosine similarity, and chooses the most similar. (Not implemented)
    """
    N, c = word_lookup.shape
    M, c2 = x.shape

    assert c == c2, "The lookup should have the same number of channels as the input."

    if mode == "l2":
        return (
            ((word_lookup[None, :, :] - x[:, None, :]) ** 2).sum(dim=-1).argmin(dim=-1)
        )
    else:
        raise NotImplementedError


def train_model(epoch_idx, model):
    train_loss = 0.0
    total = 0
    correct = 0

    start_time = time.time()

    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        device = get_device()
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, model.word_lookup[labels])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        total += labels.size(0)

        correct += (
            (find_closest_words(model.word_lookup, outputs) == labels).sum().item()
        )

    epoch_loss = train_loss / len(trainloader)
    epoch_accuracy = correct * 100 / total

    elapsed = time.time() - start_time
    print(
        f"Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} | Accuracy: {epoch_accuracy:6.2f}% || Time: {elapsed:6.2f}"
    )

    return epoch_loss, epoch_accuracy


def validate_model(epoch):
    test_loss = 0.0
    total = 0
    correct = 0

    global best_accuracy
    global min_loss

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            device = get_device()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, model.word_lookup[labels])
            test_loss += loss.item()

            total += labels.size(0)
            correct += (
                (find_closest_words(model.word_lookup, outputs) == labels).sum().item()
            )

    epoch_loss = test_loss / len(testloader)
    epoch_accuracy = correct * 100 / total

    state = {
        "net": model.state_dict(),
        "acc": epoch_accuracy,
        "epoch": epoch,
        "loss": epoch_loss,
    }
    if best_accuracy < epoch_accuracy:
        best_accuracy = epoch_accuracy
        print(
            "Saving model with acc: {:6.2f}%, loss: {:.3f}, epoch: {}".format(
                epoch_accuracy, epoch_loss, epoch
            )
        )
        best_acc_path = os.path.join(
            ROOT_PATH, "trained_models/vis_lang/cifar10_emb_best_acc.pth"
        )
        make_dirs(best_acc_path)
        torch.save(state, best_acc_path)

    if min_loss > epoch_loss:
        min_loss = epoch_loss
        print(
            "Saving model with acc: {:6.2f}%, loss: {:.3f}, epoch: {}".format(
                epoch_accuracy, epoch_loss, epoch
            )
        )
        best_loss_path = os.path.join(
            ROOT_PATH, "trained_models/vis_lang/cifar10_emb_best_loss.pth"
        )
        make_dirs(best_loss_path)
        torch.save(state, best_loss_path)

    return epoch_loss, epoch_accuracy


if __name__ == "__main__":
    # Set up data
    trainloader = load_data_cifar10()
    testloader = load_data_cifar10(train=False)

    # Set up model
    model, criterion, optimizer, scheduler = set_up_model()

    # Train model

    print("Started Training")
    train_losses = []
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    best_accuracy = 0
    min_loss = np.inf

    for epoch in range(EPOCHS):

        l, a = train_model(epoch, model)
        train_losses.append(l)
        train_accuracy.append(a)

        l, a = validate_model(epoch)
        test_losses.append(l)
        test_accuracy.append(a)

        scheduler.step()

    print("Finished Training")

    print("Training Loss: ", train_losses)
    print("Training Accuracy: ", train_accuracy)
    print("Test Loss: ", test_losses)
    print("Test Accuracy: ", test_accuracy)
