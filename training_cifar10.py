__author__ = 'bhavika'

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
import gensim.downloader
import numpy as np
import sys

word_vectors = None

def load_data_cifar10(train=True, train_on_embeddings=False):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.491, 0.482, 0.446],
                std= [0.247, 0.243, 0.261]
            )]) # TODO: Automate calculation given a dataset

    dataset = datasets.CIFAR10(root='/nethome/bdevnani3/raid/data', train=train,
                                            download=True, transform=transform)
    if train_on_embeddings:
        dataset = change_target_to_word_vectors(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
                                              shuffle=True, num_workers=2)
    return dataloader


def change_target_to_word_vectors(dataset):
    model = 'word2vec-google-news-300'
    global word_vectors
    word_vectors = gensim.downloader.load(model)

    def transform_targets(x):
        return word_vectors[idx_to_class[x]]

    idx_to_class = {y:x for x,y in dataset.class_to_idx.items()}
    dataset.targets = np.array(list(map(transform_targets, dataset.targets)))
    return dataset

def set_up_model(out_features=10, loss=nn.CrossEntropyLoss()):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=out_features)
    criterion = loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # and a learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return model, criterion, optimizer, scheduler

def train_model(model, trainloader, optimizer, criterion, scheduler, epochs=15, verbose=False, emb_model=False):
    model.train()
    losses = []
    if emb_model:
        print("########## {} ##########".format("Embedding Model"))
    else:
        print("########## {} ##########".format("Base Model"))
    for epoch in range(epochs):  # loop over the dataset multiple times
        if verbose:
            print("########## {} ##########".format(epoch+1))
        train_loss = 0.0
        total  = 0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if torch.cuda.is_available():
                model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

            # print statistics
            if verbose:
                if emb_model:
                    # print statistics
                    total += labels.size(0)
                    labels, outputs = labels.to("cpu"), outputs.to("cpu")
                    for l, o in zip(labels, outputs):
                        label_word = word_vectors.similar_by_vector(l.numpy(), topn=1)
                        output_word = word_vectors.similar_by_vector(o.data.numpy(), topn=1)
                        correct += label_word[0][0] == output_word[0][0]
                    if i % 200 == 199:    # print every 200 mini-batches
                        print("Loss: {} | Acc: {} | {}/{}".format(train_loss/200, 100.*correct/total, correct, total))
                        train_loss = 0
                        print(label_word, output_word)

                else:
                    # print statistics
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    if i % 200 == 199:    # print every 200 mini-batches
                        print("Loss: {} | Acc: {} | {}/{}".format(train_loss/200, 100.*correct/total, correct, total))
                        train_loss = 0

        epoch_loss = train_loss / len(trainloader)
        losses.append(epoch_loss)
        print(epoch_loss)
    if verbose:
        print('Finished Training')
    print(losses)
    return model

if __name__ == "__main__":

    #TODO: Add check-pointing
    #TODO: Add time logging

    if str(sys.argv[1]) == "b":
        # train base model
        trainloader = load_data_cifar10()
        model, criterion, optimizer, scheduler = set_up_model()
        model = train_model(model, trainloader, optimizer, criterion, scheduler, epochs=200, verbose=False)
        torch.save(model.state_dict(), "/nethome/bdevnani3/raid/trained_models/vis_lang/pred_class.pt")

    else:
        # train emb model
        trainloader = load_data_cifar10(True, True)
        model, criterion, optimizer, scheduler = set_up_model(out_features=300, loss=nn.MSELoss())
        model = train_model(model, trainloader, optimizer, criterion, scheduler, epochs=200, verbose=False, emb_model=True)
        torch.save(model.state_dict(), "/nethome/bdevnani3/raid/trained_models/vis_lang/pred_emb.pt")


