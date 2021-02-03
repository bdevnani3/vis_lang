__author__ = 'bhavika'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
import numpy as np
import gensim.downloader

EPOCHS = 200

def load_data_cifar10(train=True):
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

    dataset = change_target_to_word_vectors(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64,
                                              shuffle=True, num_workers=8)
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

def set_up_model():

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=300)

    criterion = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    return model, criterion, optimizer, scheduler

def train_model():
    train_loss = 0.0
    total  = 0
    correct = 0

    model.train()
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

        train_loss += loss.item()
        total += labels.size(0)
        labels, outputs = labels.to("cpu"), outputs.to("cpu")
        for l, o in zip(labels, outputs):
            label_word = word_vectors.similar_by_vector(l.numpy(), topn=1)
            output_word = word_vectors.similar_by_vector(o.data.numpy(), topn=1)
            correct += label_word[0][0] == output_word[0][0]

    epoch_loss = train_loss / len(trainloader)
    epoch_accuracy = correct*100/total

    return epoch_loss, epoch_accuracy

def validate_model(epoch):
    test_loss = 0.0
    total  = 0
    correct = 0

    global best_accuracy
    global min_loss

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)

            if torch.cuda.is_available():
                model.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            total += labels.size(0)
            labels, outputs = labels.to("cpu"), outputs.to("cpu")
            for l, o in zip(labels, outputs):
                label_word = word_vectors.similar_by_vector(l.numpy(), topn=1)
                output_word = word_vectors.similar_by_vector(o.data.numpy(), topn=1)
                correct += label_word[0][0] == output_word[0][0]

    epoch_loss = test_loss / len(testloader)
    epoch_accuracy = correct*100/total

    state = {
            'net': model.state_dict(),
            'acc': epoch_accuracy,
            'epoch': epoch,
            'loss': epoch_loss
        }
    if best_accuracy < epoch_accuracy:
        best_accuracy = epoch_accuracy
        print("Saving model with acc: {}, loss: {}, epoch: {}".format(epoch_accuracy, epoch_loss, epoch))
        torch.save(state, '/nethome/bdevnani3/raid/trained_models/vis_lang/cifar10_emb_best_acc.pth')

    if min_loss > epoch_loss:
        min_loss = epoch_loss
        print("Saving model with acc: {}, loss: {}, epoch: {}".format(epoch_accuracy, epoch_loss, epoch))
        torch.save(state, '/nethome/bdevnani3/raid/trained_models/vis_lang/cifar10_emb_best_loss.pth')

    return epoch_loss, epoch_accuracy


# Set up data
trainloader = load_data_cifar10()
testloader = load_data_cifar10(train=False)

# Set up model
model, criterion, optimizer, scheduler = set_up_model()

# Train model

print('Started Training')
train_losses = []
train_accuracy = []
test_losses = []
test_accuracy = []

best_accuracy = 0
min_loss = np.inf

for epoch in range(EPOCHS):

    l, a = train_model()
    train_losses.append(l)
    train_accuracy.append(a)

    l, a = validate_model(epoch)
    test_losses.append(l)
    test_accuracy.append(a)

    scheduler.step()

print('Finished Training')

print('Training Loss: ', train_losses)
print('Training Accuracy: ', train_accuracy)
print('Test Loss: ', test_losses)
print('Test Accuracy: ', test_accuracy)



