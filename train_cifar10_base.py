__author__ = 'bhavika'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models, datasets
import numpy as np

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

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128,
                                              shuffle=True, num_workers=8)
    return dataloader

def set_up_model():

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=10)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

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

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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
        torch.save(state, '/nethome/bdevnani3/raid/trained_models/vis_lang/cifar10_base_best_acc.pth')

    if min_loss > epoch_loss:
        min_loss = epoch_loss
        print("Saving model with acc: {}, loss: {}, epoch: {}".format(epoch_accuracy, epoch_loss, epoch))
        torch.save(state, '/nethome/bdevnani3/raid/trained_models/vis_lang/cifar10_base_best_loss.pth')

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


