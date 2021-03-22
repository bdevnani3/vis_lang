import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import copy

import os
import time
import json


class Base:
    """
    Base class for handling experiments.
    """

    def __init__(
        self,
        model,
        checkpoints_path: str = "/nethome/bdevnani3/raid/continual/base/models/",
        epochs: int = 200,
    ):
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.checkpoints_path = checkpoints_path
        self.epochs = epochs
        self.dataloader_args = {"batch_size": 32, "shuffle": True, "num_workers": 2}

        self.class_names = None
        self.test_loader = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.train_losses = {}
        self.train_accuracy = {}
        self.test_losses = {}
        self.test_accuracy = {}
        self.best_accuracy = {}
        self.min_loss = {}

        # Logs
        self.logs_path = os.path.join(self.checkpoints_path, "logs.txt")

        self.init_model_helpers()

    def init_model_helpers(self, criterion=nn.CrossEntropyLoss):
        self.criterion = criterion()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[150, 250, 350], gamma=0.1
        )
        if torch.cuda.is_available():
            self.criterion.cuda()

    def load_best(self):
        """
        Loads the best accuracy model checkpoint into self.model.
        """
        path = os.path.join(self.checkpoints_path, f"best_acc.pth")

        self.model.load_state_dict(torch.load(path)["net"])

    def train_single_epoch(self, epoch_idx, task_id, data_loader):
        """
        Ensure to update self.train_losses & self.train_accuracy
        :param epoch_idx: Index of the epoch
        :param data_loader: dataloader object for the training dataset
        :return: None
        """
        train_loss = 0.0
        total = 0
        correct = 0

        start_time = time.time()

        self.model.train()
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            device = get_device()
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)

            loss = self.calc_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += labels.size(0)

            correct += self.num_correct_preds(outputs, labels)

        epoch_loss = train_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        elapsed = time.time() - start_time
        print(
            f"Training: Task ID: {task_id} || Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}% || Time: {elapsed:6.2f}"
        )

        if not task_id in self.train_losses:
            self.train_losses[task_id] = []

        if not task_id in self.train_accuracy:
            self.train_accuracy[task_id] = []

        self.train_losses[task_id].append(epoch_loss)
        self.train_accuracy[task_id].append(epoch_accuracy)

    def num_correct_preds(self, outputs, labels):
        val, predicted = outputs.max(1)
        return predicted.eq(labels).sum().item()

    def calc_loss(self, outputs, labels):
        return self.criterion(outputs, labels)

    def validate_single_epoch(self, epoch_idx, task_id, data_loader):
        """
        Ensure to update self.test_losses & self.test_accuracy
        :param epoch_idx: Index of the epoch
        :param test_loader: dataloader object for the test dataset
        :return: None
        """
        test_loss = 0.0
        total = 0
        correct = 0

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data
                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels)
                test_loss += loss.item()

                total += labels.size(0)
                correct += self.num_correct_preds(outputs, labels)

        epoch_loss = test_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        state = {
            "net": self.model.state_dict(),
            "acc": epoch_accuracy,
            "epoch": epoch_idx,
            "loss": epoch_loss,
            "task_id": task_id,
        }
        # print(
        #     f"Testing : Task ID: {task_id} || Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}%"
        # )

        if task_id not in self.best_accuracy:
            self.best_accuracy[task_id] = 0

        if self.best_accuracy[task_id] <= epoch_accuracy:
            self.best_accuracy[task_id] = epoch_accuracy
            p = os.path.join(self.checkpoints_path, f"model_{task_id}.pth")
            print(f"Saving model at {p} as it has best accuracy so far.")
            torch.save(
                state,
                p,
            )

        if task_id not in self.min_loss:
            self.min_loss[task_id] = np.inf

        if self.min_loss[task_id] > epoch_loss:
            self.min_loss[task_id] = epoch_loss

        if task_id not in self.test_losses:
            self.test_losses[task_id] = []

        if task_id not in self.test_accuracy:
            self.test_accuracy[task_id] = []

        self.test_losses[task_id].append(epoch_loss)
        self.test_accuracy[task_id].append(epoch_accuracy)

    def train_model(self, task_id, data_loader):

        print("Started Training")

        for epoch in range(self.epochs):
            self.train_single_epoch(epoch, task_id, data_loader["train"])
            self.validate_single_epoch(epoch, task_id, data_loader["train"])
            self.scheduler.step()
            print()

        print("Finished Training")

        print("Training Loss: ", self.train_losses[task_id])
        print("Training Accuracy: ", self.train_accuracy[task_id])
        print("Test Loss: ", self.test_losses[task_id])
        print("Test Accuracy: ", self.test_accuracy[task_id])

        self.export_data(task_id)

    def test_model(self, task_id, data_loader):
        test_loss = 0.0
        total = 0
        correct = 0

        self.model.eval()

        def num_correct_preds_test(outputs, labels):
            val, predicted = outputs.max(1)
            return predicted.eq(labels).sum().item()

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data
                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels)
                test_loss += loss.item()

                total += labels.size(0)
                correct += self.num_correct_preds(outputs, labels)

        epoch_loss = test_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        state = {
            "net": self.model.state_dict(),
            "acc": epoch_accuracy,
            "loss": epoch_loss,
            "task_id": task_id,
        }
        # print(
        #     f"Testing Overall :  Task ID: {task_id} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}%"
        # )
        return state

    def export_data(self, task_id):

        filename = os.path.join(self.checkpoints_path, f"raw_data_{task_id}")

        print(f"Saving data at {filename}")

        data = {
            "Train Loss": self.train_losses[task_id],
            "Train Acc": self.train_accuracy[task_id],
            "Test Loss": self.test_losses[task_id],
            "Test Acc": self.test_accuracy[task_id],
            "Best Train Acc": max(self.train_accuracy[task_id]),
            "Best Test Acc": max(self.test_accuracy[task_id]),
            "Task ID": task_id,
        }

        with open(filename, "w") as f:
            json.dump(data, f)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
