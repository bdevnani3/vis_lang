import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

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

        self.train_learning_type = "class"
        self.test_learning_type = "class"

        self.task_labels = None

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

            correct += self.num_correct_preds(outputs, labels, self.train_learning_type)

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

    # def num_correct_preds(self, outputs, labels):
    #     val, predicted = outputs.max(1)
    #     return predicted.eq(labels).sum().item()

    def num_correct_preds(self, outputs, labels, learning_type):
        if learning_type == "task":
            z = torch.zeros_like(outputs)
            z[:, self.task_labels] = 1
            outputs = outputs * z
            outputs[outputs == 0] = -np.inf
        _, predicted = outputs.max(1)
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
                correct += self.num_correct_preds(
                    outputs, labels, self.train_learning_type
                )

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
        self.model.train()
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

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data

                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels)
                test_loss += loss.item()

                total += labels.size(0)

                correct += self.num_correct_preds(
                    outputs, labels, self.test_learning_type
                )

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


###################################################################################


class Bert(Base):
    def __init__(
        self,
        model,
        class_names,
        checkpoints_path: str = "/nethome/bdevnani3/raid/continual/base/models/",
        epochs: int = 200,
    ):
        super().__init__(model=model, checkpoints_path=checkpoints_path, epochs=epochs)

        # The similarity mode to pass into find_closest_words
        self.similarity_mode = "cossim"
        self.model.logit_scale = nn.Parameter(torch.ones([], device=get_device()))
        self.class_names = class_names
        self.init_bert_model()
        self.init_word_lookup()

    def find_closest_words(
        self,
        word_lookup: torch.Tensor,
        x: torch.Tensor,
        mode: str = "l2",
        eps: float = 0.00001,
    ) -> torch.Tensor:
        """
        Given a size [N, c] lookup table (N classes, c channels per vector) and a set of [M, c] vectors to look up,
        returns a size [M] vector of indices from 0 to N-1 containing the closest vector in the lookup for that input.

        Modes:
            l2     - Computes pairwise L2 distance and chooses the lowest one.
            cossim - Computes pairwise cosine similarity, and chooses the most similar.
            dot    - Computes pairwise dot product similarity, and choses the most similar
        """
        N, c = word_lookup.shape
        M, c2 = x.shape

        assert (
            c == c2
        ), "The lookup should have the same number of channels as the input."

        if mode == "l2":
            return (
                ((word_lookup[None, :, :] - x[:, None, :]) ** 2)
                .sum(dim=-1)
                .argmin(dim=-1)
            )
        elif mode == "cossim":
            # Note: we don't need to divide by the length of x here, because it's the same for the argmax.
            # Also, it's imporant that we can get away with that for numerical stability.
            return (
                (x @ word_lookup.t()) / (word_lookup.norm(dim=-1)[None, :] + eps)
            ).argmax(dim=-1)
        elif mode == "dot":
            return (x @ word_lookup.t()).argmax(dim=-1)
        else:
            raise NotImplementedError

    def init_bert_model(self, model="stsb-bert-base"):
        print(f"Initializing {model}...")
        self.transformer_model = SentenceTransformer(model)

    def init_word_lookup(self):
        word_vectors = [
            self.transformer_model.encode(_class) for _class in self.class_names
        ]
        self.model.word_lookup = torch.from_numpy(np.stack(word_vectors)).to(
            get_device()
        )
        self.model.word_lookup = self.model.word_lookup / self.model.word_lookup.norm(
            dim=-1, keepdim=True
        )

    def num_correct_preds(self, outputs, labels, learning_type):
        if learning_type == "task":
            z = torch.zeros_like(self.model.word_lookup)
            z[self.task_labels, :] = 1
            z = self.model.word_lookup * z
            z[z == 0] = np.inf
            return (
                (
                    self.find_closest_words(z, outputs, mode=self.similarity_mode)
                    == labels
                )
                .sum()
                .item()
            )
        return (
            (
                self.find_closest_words(
                    self.model.word_lookup, outputs, mode=self.similarity_mode
                )
                == labels
            )
            .sum()
            .item()
        )

    def calc_loss(self, outputs, labels):
        """ Most of this is copied line-for-line from https://github.com/openai/CLIP/blob/main/clip/model.py#L353 """

        image_features = outputs
        text_features = self.model.word_lookup

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Note: I prenormalize the text features in init
        # text_features  = text_features  / text_features .norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        return self.criterion(logits_per_image, labels)
