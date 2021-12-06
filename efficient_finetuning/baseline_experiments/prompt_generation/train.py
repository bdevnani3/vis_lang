import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import models
from utils import get_device, make_dirs

import os
import time
import json

import gensim


class Experiment:
    """
    Base class for handling model training/test/evaluation.
    """

    def __init__(
        self,
        root_path: str = ".",
        project_path: str = "vis_lang",
        variant_name: str = "base",
        epochs: int = 200,
        loss_name: str = "default"
    ):
        self.root_path = root_path
        self.project_path = project_path
        self.variant_name = variant_name
        self.epochs = epochs
        self.checkpoints_path = os.path.join(
            self.root_path, self.project_path, self.variant_name, "models/"
        )
        make_dirs(self.checkpoints_path)

        self.plots_path = os.path.join(
            self.root_path, self.project_path, self.variant_name, "plots/"
        )
        make_dirs(self.plots_path)

        self.class_names = None
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.train_losses = []
        self.train_accuracy = []
        self.test_losses = []
        self.test_accuracy = []
        self.best_accuracy = 0
        self.min_loss = np.inf

        # Logs
        self.logs_path = os.path.join(
            self.root_path, project_path, self.variant_name, "logs.txt"
        )

        self.loss_name = loss_name
        self.similarity_mode = None

    def set_up_model_architecture(self, num_features_in_last_layer: int):
        """
        Set up architecture of the model. Since we will most likely be altering the final
        layer of pre-existing architectures, this supports that functionality. Initialize
        self.model.
        """
        model = models.resnet18()
        model.linear = nn.Linear(
            in_features=512, out_features=num_features_in_last_layer
        )
        self.model = model
        self.model.word_lookup = None
        if torch.cuda.is_available():
            self.model.cuda()

    def init_model_helpers(self, criterion=nn.MSELoss):
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

    def train_single_epoch(self, epoch_idx):
        """
        Ensure to update self.train_losses & self.train_accuracy
        :param epoch_idx: Index of the epoch
        :param train_loader: dataloader object for the training dataset
        :return: None
        """
        train_loss = 0.0
        total = 0
        correct = 0

        start_time = time.time()

        self.model.train()
        for i, data in enumerate(self.train_loader, 0):
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

        epoch_loss = train_loss / len(self.train_loader)
        epoch_accuracy = correct * 100 / total

        elapsed = time.time() - start_time
        print(
            f"Training: Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}% || Time: {elapsed:6.2f}"
        )

        self.train_losses.append(epoch_loss)
        self.train_accuracy.append(epoch_accuracy)

    def eval_acc(self):
        """ Returns the per-class accuracy evaluated on the validation set. """
        self.model.eval()

        totals = [0] * len(self.class_names)
        correct = [0] * len(self.class_names)

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)

                for idx, _class in enumerate(self.class_names):
                    is_class = labels == idx

                    cur_labels = labels[is_class]
                    if cur_labels.shape[0] == 0:
                        continue

                    cur_outputs = outputs[is_class]

                    totals[idx] += cur_labels.size(0)
                    correct[idx] += self.num_correct_preds(cur_outputs, cur_labels)

        return [c * 100 / t for c, t in zip(correct, totals)]

    def validate_single_epoch(self, epoch_idx):
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
            for i, data in enumerate(self.test_loader, 0):
                inputs, labels = data
                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels)
                test_loss += loss.item()

                total += labels.size(0)
                correct += self.num_correct_preds(outputs, labels)

        epoch_loss = test_loss / len(self.test_loader)
        epoch_accuracy = correct * 100 / total

        state = {
            "net": self.model.state_dict(),
            "acc": epoch_accuracy,
            "epoch": epoch_idx,
            "loss": epoch_loss,
        }
        print(
            f"Testing : Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}%"
        )
        if self.best_accuracy < epoch_accuracy:
            self.best_accuracy = epoch_accuracy
            path = os.path.join(self.checkpoints_path, f"best_acc.pth")
            print(f"Saving model at {path} as it has best accuracy so far.")
            torch.save(
                state,
                path
            )

        if epoch_idx == 190:
            path = os.path.join(self.checkpoints_path, f"last_model.pth")
            print(f"Saving model at {path} due to epoch idx = 190.")
            torch.save(
                state,
                path
            )

        if self.min_loss > epoch_loss:
            self.min_loss = epoch_loss
            # print(f"Saving model as it has best LOSS so far.")
            # torch.save(
            #     state,
            #     os.path.join(self.checkpoints_path, "best_loss.pth"),
            # )
        self.test_losses.append(epoch_loss)
        self.test_accuracy.append(epoch_accuracy)

    def train_model(self):

        print("Started Training")

        for epoch in range(self.epochs):
            self.train_single_epoch(epoch)
            self.validate_single_epoch(epoch)
            self.scheduler.step()
            print()

        print("Finished Training")

        print("Training Loss: ", self.train_losses)
        print("Training Accuracy: ", self.train_accuracy)
        print("Test Loss: ", self.test_losses)
        print("Test Accuracy: ", self.test_accuracy)

        self.export_plots()
        self.export_data()

    def export_data(self):

        filename = os.path.join(
            self.root_path, self.project_path, self.variant_name, "raw_data"
        )

        print(f"Saving data at {filename}")

        data = {
            "Train Loss": self.train_losses,
            "Train Acc": self.train_accuracy,
            "Test Loss": self.test_losses,
            "Test Acc": self.test_accuracy,
            "Best Train Acc": max(self.train_accuracy),
            "Best Test Acc": max(self.test_accuracy),
        }

        with open(filename, "w") as f:
            json.dump(data, f)

    def export_plots(self):

        print(f"Saving plots at {self.plots_path}")

        # TODO: Clean up redundancy

        train_losses_fig = plt.figure()
        plt.plot(self.train_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")
        plt.title(self.variant_name)
        train_losses_fig.savefig(os.path.join(self.plots_path, "train_loss.png"))

        test_losses_fig = plt.figure()
        plt.plot(self.test_losses)
        plt.xlabel("Epochs")
        plt.ylabel("Test Loss")
        plt.title(self.variant_name)
        test_losses_fig.savefig(os.path.join(self.plots_path, "test_loss.png"))

        train_acc_fig = plt.figure()
        plt.plot(self.train_accuracy)
        plt.xlabel("Epochs")
        plt.ylabel("Train Acc")
        plt.title(self.variant_name)
        train_acc_fig.savefig(os.path.join(self.plots_path, "train_acc.png"))

        test_acc_fig = plt.figure()
        plt.plot(self.test_accuracy)
        plt.xlabel("Epochs")
        plt.ylabel("Test Acc")
        plt.title(self.variant_name)
        test_acc_fig.savefig(os.path.join(self.plots_path, "test_acc.png"))


#####################
# Embedding Functions
#####################

    def find_closest_words(
        self,
        word_lookup: torch.Tensor,
        x: torch.Tensor,
        mode: str = None,
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

    def num_correct_preds(self, outputs, labels):
        if self.model.word_lookup is None:
            _, predicted = outputs.max(1)
            return predicted.eq(labels).sum().item()
        else:
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

########################
# Losses
########################

    def calc_loss(self, outputs, labels):

        if self.model.word_lookup is not None:
            loss_labels = self.model.word_lookup[labels]
        else:
            loss_labels = labels
        if self.loss_name == "default":
            return self.criterion(outputs, loss_labels) 

        elif self.loss_name == "dot":
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            return -(outputs * loss_labels).sum(-1).mean()

        elif self.loss_name == "cosine":
            loss = torch.mean(self.criterion(outputs, loss_labels))
            return 1.0 - loss

        elif self.loss_name == "clip":
            image_features = outputs
            text_features = self.model.word_lookup

            # normalized features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            return self.criterion(logits_per_image, labels)

        elif self.loss_name == "fuzzy_mse":
            criterion = nn.MSELoss(reduction="none")
            mse = criterion(outputs, loss_labels)
            is_correct = self.find_closest_words(self.model.word_lookup, outputs, self.similarity_mode) == labels
            return (mse * ~is_correct[:, None]).mean()

def cosine_loss(output, target):
    loss = 1 - torch.cosine_similarity(output, target)
    return loss
        