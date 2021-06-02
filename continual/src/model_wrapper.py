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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Base:
    """
    Base class for handling experiments.
    """

    def __init__(
        self,
        model,
        checkpoints_path: str = "/nethome/bdevnani3/raid/continual/base/models/",
        epochs: int = 200,
        early_stop: bool = False,
        pick_best_model:bool = True,
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

        self.deep_test_losses = {}
        self.deep_test_accuracies = {}

        self.early_stop = early_stop
        self.pick_best_model = pick_best_model

        self.seen_labels = []

        # Logs
        self.logs_path = os.path.join(
            self.checkpoints_path, "logs.txt"
        )
        self.criterion=nn.CrossEntropyLoss
        self.init_model_helpers()

        self.train_learning_type="class"
        self.test_learning_type="class"

        self.task_labels=None

        self.emb_vis = {}
        self.collect_vis_data = False

    def init_model_helpers(self, criterion=nn.CrossEntropyLoss):
        self.criterion = criterion()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[], gamma=0.1
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
        cla_correct = 0
        task_correct = 0

        start_time = time.time()

        self.model.train()
        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels= data
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

            cla_correct += self.num_correct_preds(outputs, labels, "class")
            task_correct += self.num_correct_preds(outputs, labels, "task")

        epoch_loss = train_loss / len(data_loader)
        epoch_cla_accuracy = cla_correct * 100 / total
        epoch_task_accuracy = task_correct * 100 / total

        elapsed = time.time() - start_time
        print(
            f"Training: Task ID: {task_id} || Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Class Accuracy: {epoch_cla_accuracy:6.2f}% || Task Accuracy: {epoch_task_accuracy:6.2f}% || Time: {elapsed:6.2f}"
        )

        if not task_id in self.train_losses:
            self.train_losses[task_id] = []

        if not task_id in self.train_accuracy:
            self.train_accuracy[task_id] = []

        self.train_losses[task_id].append(epoch_loss)
        if self.train_learning_type == "class": 
            self.train_accuracy[task_id].append(epoch_cla_accuracy)
        else:
            self.train_accuracy[task_id].append(epoch_task_accuracy)

    def num_correct_preds(self, outputs, labels, learning_type):
        if learning_type == "task":
            z = torch.zeros_like(outputs)
            z[:,self.task_labels] = 1
            outputs = outputs*z
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
                correct += self.num_correct_preds(outputs, labels, self.train_learning_type)

        epoch_loss = test_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        state = {
            "net": self.model.state_dict(),
            "acc": epoch_accuracy,
            "epoch": epoch_idx,
            "loss": epoch_loss,
            "task_id": task_id
        }
        print(
            f"Testing : Task ID: {task_id} || Epoch {epoch_idx} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}%"
        )
        if self.pick_best_model == True:
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

        else:
            if epoch_idx == self.epochs-1:
                p = os.path.join(self.checkpoints_path, f"model_{task_id}.pth")
                print(f"Saving model at {p} as we have reached the last epoch.")
                torch.save(
                    state,
                    p,
                )

        if task_id not in self.test_losses:
            self.test_losses[task_id] = []

        if task_id not in self.test_accuracy:
            self.test_accuracy[task_id] = []

        self.test_losses[task_id].append(epoch_loss)
        self.test_accuracy[task_id].append(epoch_accuracy)

    def test_single_epoch(self, epoch_idx, task_id, test_task_id ,data_loader): 
        """
        Method to test accuracy on previous tasks.
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
                correct += self.num_correct_preds(outputs, labels, self.test_learning_type)

        epoch_loss = test_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        key = f"{task_id}_{test_task_id}"

        if key not in self.deep_test_losses:
            self.deep_test_losses[key] = []

        if key not in self.deep_test_accuracies:
            self.deep_test_accuracies[key] = []

        self.deep_test_losses[key].append(epoch_loss)
        self.deep_test_accuracies[key].append(epoch_accuracy)

    def test_single_epoch_vis(self, epoch_idx, task_id, test_task_id ,data_loader): 
        """
        Method to test accuracy on previous tasks.
        """
        self.model.eval()
        if task_id not in self.emb_vis:
            self.emb_vis[task_id] = {}

        if test_task_id not in self.emb_vis[task_id]:
            self.emb_vis[task_id][test_task_id] = {}

        if epoch_idx not in self.emb_vis[task_id][test_task_id]:
            self.emb_vis[task_id][test_task_id][epoch_idx] = {}

        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels = data
                device = get_device()
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.calc_loss(outputs, labels)

                labels = labels.cpu().numpy().tolist()

                for i, label in enumerate(labels):

                    if label not in self.emb_vis[task_id][test_task_id][epoch_idx]:
                        self.emb_vis[task_id][test_task_id][epoch_idx][label] = {}
                        self.emb_vis[task_id][test_task_id][epoch_idx][label]["embeddings"] = []
                        self.emb_vis[task_id][test_task_id][epoch_idx][label]["targets"] = []

                    self.emb_vis[task_id][test_task_id][epoch_idx][label]["embeddings"].append(outputs[i].cpu().numpy().tolist())
                    self.emb_vis[task_id][test_task_id][epoch_idx][label]["targets"].append(labels[i])

    def train_model(self, task_id, data_loader, deep_test=False, collect_vis_data=False):
        print("Started Training")
        self.model.train()
        for epoch in range(self.epochs):
            self.task_labels = data_loader[task_id]['task_labels']
            self.train_single_epoch(epoch, task_id, data_loader[task_id]['train'])
            self.validate_single_epoch(epoch, task_id, data_loader[task_id]['valid'])
            self.seen_labels.extend(self.task_labels)
            if deep_test:
                print("################ Deep Testing ################")
                for u in range(task_id+1):
                    self.task_labels = data_loader[u]['task_labels']
                    self.test_single_epoch(epoch, task_id, u, data_loader[u]['test'])
                # for u in self.seen_labels:
                #     self.task_labels = data_loader[u]['test_tasksize_1']
                #     self.test_single_epoch(epoch, task_id, u, data_loader[u]['test_tasksize_1'])
            if collect_vis_data:
                for u in range(task_id+1):
                    self.task_labels = data_loader[u]['task_labels']
                    self.test_single_epoch_vis(epoch, task_id, u, data_loader[u]['test'])
            self.scheduler.step()
            print()

        print("Finished Training")

        # print("Training Loss: ", self.train_losses[task_id])
        # print("Training Accuracy: ", self.train_accuracy[task_id])
        # print("Test Loss: ", self.test_losses[task_id])
        # print("Test Accuracy: ", self.test_accuracy[task_id])

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

                correct += self.num_correct_preds(outputs, labels, self.test_learning_type)

        epoch_loss = test_loss / len(data_loader)
        epoch_accuracy = correct * 100 / total

        state = {
            "net": self.model.state_dict(),
            "acc": epoch_accuracy,
            "loss": epoch_loss,
            "task_id": task_id
        }
        # print(
        #     f"Testing Overall :  Task ID: {task_id} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:6.2f}%"
        # )
        return state

    def export_data(self, task_id):

        filename = os.path.join(
            self.checkpoints_path, f"raw_data_{task_id}"
        )

        print(f"Saving data at {filename}")

        # dta = {}
        # dtl = {}
        # for k in self.deep_test_accuracies:
        #     if k.startswith(f"{task_id}_"):
        #         dta[k] = self.deep_test_accuracies[k]
        #         dtl[k] = self.deep_test_losses[k]


        data = {
            "Train Loss": self.train_losses[task_id],
            "Train Acc": self.train_accuracy[task_id],
            "Test Loss": self.test_losses[task_id],
            "Test Acc": self.test_accuracy[task_id],
            "Best Train Acc": max(self.train_accuracy[task_id]),
            "Best Test Acc": max(self.test_accuracy[task_id]),
            "Task ID": task_id,
            "Deep Test Acc": self.deep_test_accuracies,
            "Deep Test Loss": self.deep_test_losses
        }

        with open(filename, "w") as f:
            json.dump(data, f)

        filename = os.path.join(
            self.checkpoints_path, f"vis_data_{task_id}"
        )

        print(f"Saving data at {filename}")

        with open(filename, "w") as f:
            json.dump(self.emb_vis, f)



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
        early_stop: bool = False,
        pick_best_model:bool = True,
    ):
        super().__init__(model=model, checkpoints_path=checkpoints_path, epochs=epochs, 
        early_stop=early_stop, pick_best_model = pick_best_model)

        self.similarity_mode = "cossim"
        self.model.logit_scale = nn.Parameter(torch.ones([], device=get_device()))
        self.class_names = class_names
        self.init_bert_model()
        self.init_word_lookup()
        self.init_model_helpers()

    def init_model_helpers(self, criterion=nn.CrossEntropyLoss):
        self.criterion = criterion()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[500], gamma=0.1
        )
        if torch.cuda.is_available():
            self.criterion.cuda()

    def find_dist_words(
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
                -1*((word_lookup[None, :, :] - x[:, None, :]) ** 2)
                .sum(dim=-1)
            )
        elif mode == "cossim":
            # Note: we don't need to divide by the length of x here, because it's the same for the argmax.
            # Also, it's imporant that we can get away with that for numerical stability.
            return (
                (x @ word_lookup.t()) / (word_lookup.norm(dim=-1)[None, :] + eps)
            )
        elif mode == "dot":
            return (x @ word_lookup.t())
        else:
            raise NotImplementedError
            # .argmin(dim=-1)

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
        out = self.find_dist_words(
                self.model.word_lookup, outputs, mode="cossim")
        if learning_type == "task":
            z = torch.zeros_like(out)
            z[:,self.task_labels] = 1
            out = out*z
            out[out == 0] = -np.inf
        out = out.argmax(dim=-1)
        return ((out == labels).sum().item())

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

###################################################################################
import gensim.downloader

class Word2Vec(Base):
    def __init__(
        self,
        model,
        class_names,
        checkpoints_path: str = "/nethome/bdevnani3/raid/continual/base/models/",
        epochs: int = 200,
        early_stop: bool = False
    ):
        super().__init__(model=model, checkpoints_path=checkpoints_path, epochs=epochs, early_stop=early_stop)

        self.similarity_mode = "cossim"
        self.model.logit_scale = nn.Parameter(torch.ones([], device=get_device()))
        self.class_names = class_names
        self.init_word_vectors()
        self.init_word_lookup()

    def find_dist_words(
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
                -1*((word_lookup[None, :, :] - x[:, None, :]) ** 2)
                .sum(dim=-1)
            )
        elif mode == "cossim":
            # Note: we don't need to divide by the length of x here, because it's the same for the argmax.
            # Also, it's imporant that we can get away with that for numerical stability.
            return (
                (x @ word_lookup.t()) / (word_lookup.norm(dim=-1)[None, :] + eps)
            )
        elif mode == "dot":
            return (x @ word_lookup.t())
        else:
            raise NotImplementedError
            # .argmin(dim=-1)

    def init_word_vectors(self):
        self.word_vectors = gensim.downloader.load(name="word2vec-google-news-300")

    def init_word_lookup(self):

        # Note: we store the word lookup in the model, not the dataset because
        #   1.) The word lookup should be on the same device as the model
        #   2.) If using multiple GPUs, the model will get duplicated to each device, but the dataset won't
        #   3.) The word model (i.e., textual feature encoder) is a property of the model not the dataset
        cn = self.class_names
        for i in range(len(cn)):
            if cn[i] == 'aquarium_fish':
                cn[i] = 'fish'
            if cn[i] == "sweet_pepper":
                cn[i] = 'bell_pepper'
        self.model.word_lookup = torch.from_numpy(
            np.stack([self.word_vectors[_class] for _class in cn])
        ).to(get_device())

    def num_correct_preds(self, outputs, labels, learning_type):
        out = self.find_dist_words(
                self.model.word_lookup, outputs, mode="cossim")
        if learning_type == "task":
            z = torch.zeros_like(out)
            z[:,self.task_labels] = 1
            out = out*z
            out[out == 0] = -np.inf
        out = out.argmax(dim=-1)
        return ((out == labels).sum().item())

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
