import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .expt_template import Base
from .utilities import MaxPool1dFixed

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP_1_hidden(nn.Module):
    def __init__(self, input_size = 1024, hidden_size = 512):
        super().__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.logit_scale = nn.Parameter(torch.ones([], device=device))

    def forward(self, input):
        output = self.out(self.fc1(input))
        return output


class MLPExpt(Base):
    def __init__(self,variant="MLP_1_hidden", n_shot=5, lr=0.003):

        super().__init__()

        if variant == "MLP_1_hidden":
            self.model = MLP_1_hidden().to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # to make calculation of loss and accuracy comparable
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[150, 250, 350], gamma=0.1
        )
        self.cossim = torch.nn.CosineSimilarity()
        self.softmax = torch.nn.Softmax(dim=1)
        self.max_pool = MaxPool1dFixed(n_shot)

    def calc_loss(self, prototypes, support_labels, queries, query_labels, reduction="max"):
        """
        prototypes: NWAY*NSHOT, 512
        queries: NWAY*NQueries, 512
        query_labels: NWAY*NQueries
        """
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * queries @ prototypes.t() # NWAY*NQueries, NWAY*NSHOT
        probabilities = self.softmax(logits_per_image)

        #Assuming that relative sorting of both support labels are grouped by label, hence maxpooling makes sense
        assert all(torch.eq(support_labels,torch.sort(support_labels)[0]))

        probabilities_collapsed = self.max_pool(probabilities)

        return self.criterion(probabilities_collapsed, query_labels)

    def num_correct_preds(self, prototypes, support_labels, queries, query_labels, reduction="max"):
        probabilities = queries @ prototypes.t()
        probabilities_collapsed = self.softmax(self.max_pool(probabilities))
        return torch.sum(probabilities_collapsed.argmax(dim=1) == query_labels).item()

    def train_loop(self, train_loader, epoch, class_prototype_calculation="max"):

        """

        class_prototype_calculation: Defines method by which we will be calculating loss/accuracy
            "max": take argmax of all prototypes and predict for corresponding label
            "mean": takes mean of all the prototypes
 
        Doesn't use support labels because they have already been used in Task Sampler to generate the appended img text embedding
        """

        train_loss = 0.0
        total = 0
        correct = 0

        start_time = time.time()
        for i, (
            support_inputs, # N_WAY x N_SHOT , 1024 - tensor
            support_labels, # N_WAY x N_SHOT , 1 - tensor
            query_inputs, # N_WAY x N_QUERY , 1024 - tensor
            query_labels, # N_WAY x N_QUERY , 1 - tensor
            true_class_ids, # N_WAY  - list
        ) in enumerate(train_loader):

            support_inputs = support_inputs.to(device)
            support_labels = support_labels.to(device)
            query_inputs = query_inputs.to(device)
            query_labels = query_labels.to(device)

            support_labels_sorted, indices = torch.sort(support_labels)
            support_inputs_sorted = support_inputs[indices]

            self.model.train()
            loss= 0
            
            self.optimizer.zero_grad()
            
            class_prototype_predictions = self.model(support_inputs_sorted) # N_WAY x N_SHOT, 512

            query_image_embs = query_inputs[:,:512]
            loss = self.calc_loss(class_prototype_predictions, support_labels_sorted, query_image_embs, query_labels.cuda())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += query_labels.size(0)

            corr = self.num_correct_preds(class_prototype_predictions, support_labels_sorted, query_image_embs, query_labels.cuda())
            correct += corr

            print(i, loss.item()/query_labels.size(0), corr/query_labels.size(0))

        self.scheduler.step()
        epoch_loss = train_loss/total
        epoch_accuracy = correct*100/total
        return (epoch_loss, epoch_accuracy)

    def evaluate(self, test_loader):

        for i, (
            support_inputs, # N_WAY x N_SHOT , 1024 - tensor
            support_labels, # N_WAY x N_SHOT , 1 - tensor
            query_inputs, # N_WAY x N_QUERY , 1024 - tensor
            query_labels, # N_WAY x N_QUERY , 1 - tensor
            true_class_ids, # N_WAY  - list
        ) in enumerate(test_loader):


            with torch.no_grad():

                # train_labels_sorted, indices = torch.sort(train_labels)
                # train_inputs_sorted = train_embeddings[indices]

                support_labels_sorted, indices = torch.sort(support_labels)
                support_inputs_sorted = support_inputs[indices]

                self.model.eval()
                loss= 0
                    
                class_prototype_predictions = self.model(support_inputs_sorted)  # N_WAY x N_SHOT, 512

                test_image_embs = query_inputs[:,:512]
                loss = self.calc_loss(class_prototype_predictions, support_labels_sorted, test_image_embs, query_labels.cuda())
                test_loss = loss.item()
                total = query_labels.size(0)

                corr = self.num_correct_preds(class_prototype_predictions, support_labels_sorted, test_image_embs, query_labels.cuda())

                epoch_loss = test_loss/total
                epoch_accuracy = corr*100/total
                print(
                    f"Accuracy: {epoch_accuracy:7.3f}%"
                )
                return (epoch_loss, epoch_accuracy)


