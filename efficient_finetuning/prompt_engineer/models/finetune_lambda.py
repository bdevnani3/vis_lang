import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .expt_template import Base

device = "cuda" if torch.cuda.is_available() else "cpu"

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()
        self.lam = torch.nn.Parameter(torch.tensor([-1.5]).cuda())
        self.sig = nn.Sigmoid()
        self.sig_lam = self.sig(self.lam)

    def forward(self, image_prototype, text_prototpe, images):
        sig_lam = self.sig(self.lam)
        self.sig_lam = sig_lam
        dot = sig_lam * (images @ image_prototype.T) + (1 - sig_lam) * (
            images @ text_prototpe.T
        )
        return dot

    def string(self):
        return f"Lambda: {self.sig(self.lam).item()}"

class FinetuneLambda(Base):
    def __init__(self, lr = 0.01):
        super().__init__()
        self.model = Lambda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[150, 250, 350], gamma=0.1
        )

    def generate_prototype(self, support_inputs, support_labels):
        """
        Returns average of image prototype as well as image
        support_inputs - > N_WAY x N_SHOT, 1024
        support_inputs - > 1, 1024
        """
        # with torch.autograd.set_detect_anomaly(True):
        image_prototype_dict = {}
        text_prototype_dict = {}
        count = {}
        for i in range(support_inputs.size(0)):
            label = support_labels[i].item()
            im_emb = support_inputs[i, :512].clone()
            text_emb = support_inputs[i, 512:].clone()
            if label in image_prototype_dict:
                image_prototype_dict[label] = torch.add(image_prototype_dict[label], im_emb) # 1, 512; 
            else:
                image_prototype_dict[label] = im_emb # 1, 512; 

            if label in count:
                count[label] += 1.0
            else:
                count[label] = 1.0

            text_prototype_dict[label] = text_emb  # 1, 512; text is second half, doesn't change

        image_prototypes = []
        text_prototypes = []
        for label in sorted(list(image_prototype_dict.keys())):
            image_prototypes.append(image_prototype_dict[label].unsqueeze(0)/count[label])
            text_prototypes.append(text_prototype_dict[label].unsqueeze(0))
        image_prototypes = torch.cat(image_prototypes)  # N_WAY, 512
        text_prototypes = torch.cat(text_prototypes) # N_WAY, 512

        # out = (torch.zeros(5,25).cuda() @ support_inputs) @ torch.zeros(1024,512).cuda()
        # return out, out
        return image_prototypes, text_prototypes

    def calc_loss(self, predictions, labels):
        return self.criterion(predictions, labels)

    def num_correct_preds(self, predictions, labels):
        return torch.sum(predictions.argmax(dim=1) == labels).item()

    def train_loop(self, train_loader, epoch):

        """
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
            self.model.train()
            loss= 0

            image_prototypes, text_prototypes = self.generate_prototype(support_inputs, support_labels) # N_WAY, 512; N_WAY, 512 

            query_image_embs = query_inputs[:,:512]
            
            self.optimizer.zero_grad()
            
            query_predictions = self.model(image_prototypes, text_prototypes, query_image_embs) # N_WAY x N_QUERY, N_WAY - this is because we have N_WAY x N_QUERY instances, and they fall into 1 of NWAY categories
            loss = self.calc_loss(query_predictions, query_labels.cuda())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += query_labels.size(0)

            corr = self.num_correct_preds(query_predictions, query_labels.cuda())
            correct += corr

            print(i, loss.item()/query_labels.size(0), corr/query_labels.size(0), self.model.string())

        self.scheduler.step()
        epoch_loss = train_loss/total
        epoch_accuracy = correct*100/total
        print(
            f"Training: Epoch {epoch} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:7.3f} || {self.model.string()}"
        )
        return (epoch_loss, epoch_accuracy, self.model.string())

    def test_loop(self, test_loader, epoch):
        with torch.no_grad():
            test_loss = 0.0
            total = 0
            correct = 0

            for i, (
                support_inputs, # N_WAY x N_SHOT , 1024 - tensor
                support_labels, # N_WAY x N_SHOT , 1 - tensor
                query_inputs, # N_WAY x N_QUERY , 1024 - tensor
                query_labels, # N_WAY x N_QUERY , 1 - tensor
                true_class_ids, # N_WAY  - list
            ) in enumerate(test_loader):
                self.model.eval()
                with torch.no_grad():
                    loss= 0
                    image_prototypes, text_prototypes = self.generate_prototype(support_inputs, support_labels) # N_WAY, 512; N_WAY, 512 

                    query_image_embs = query_inputs[:,:512]
                    
                    query_predictions = self.model(image_prototypes, text_prototypes, query_image_embs) # N_WAY x N_QUERY, N_WAY - this is because we have N_WAY x N_QUERY instances, and they fall into 1 of NWAY categories
                    loss = self.calc_loss(query_predictions, query_labels.cuda())
                    test_loss += loss.item()
                    total += query_labels.size(0)

                    corr = self.num_correct_preds(query_predictions, query_labels.cuda())
                    correct += corr

                    print(i, loss.item(), corr/query_labels.size(0), self.model.string())


            epoch_loss = test_loss/total
            epoch_accuracy = correct*100/total
            print(
                f"Testing: Epoch {epoch} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:7.3f} || {self.model.string()}"
            )
            return (epoch_loss, epoch_accuracy, self.model.string())

        
