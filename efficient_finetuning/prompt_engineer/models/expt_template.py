import torch.nn as nn
import torch
import time

class Base():
    def __init__(self):
        # super().__init__()
        pass

    def calc_loss(self, outputs, y):
        pass

    def num_correct_preds(self, outputs, y):
        pass

    def train_loop(self, train_loader):

        train_loss = 0.0
        total = 0
        correct = 0

        start_time = time.time()
        for i, (x,y,att) in enumerate(train_loader):
            loss= 0
            self.n_query = x.size(1) - self.n_support 
            self.optimizer.zero_grad()

            outputs = self.model(x)
            loss = self.calc_loss(outputs, y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += y.size(0)
            correct += self.num_correct_preds(outputs, y)

    def evaluate(self, train_embeddings, train_labels, test_embeddings, test_labels):

        with torch.no_grad():

            train_labels_sorted, indices = torch.sort(train_labels)
            train_inputs_sorted = train_embeddings[indices]

            self.model.eval()
            loss= 0
                
            class_prototype_predictions = self.model(train_inputs_sorted) # N_WAY x N_SHOT, 512

            test_image_embs = test_embeddings[:,:512]
            loss = self.calc_loss(class_prototype_predictions, train_labels_sorted, test_image_embs, test_labels)
            test_loss = loss.item()
            total = test_labels.size(0)

            corr = self.num_correct_preds(class_prototype_predictions, train_labels_sorted, test_image_embs, test_labels)

            epoch_loss = test_loss/total
            epoch_accuracy = corr*100/total
            print(
                f"Accuracy: {epoch_accuracy:7.3f}%"
            )
            return (epoch_loss, epoch_accuracy)
