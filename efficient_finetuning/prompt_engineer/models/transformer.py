import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from .expt_template import Base
from .utilities import MaxPool1dFixed, AvgPool1dFixed

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.nn.functional as F
from torch.nn import MultiheadAttention, Linear, Dropout, LayerNorm
from typing import Optional, Any, Union, Callable
from torch import Tensor

# class TransformerEncoderLayer(nn.Module):
#     """ Taken from pytorch 1.7 branch on github, made some small changes."""
#     __constants__ = ['batch_first', 'norm_first']

#     def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  device=None, dtype=None, kdim=None, vdim=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(TransformerEncoderLayer, self).__init__()
#         if kdim == None:
#             kdim = d_model
#         if vdim == None:
#             vdim = d_model

#         self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
#         # Implementation of Feedforward model
#         self.linear1 = Linear(d_model, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         self.linear2 = Linear(dim_feedforward, d_model) # Bhavika changed this

#         self.norm_first = norm_first
#         self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
#         self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)

#         # Legacy string support for activation function.
#         if isinstance(activation, str):
#             self.activation = _get_activation_fn(activation)
#         else:
#             self.activation = activation

#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer, self).__setstate__(state)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """

#         # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

#         x = src
#         if self.norm_first:
#             x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
#             x = x + self._ff_block(self.norm2(x))
#         else:
#             x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
#             x = self.norm2(x + self._ff_block(x))

#         return x

#     # self-attention block
#     def _sa_block(self, x: Tensor,
#                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         x, attn = self.self_attn(x, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=True)

#         self.attention_matrix = attn
#         return self.dropout1(x)

#     # feed forward block
#     def _ff_block(self, x: Tensor) -> Tensor:
#         x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         return self.dropout2(x)

class Transformer(nn.Module):
    def __init__(self, embedding_dim=1024, feedforward_dim=512, reduction_factor=2, kdim=None, vdim=None, n_layers=1, n_heads=1):
        super().__init__()

        d_model = int(embedding_dim/reduction_factor)

        if kdim == -1:
            kdim = None
        if vdim == -1:
            vdim = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dim_reducer = nn.Linear(embedding_dim, d_model)
        self.logit_scale = nn.Parameter(torch.ones([], device=device))

    def forward(self, input):
        output = input
        output = self.dim_reducer(input)
        output = output.permute(1,0,2)  # Multi headed attention expects batch to be the second dimension. See https://github.com/pytorch/pytorch/blob/1d90f29f144a8423c6231bf625261e291b4aa02f/torch/nn/functional.py#L4956 and https://github.com/pytorch/pytorch/blob/cdf93b03de7ffc33e2762f7fefb8f3c7586fd051/torch/nn/modules/activation.py#L942
        output = self.transformer_encoder(output)
        output = output.permute(1,0,2)
        return output

class TransformerExpt(Base):
    def __init__(self,n_shot=5, lr=0.003, feedforward_dim=512, reduction_factor=2, kdim=None, vdim=None, n_layers=1, reduction="avg", n_heads=1):
        print("Initializing Transformer Expt")

        super().__init__()

        self.model = Transformer(feedforward_dim=feedforward_dim, reduction_factor=reduction_factor, kdim=kdim, vdim=vdim, n_layers=n_layers, n_heads=n_heads).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # to make calculation of loss and accuracy comparable
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[250, 350, 450], gamma=0.1
        )
        self.cossim = torch.nn.CosineSimilarity()
        self.softmax = torch.nn.Softmax(dim=1)
        self.max_pool = MaxPool1dFixed(n_shot)
        self.avg_pool = AvgPool1dFixed(n_shot)
        self.reduction = reduction

    def calc_loss(self, prototypes, support_labels, queries, query_labels, reduction="avg"):
        """
        prototypes: NWAY*NSHOT, 512
        queries: NWAY*NQueries, 512
        query_labels: NWAY*NQueries
        """
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * queries @ prototypes.t() # NWAY*NQueries, NWAY*NSHOT
        # logits_per_image = queries @ prototypes.t()
        probabilities = self.softmax(logits_per_image)

        #Assuming that relative sorting of both support labels are grouped by label, hence maxpooling makes sense
        assert all(torch.eq(support_labels,torch.sort(support_labels)[0]))

        if reduction == "max":
            probabilities_collapsed = self.max_pool(probabilities)
        if reduction == "avg":
            probabilities_collapsed = self.avg_pool(probabilities)

        return self.criterion(probabilities_collapsed, query_labels)

    def num_correct_preds(self, prototypes, support_labels, queries, query_labels, reduction="avg"):
        probabilities = queries @ prototypes.t()
        if reduction == "max":
            probabilities_collapsed = self.softmax(self.max_pool(probabilities))
        if reduction == "avg":
            probabilities_collapsed = self.softmax(self.avg_pool(probabilities))
        return torch.sum(probabilities_collapsed.argmax(dim=1) == query_labels).item()

    def train_loop(self, train_loader, epoch):

        """
        class_prototype_calculation: Defines method by which we will be calculating loss/accuracy
            "max": take argmax of all prototypes and predict for corresponding label
            "mean": takes mean of all the prototypes
 
        Doesn't use support labels because they have already been used in Task Sampler to generate the appended img text embedding
        """

        train_loss = 0.0
        total = 0
        correct = 0

        # import pdb; pdb.set_trace()
        for i, (
            support_inputs, # N_WAY x N_SHOT , 1024 - tensor
            support_labels, # N_WAY x N_SHOT , 1 - tensor
            query_inputs, # N_WAY x N_QUERY , 1024 - tensor
            query_labels, # N_WAY x N_QUERY , 1 - tensor
            true_class_ids, # N_WAY  - list
        ) in enumerate(train_loader):
            # import pdb; pdb.set_trace()

            support_inputs = support_inputs.to(device)
            support_labels = support_labels.to(device)
            query_inputs = query_inputs.to(device)
            query_labels = query_labels.to(device)

            support_labels_sorted, indices = torch.sort(support_labels)
            support_inputs_sorted = support_inputs[indices]

            self.model.train()
            loss= 0
            
            self.optimizer.zero_grad()
            class_prototype_predictions = self.model(support_inputs_sorted.unsqueeze(0)).squeeze(0) # 1, N_WAY x N_SHOT, 1024
            
            query_image_embs = query_inputs[:,:512]
            loss = self.calc_loss(class_prototype_predictions, support_labels_sorted, query_image_embs, query_labels.cuda(),reduction=self.reduction)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            total += query_labels.size(0)

            corr = self.num_correct_preds(class_prototype_predictions, support_labels_sorted, query_image_embs, query_labels.cuda(), reduction="max")
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

                support_labels_sorted, indices = torch.sort(support_labels)
                support_inputs_sorted = support_inputs[indices]

                self.model.eval()
                loss= 0
                    
                class_prototype_predictions = self.model(support_inputs_sorted.unsqueeze(0)).squeeze(0)  # N_WAY x N_SHOT, 512

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
