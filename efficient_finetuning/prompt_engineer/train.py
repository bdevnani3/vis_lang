import json
import numpy as np
import os
import torch

from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

from data.datasets import *
from data.datasets_pe import *
from data.utils import SmallDataset
from models.finetune_lambda import FinetuneLambda
from models.mlp import MLPExpt
from models.transformer import TransformerExpt
from utils import *


import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(train_configs, n_way, n_shot, n_episodes):

    model_name = train_configs["model"]
    n_epochs = train_configs["n_epochs"]

    # Initialize
    max_acc = 0
    losses = []
    accuracy = []

    # Model Selection
    if model_name == "lambda":
        lr = train_configs["lambda"]["lr"]
        model_expt = FinetuneLambda(lr=lr)

    elif model_name == "mlp":
        lr = train_configs["mlp"]["lr"]
        model_expt = MLPExpt(lr=lr, n_shot=n_shot)

    elif model_name == "transformer_encoder":
        lr = float(train_configs["transformer_encoder"]["lr"])
        feedforward_dim = int(
            train_configs["transformer_encoder"]["feedforward_dim"]
        )
        reduction_factor = int(
            train_configs["transformer_encoder"]["reduction_factor"]
        )
        kdim = train_configs["transformer_encoder"]["kdim"]
        vdim = train_configs["transformer_encoder"]["vdim"]
        n_layers = int(train_configs["transformer_encoder"]["n_layers"])
        loss_reduction = train_configs["transformer_encoder"]["loss_reduction"]
        n_heads = train_configs["transformer_encoder"]["n_heads"]
        model_expt = TransformerExpt(
            lr=lr,
            n_shot=n_shot,
            n_layers=n_layers,
            reduction_factor=reduction_factor,
            feedforward_dim=feedforward_dim,
            kdim=kdim,
            vdim=vdim,
            reduction=loss_reduction,
            n_heads=n_heads
        )

    # Generate Checkpoint Directory
    str_model_config = generate_config_string(train_configs[model_name])
    expt_name = f"/{train_dataset_name}_{model_name}_{n_way}way_{n_shot}shot_{n_episodes}episodes_{n_epochs}epochs_{str_model_config}"

    model_path = Path(model_save_dir + expt_name)
    model_path.mkdir(parents=True, exist_ok=True)

    results_path = Path(results_save_dir + expt_name)
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving model at {model_path} and results at {results_path}")

    # Train Loop
    print(
        "................................Started Training................................"
    )
    for epoch in tqdm(range(n_epochs)):
        dataloader = meta_learning_dataset.get_training_epoch(
            n_episodes, n_way, n_shot
        )
        epoch_loss, epoch_accuracy = model_expt.train_loop(dataloader, epoch)

        losses.append(epoch_loss)
        accuracy.append(epoch_accuracy)

        best_acc = max(accuracy)

        print(
            f"Training: Epoch {epoch} || Loss: {epoch_loss:7.3f} || Accuracy: {epoch_accuracy:7.3f}% || Best Accuracy: {best_acc:7.3f}%"
        )

        state = {
            "net": model_expt.model.state_dict(),
            "acc": epoch_accuracy,
            "epoch": epoch,
            "loss": epoch_loss,
        }

        if max_acc < epoch_accuracy:
            max_acc = epoch_accuracy
            path = os.path.join(model_path, f"best_acc.pth")
            print(f"Saving model at {path} as it has best accuracy so far.")
            torch.save(state, path)

        if epoch % 50 == 0:
            path = os.path.join(model_path, f"last_model.pth")
            print(f"Saving model at {path} due to epoch idx % 50 == 0.")
            torch.save(state, path)

            filename = "plots_raw.json"
            path = os.path.join(results_path, filename)

            print(f"Saving data at {path}")

            data = {
                "Train Loss": losses,
                "Train Acc": accuracy,
                "Best Train Acc": max(accuracy),
            }

            with open(path, "w") as f:
                json.dump(data, f)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-cf", "--config_file", dest="train_configs")
    args = parser.parse_args()

    general_configs = parse_yaml("configs/general_configs.yaml")
    model_save_dir = general_configs["model_save_dir"]
    results_save_dir = general_configs["results_save_dir"]

    train_configs = parse_yaml(args.train_configs)

    train_dataset_name = train_configs["train_datasets"]

    # Load data
    if train_dataset_name == "flowers":
        train_dataset_obj = Flowers102Clip()
    elif train_dataset_name == "cifar100":
        train_dataset_obj = Cifar100Clip()
    elif train_dataset_name == "cifar10":
        train_dataset_obj = Cifar10Clip()

    # Load training configs
    n_ways = train_configs["n_ways"]
    n_shots = train_configs["n_shots"]
    n_episodes_s = train_configs["n_episodes"]

    print(train_configs)

    print("Loading Data")
    meta_learning_dataset = SmallDataset(train_dataset_obj)

    for n_way in n_ways:
        for n_shot in n_shots:
            for n_episodes in n_episodes_s:
                train(train_configs, n_way, n_shot, n_episodes)
