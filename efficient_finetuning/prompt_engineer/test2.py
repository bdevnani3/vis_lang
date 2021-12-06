import json
import numpy as np
import os
import time
import torch

from pathlib import Path
from tqdm import tqdm

from data.task_sampler import CLIPTaskSampler
from data.datasets import *
from data.datasets_pe import *
from data.utils import *
from models.finetune_lambda import FinetuneLambda
from models.mlp import MLPExpt
from models.transformer import TransformerExpt
from utils import *

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"

ITER = 5

if __name__ == "__main__":

    # Load configs
    general_configs = parse_yaml("configs/general_configs.yaml")
    model_save_dir = general_configs["model_save_dir"]
    results_save_dir = general_configs["results_save_dir"]
    
    test_configs = parse_yaml("configs/experimental_configs/test_template.yaml")

    trained_model_prefix = test_configs["trained_model"]

    trained_model_path = test_configs["trained_model"]
    clip_model, clip_preprocess = clip.load("ViT-B/32", device)

    # Load Data
    if "flowers" in test_configs["test_dataset"]:
        test_dataset_obj = Flowers102Clip()
    elif "cifar100" in test_configs["test_dataset"]:
        test_dataset_obj = Cifar100Clip()
    elif "cifar10" in test_configs["test_dataset"]:
        test_dataset_obj = Cifar10Clip()

    fsl_eval_dataset = SmallDataset(test_dataset_obj,test=True)

    def parse_configs_in_expt_name(trained_model_path):
        d = {}
        configs = trained_model_path.strip.split("_")
        for config in configs:
            if "feedforward-dim" in config:
                d["ffd"] = int(config.split("feedforward-dim")[1])
            if "reduction-factor" in config:
                d["rf"] = int(config.split("reduction-factor")[1])

        return d


    # Load model
    if "mlp" in trained_model_path:
        model_class = MLPExpt
        flag_transformer = False
    elif "transformer_encoder" in trained_model_path:
        model_class = TransformerExpt
        flag_transformer = True
        configs = parse_configs_in_expt_name(trained_model_path)

    best_model_path = os.path.join(model_save_dir, trained_model_path, "best_acc.pth")
    
    shot_range = test_configs["n_shots"]

    results = {}

    for n_shot in shot_range:
        results[n_shot] = []
        if flag_transformer:
            expt = model_class(n_shot=n_shot, feedforward_dim=configs["ffd"], reduction_factor=configs["rf"])
        else:
            expt = model_class(n_shot=n_shot)
        expt.model.load_state_dict(torch.load(best_model_path)["net"])

        for _ in range(ITER):
            # train_loader, _, _, _ = train_dataset_obj.get_train_loaders(transform_fn=clip_preprocess,num_elements_per_class=n_shot)
            # train_embeddings, train_labels = get_batch_concat_embedding(train_loader, train_dataset_obj.classes, clip_model, device="cuda:0", phrase=phrase)

            dataloader = fsl_eval_dataset.get_test_episode_set(n_episodes=1, n_shot=n_shot)        
            loss, accuracy = expt.evaluate(dataloader)

            results[n_shot].append(accuracy)

    for k,v in results.items():
        acc = np.average(v)
        std = np.std(v)
        print(f"{k} shots: {acc} +- {std}")
        
