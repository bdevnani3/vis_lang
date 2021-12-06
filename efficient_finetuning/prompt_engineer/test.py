import json
import numpy as np
import os
import time
import torch

from pathlib import Path


from data.task_sampler import CLIPTaskSampler
from data.datasets import *

from models.finetune_lambda import FinetuneLambda
from models.mlp import MLPExpt
from models.transformer import TransformerExpt

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"

N_WAY = 5  # 5 way
N_SHOT = 4  # 5 shot
N_QUERY = 27  - N_SHOT # number or queries in the validation set
N_EPISODES = 10
N_EPOCHS = 500
TRAIN_DATASET = "flowersFSLTrain"
TEST_DATASET = "flowersFSLTest"
PHRASE = "This is a {} with petals"
MODEL = "Transformer"
# MODEL = "MLP"


CHECKPOINTS_DIR = "/nethome/bdevnani3/raid/models/PE/"


if __name__ == "__main__":

    print("................................Loading Data................................")

    # Load data
    if TEST_DATASET == "flowersFSLTest":
        test_dataset_obj = Flowers102FSLTest(4, 5)
    elif TEST_DATASET == "flowers":
        test_dataset_obj = Flowers102(4, 5)

    clip_model, clip_preprocess = clip.load("ViT-B/32", device)

    test_dataset, _ = test_dataset_obj.get_all_data_loader(transform_fn=clip_preprocess)
    test_dataset.labels = np.asarray(test_dataset.targets)
    test_task_sampler = CLIPTaskSampler(
        dataset=test_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        n_tasks=100,
        clip_model=clip_model,
        phrase=PHRASE,
        testing=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_sampler=test_task_sampler,
        num_workers=0,
        collate_fn=test_task_sampler.episodic_collate_fn,
    )

    # Generate Checkpoint Directory

    expt_path = Path(CHECKPOINTS_DIR + f"{TRAIN_DATASET}/{MODEL}/nway_{N_WAY}_nshot_{N_SHOT}_nquery_{N_QUERY}_nepisodes_{N_EPISODES}_nepochs_{N_EPOCHS}/test/{TEST_DATASET}")
    expt_path.mkdir(parents=True, exist_ok=True)
    print(f"The directory is {expt_path}")

    # Initialize
    max_acc = 0
    losses = []
    accuracy = []

    # Model Selection
    if MODEL == "FinetuneLambda":
        model = FinetuneLambda()
    elif MODEL == "MLP":
        model = MLPExpt()
    elif MODEL == "Transformer":
        model = TransformerExpt()

    best_model_path = Path(CHECKPOINTS_DIR + f"{TRAIN_DATASET}/{MODEL}/nway_{N_WAY}_nshot_{N_SHOT}_nquery_{N_QUERY}_nepisodes_{N_EPISODES}_nepochs_{N_EPOCHS}/best_acc.pth")
    model.model.load_state_dict(torch.load(best_model_path)["net"])

    # Train Loop
    print("................................Started Testing................................")
    for epoch in range(1):            
        epoch_loss, epoch_accuracy, accuracies = model.test_loop(test_dataloader, epoch)

    filename = "plots_raw.json"
    path = os.path.join(expt_path, filename)

    print(f"Saving data at {path}")

    data = {
        "Test Losses": epoch_loss,
        "Test Accs": epoch_accuracy,
        "Test Acc Mean": np.mean(accuracies),
        "Test Acc Std": np.std(accuracies),
    }

    print(data)

    with open(path, "w") as f:
        json.dump(data, f)
                