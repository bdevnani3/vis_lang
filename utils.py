# Utils

import os
import torch


def make_dirs(path: str):
    """ Why is this not how the standard library works? """
    path = os.path.split(path)[0]
    if path != "":
        os.makedirs(path, exist_ok=True)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_root():
    if os.path.exists("/nethome/bdevnani3/raid"):
        return "/nethome/bdevnani3/raid"
    else:
        return "."
