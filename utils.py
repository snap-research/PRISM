import argparse
import json
import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
import tqdm
from numpy import linalg as LA
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.utils import (
    degree,
    k_hop_subgraph,
    remove_self_loops,
    to_dense_adj,
    to_networkx,
)

ALL_DATASET_ATTRS = {
    "MovieLens1M": {
        "user": "user",
        "action": "rates",
        "item": "movie",
        "inverse_action": "rated_by",
        "label": "rating",
    },
    "AmazonBook": {
        "user": "user",
        "action": "rates",
        "inverse_action": "rated_by",
        "item": "book",
    },
}



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_param_files(
    folder_name,
    seed_filter=None,
    hidden_dim_filter=None,
    loss_filter=None,
    batch_filter=None,
    best_hyperparam=False,
    get_intermed=False,
    use_warm_start_epoch=False,
):

    files = os.listdir(folder_name)
    param_files = [f for f in files if f.endswith("params")]

    file_details = {}

    best_ndcg = 0
    best_model_save_path = None

    
    for param_file in param_files:
 
        with open(folder_name + "/" + param_file, "r") as f:
            data = json.load(f)

            seed = data["seed"]
            if "loss" not in data:
                continue
            loss_func = data["loss"]
            hidden_dim = data["hidden_dim"]

            model_save_path = folder_name + '/' + param_file.replace("params", "pt")
       
          
            # option to filter on models that were trained with a particular seed, loss, or hidden dim
            if seed_filter:
                if seed != seed_filter:
                    continue
            if loss_filter:
                if loss_func != loss_filter:
                    continue
            if hidden_dim_filter:
                if hidden_dim != hidden_dim_filter:
                    continue

            if batch_filter:
                if data["batch_size"] != batch_filter:
                    continue
            file_details[model_save_path] = data

    file_names = list(file_details.keys())
    file_names.sort()

    return file_names, file_details


def get_processed_params(save_path, dataset, model, loss, regs):

    model_folder = f"{save_path}/{dataset}/{model}_{loss}"
    model_folder += f"_{regs}" if regs else ""

    print(model_folder)
    # if return_best, we will return only the best model, rather than all
    param_dict_processed = set()
    # havent had the chance to allocate this folder yet, will get allocated later
    if os.path.exists(model_folder):
        experiments = os.listdir(model_folder)
    else:
        return param_dict_processed

    experiments = [exp for exp in experiments if exp.endswith(".params")]

    for experiment in experiments:
        data = json.load(open(f"{model_folder}/{experiment}"))

        params = [
            data["seed"],
            data["num_layers"],
            data["hidden_dim"],
            data["lr"],
            data["weight_decay"],
            data["neg_ratio"],
            data["loss"],
            data["reg_types"],
            data["gamma_vals"],
            data["batch_size"],
            data["batched_wd"],
            data["degree_init"],
            float(data["degree_init_str"])
        ]

        param_dict_processed.add(tuple(params))

    return param_dict_processed

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

def groupby_apply(keys, values, bins):

    uniques, counts = keys.unique(return_counts=True)
    groups = torch.stack([torch.sum(item) for item in torch.split_with_sizes(values, tuple(counts))])
    reduced = torch.zeros(bins, dtype=values.dtype, device=values.device).scatter(dim=0, index=uniques, src=groups)

    return reduced