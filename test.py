import argparse
import json
import os
import sys
from datetime import datetime
from typing import overload

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from dataloader import Data
from evaluation_metrics import *
from models import RecModel
from torch.optim import Adam
from torch_geometric import seed_everything
from torch_geometric.utils import sort_edge_index
from tqdm import tqdm
from utils import *

def stable_rank(x):
    x = F.normalize(x, dim=-1)
    return (
        (torch.linalg.matrix_norm(x, ord="fro").pow(2))
        / (torch.linalg.matrix_norm(x, ord=2).pow(2))
    )


def alignment(x, y):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(2).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


class Tester:
    def __init__(self, model, model_type, device, dataset):

        self.model = model
        self.model.to(device)
        self.model_type = model_type
        self.device = device
        self.user_attr = dataset.user
        self.action_attr = dataset.action
        self.item_attr = dataset.item

    def eval_model(self, train_data, val_data, test_data, batch_size=-1, k=-1, method='dot_prod'):

        with torch.no_grad():
            test_data.to(self.device)

            # get the train/val indices for interactions, will remove these from rec matrix
            train_label_idx = train_data[
                self.user_attr, self.action_attr, self.item_attr
            ]["link_index"]
            val_label_idx = val_data[self.user_attr, self.action_attr, self.item_attr][
                "link_index"
            ]

            # these are what we will use for performance
            test_label_idx = test_data[
                self.user_attr, self.action_attr, self.item_attr
            ]["link_index"]

            
            if self.model_type == "MLP":
                x_user, x_item = self.model.get_embeddings(test_data.x_dict)
            else:
                x_user, x_item = self.model.get_embeddings(
                    test_data.x_dict,
                    test_data.edge_index_dict,
                )

            test_preds = torch.zeros((x_user.shape[0], x_item.shape[0]))
            # The full matrix mul doesnt fit into memory in many cases, thus will sometimes need to batch it
            num_users = x_user.shape[0]
            for user_start in torch.arange(0, num_users, step=batch_size):
                if method == 'dot_prod':
                    user_preds = torch.matmul(
                        x_user[user_start : user_start + batch_size, :], x_item.T
                    )
                elif method == 'cosine_sim':
                    x_user_subset = x_user[user_start : user_start + batch_size, :]
                    x_user_subset_norm = x_user_subset / x_user_subset.norm(dim=1)[:, None]
                    x_item_norm = x_item / x_item.norm(dim=1)[:, None]
                    user_preds = torch.mm(x_user_subset_norm, x_item_norm.transpose(0,1))
                test_preds[user_start : user_start + batch_size, :] = user_preds
        
            # zero out vals from train and val
            train_sources = train_label_idx[0, :]
            train_targets = train_label_idx[1, :]
            test_preds[train_sources, train_targets] = -1 * np.inf

            val_sources = val_label_idx[0, :]
            val_targets = val_label_idx[1, :]
            test_preds[val_sources, val_targets] = -1 * np.inf

            top_k_recs = torch.topk(test_preds, k=k, dim=1).indices
            sorted_label_idx = sort_edge_index(test_label_idx)

            train_val_source = torch.cat((train_sources, val_sources, test_label_idx[0, :]), dim=0)
            train_val_targets = torch.cat((train_targets, val_targets, test_label_idx[1, :]), dim=0)
            unique_user, user_count = torch.unique(train_val_source, return_counts=True)
            unique_item, item_count = torch.unique(train_val_targets, return_counts=True)
            user_degree = dict(zip(unique_user.tolist(), user_count.tolist()))
            item_degree = dict(zip(unique_item.tolist(), item_count.tolist()))

            # if model is NeuMF, get embeddings after MLP
            #if self.model_type == 'NeuMF':
            #    x_user, x_item = self.model.get_mlp_embeddings(test_data.x_dict)
                

        return sorted_label_idx, top_k_recs, item_degree, user_degree


def main():

    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--dataset", type=str, default="MovieLens1M")
    parser.add_argument("--dataset_save_path", type=str, default="./datasets")

    # model parameters
    parser.add_argument(
        "--model", type=str, default="LGConv", choices=["LGConv", "MLP"]
    )
    parser.add_argument(
        "--model_save_path", type=str, default="./model_chkps"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--loss", type=str, default="align")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--best_hyperparam", type=str2bool, default=False)
    parser.add_argument(
        "--reg_types",
        type=str,
        default="uniformity",
        help="pass delimited string of regularization terms to be parsed",
    )

    # evaluation parameters
    parser.add_argument("--k", type=int, default=20)

    args = parser.parse_args()

    # seed everything
    seed_everything(args.seed)

    ### GET TEST PERFORMANCE ###
    # split path and get last folder
    split_path = args.model_save_path.split("/")[-1]
    
    if not os.path.isdir(f'results/{split_path}/'):
        os.makedirs(f'results/{split_path}/')

    if args.reg_types != "-1":
        results_file = f"results/{split_path}/{args.dataset}_{args.model}_{args.loss}_{args.reg_types}_all_results_{args.seed}.json"
    else:
        results_file = f"results/{split_path}/{args.dataset}_{args.model}_{args.loss}_all_results_{args.seed}.json"
    # check if results file exists
    if os.path.isfile(results_file):
        print("Already Processed")
        #sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_save_folder = args.dataset_save_path + "/" + args.dataset + "/"

    # Set up save folders
    if args.dataset == "Gowalla" or args.dataset == "Yelp2018":
        args.dataset_save_path = (
            args.dataset_save_path + "/" + args.dataset + "/data.pt"
        )
    else:
        args.dataset_save_path = args.dataset_save_path + "/" + args.dataset
    args.base_model_save_path = (
        args.model_save_path + "/" + args.dataset + "/" + args.model + "_" + args.loss
    )
 
    args.base_model_save_path += (
        "_" + args.reg_types.replace(",", "_") if args.reg_types != "-1" else ""
    )

    if not os.path.exists(args.base_model_save_path):
        os.makedirs(args.base_model_save_path)

    # load saved model if it exist, if not exit
    # to figure out where the model is saved, we will load all the params files and parse them
 
    file_names, files = parse_param_files(
        args.base_model_save_path,
        loss_filter=args.loss,
        hidden_dim_filter=args.hidden_dim,
        seed_filter=args.seed,
        best_hyperparam=args.best_hyperparam,
    )

    if len(file_names) == 0:
        # there were no files for this dataset, model, seed combo, just exit
        print(f"No models for {args.model}, {args.dataset}, {args.seed}")
        sys.exit()

    # iterate through the trained models, set up associated dataloaders, and get testing metrics
    all_results = {}
    all_results["dot_ndcg"] = []
    all_results["dot_ndcg_popular"] = []
    all_results["dot_ndcg_neutral"] = []
    all_results["dot_ndcg_unpopular"] = []


    all_results["train_loss"] = []
    all_results["val_ndcg"] = []
    all_results["model_configs"] = []
    all_results["user_embed"] = []
    all_results["item_embed"] = []
    all_results["user_deg_mag"] = []
    all_results["item_deg_mag"] = []

    if args.model == 'NeuMF':
        all_results["mlp_user_embed"] = []
        all_results["mlp_item_embed"] = []
        all_results["mlp_user_deg_mag"] = []
        all_results["mlp_item_deg_mag"] = []
        all_results["mag_after_mlp"] = []
 
    for file in file_names:

        model_configs = files[file]

        
        # unpack key info
        num_layers = model_configs["num_layers"]
        hid_dims = model_configs["hidden_dim"]

        # dataset object holds the additional info to index into data object
        dataset = Data(
            args.dataset,
            args.dataset_save_path,
            split_save_folder,
        )

        # only need test data here
        train_data, val_data, test_data = dataset.get_dataloaders(
            device, num_layers=num_layers, testing=True, seed=args.seed
        )

        num_users = train_data[dataset.user].num_nodes
        num_items = train_data[dataset.item].num_nodes
        dataset.data = train_data


        # sets up the model for us, including the encoder and decoder steps
        model = RecModel(
            dataset,
            num_users,
            num_items,
            args.model,
            hidden_dim=hid_dims,
            depth=num_layers,
            device=device,
            loss=args.loss,
            act='ReLU' if args.model in ['MLP'] else None,
        )

     
        # load model from save file
        model.load_state_dict(torch.load(file, map_location=device))

        user_embeddings = model.user_embedding.weight.cpu().detach().numpy()
        item_embeddings = model.item_embedding.weight.cpu().detach().numpy()

        
        tester = Tester(model, args.model, device, dataset)

        # get recommendation matrix
        # FOR DOT PRODUCT
        sorted_label_idx, rec_matrix, item_degree, user_degree = tester.eval_model(
            train_data, val_data, test_data, batch_size=500, k=args.k, method='dot_prod'
        )

        (
            dot_ndcg_at_k,
            _,
        ) = compute_ndcg_at_k(rec_matrix, sorted_label_idx)

        (
            dot_ndcg_popular,
            dot_ndcg_neutral,
            dot_ndcg_unpopular,
        ) = compute_ndcg_at_k_popularity(rec_matrix, sorted_label_idx, item_degree)

        # save the source nodes and user_recall_at_k
        # save all of this data for later computation
        all_results["dot_ndcg"].append(dot_ndcg_at_k)
        all_results["dot_ndcg_popular"].append(dot_ndcg_popular)
        all_results["dot_ndcg_neutral"].append(dot_ndcg_neutral)
        all_results["dot_ndcg_unpopular"].append(dot_ndcg_unpopular)

        all_results["train_loss"].append(model_configs["train_loss"])
        all_results["val_ndcg"].append(model_configs["val_ndcg"])
        all_results["model_configs"].append(model_configs)
        all_results["user_embed"].append(stable_rank(torch.Tensor(user_embeddings)).item())
        all_results["item_embed"].append(stable_rank(torch.Tensor(item_embeddings)).item())
        
        user_magnitude = torch.norm(torch.Tensor(user_embeddings), dim=1)
        item_magnitude = torch.norm(torch.Tensor(item_embeddings), dim=1)

       

        for k in user_degree.keys():
            mag = user_magnitude[k].item()
            user_degree[k] = (user_degree[k], mag) 
            
        for k in item_degree.keys():
            mag = item_magnitude[k].item()
            item_degree[k] = (item_degree[k], mag)

        all_results['user_deg_mag'].append(user_degree)
        all_results['item_deg_mag'].append(item_degree)
       
        print(f"Dot NDCG@{args.k}: {dot_ndcg_at_k}, Popular: {dot_ndcg_popular}, Neutral: {dot_ndcg_neutral}, Unpopular: {dot_ndcg_unpopular}")
    with open(results_file, "w") as f:
        json.dump(all_results, f)
     

if __name__ == "__main__":
    main()
