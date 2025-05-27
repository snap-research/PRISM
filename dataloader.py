import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric import seed_everything
from torch_geometric.datasets import AmazonBook, MovieLens1M, MovieLens100K
from torch_geometric.loader import LinkLoader, LinkNeighborLoader
from torch_geometric.utils import bipartite_subgraph, sort_edge_index
from utils import ALL_DATASET_ATTRS, str2bool


class Data:
    def __init__(
        self,
        dataset,
        dataset_save_path,
        split_save_folder,
    ):
        # unpack necessary args
        self.dataset = dataset
        self.dataset_save_path = dataset_save_path
        self.split_save_folder = split_save_folder

        
        # different datasets use different key terms to define their attributes, reassign for easier access
        dataset_attrs = ALL_DATASET_ATTRS[dataset]
        self.user = dataset_attrs["user"]
        self.action = dataset_attrs["action"]
        self.item = dataset_attrs["item"]

        # some datasets will be undirected by having the reversed edges already saved, usually as "{action}_by"
        if "inverse_action" in dataset_attrs:
            self.inverse_action = dataset_attrs["inverse_action"]
        else:
            self.inverse_action = None

        # set up any transforms to the dataset
        self.transforms = []
        if self.inverse_action is None:
            self.transforms.append(T.ToUndirected())
            # the ToUndirected transform populates an additional param in the data argument with this label
            self.inverse_action = f"rev_{self.action}"

        self.transforms = T.Compose(self.transforms)

    def load_and_split_data(self, device="cpu", train_val_test_split=[0.8, 0.1, 0.1]):
        # load dataset and use index to unpack the dataset object since its a single graph
        # gowalla and yelp are custom datsets, will have them follow synth pattern
        if (
            "synth" in self.dataset
            or self.dataset == "Gowalla"
            or self.dataset == "Yelp2018"
            or self.dataset == "Anime"
        ):
            data = torch.load(self.dataset_save_path).to(device)
        else:
            # for dataset built into pyg, it will simply download if not available/ran on google cloud
            data = eval(self.dataset)(root=self.dataset_save_path)[0].to(device)

        self.data = data
       
        # in feature-less data, num nodes is populated, otherwise it is not and needs to be
        if "x" in data["user"]:
            self.data[self.user].num_nodes = self.data[self.user].x.shape[0]
            self.data[self.item].num_nodes = self.data[self.item].x.shape[0]

        self.num_users = self.data[self.user].num_nodes
        self.num_items = self.data[self.item].num_nodes
        print(self.num_users, self.num_items)
        self.num_classes = 2

        # if there are no features, there is a chance there will be no x-dict, we will allocate empty vectors
        # if there are feature vectors, we will want to clear them out
        self.data[self.user].x = torch.Tensor([])
        self.data[self.item].x = torch.Tensor([])

        # most datasets include the edge_label_index as a natural test set, since we will do the split ourselves
        # we will add back into the dataset to align to datasets without edge_label_index
        if "edge_label_index" in self.data[(self.user, self.action, self.item)]:
            self.data[(self.user, self.action, self.item)].edge_index = torch.cat(
                (
                    self.data[(self.user, self.action, self.item)].edge_index,
                    self.data[(self.user, self.action, self.item)].edge_label_index,
                ),
                dim=1,
            )

            # drop edge_label_index now that it has been integrated in
            del self.data[(self.user, self.action, self.item)].edge_label_index

            # the inverse edges are now wrong and missing those we just added in
            del self.data[(self.item, self.inverse_action, self.user)]

            # need to add reverse edges back in, to do this we will undirect the graph and reset labeling
            transform = T.ToUndirected()
            # create the new inverse and rename the inverse action for the dataloader to use
            self.data = transform(self.data)
            self.inverse_action = f"rev_{self.action}"
        else:
            self.data = self.transforms(self.data)

        # Random splitting into train, val, test. Note we do not add neg samples here
        transform = T.RandomLinkSplit(
            num_val=train_val_test_split[1],
            num_test=train_val_test_split[2],
            disjoint_train_ratio=0.0,
            neg_sampling_ratio=0.0,
            add_negative_train_samples=False,
            edge_types=(self.user, self.action, self.item),
            rev_edge_types=(self.item, self.inverse_action, self.user),
            key="link",
        )

        # apply transform
        train_data, val_data, test_data = transform(self.data)

        torch.save(train_data, self.split_save_folder + f"train_{seed}.pt")
        torch.save(val_data, self.split_save_folder + f"val_{seed}.pt")
        torch.save(test_data, self.split_save_folder + f"test_{seed}.pt")

        print("Cached Train/Val/Test, closing.")

    def get_dataloaders(
        self,
        device,
        num_layers=2,
        num_neighs=-1,
        batch_size=32,
        testing=False,
        seed=123,
        degree_init_str=1
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_data = torch.load(self.split_save_folder + f"train_{seed}.pt").to(
            device
        )
        val_data = torch.load(self.split_save_folder + f"val_{seed}.pt").to(device)
        test_data = torch.load(self.split_save_folder + f"test_{seed}.pt").to(
            device
        )

     
        # if testing, we do not need to set up train loader
        if testing:
            return train_data, val_data, test_data
        else:

            # get the degree of the user and item nodes
            
            # if training we will set up a subgraph loader
            train_edge_label_index = train_data[(self.user, self.action, self.item)][
                "link_index"
            ]
            
            num_users = train_data[self.user].num_nodes
            num_items = train_data[self.item].num_nodes
            
            user_counts = torch.zeros(num_users, device=device)
            item_counts = torch.zeros(num_items, device=device)
            
            user_counts[:train_edge_label_index[0, :].max() + 1] = torch.bincount(train_edge_label_index[0, :])
            item_counts[:train_edge_label_index[1, :].max() + 1] = torch.bincount(train_edge_label_index[1, :])
            
            user_weight = (degree_init_str*torch.log(user_counts + 2)) + (1-degree_init_str)*torch.ones(user_counts.shape).to(device)
   
            
            item_weight = (degree_init_str*torch.log(item_counts + 2)) + (1-degree_init_str)*torch.ones(item_counts.shape).to(device)
      
            
            train_loader = LinkNeighborLoader(
                data=train_data,
                num_neighbors=[num_neighs] * num_layers,
                edge_label_index=(
                    (self.user, self.action, self.item),
                    train_edge_label_index,
                ),
                batch_size=batch_size,
                shuffle=False,
            )

            return train_loader, train_data, val_data, user_weight, item_weight, user_counts, item_counts
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_save_path", type=str, default="./datasets")
    parser.add_argument("--num_layers", type=int, default=3)
    args = parser.parse_args()

    # this is used to populate datasets with seeded train/val/test splits
    seeds = [123, 246, 492]
    datasets = ["MovieLens1M"]

    for seed in seeds:
        seed_everything(seed)
        for dataset_name in datasets:

            split_save_folder = args.dataset_save_path + "/" + dataset_name + "/"
            # Set up save folders
            
            if dataset_name == "Gowalla" or dataset_name == "Yelp2018" or dataset_name == "Anime":
                dataset_save_path = (
                    args.dataset_save_path + "/" + dataset_name + "/data.pt"
                )
            else:
                dataset_save_path = args.dataset_save_path + "/" + dataset_name

            # dataset object holds the additional info to index into data object
            dataset = Data(
                dataset_name,
                dataset_save_path,
                split_save_folder,
            )

            # loads the actual data and does light pre-processing
            dataset.load_and_split_data()
