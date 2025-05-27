from typing import overload

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch.nn import Embedding, LeakyReLU, Linear, ModuleList, ReLU
from torch_geometric.nn import HeteroConv, LightGCN, to_hetero
from torch_geometric.nn.conv import GCNConv, LGConv, SAGEConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj
from torch_geometric.utils import degree, to_undirected
import torch.nn.utils.weight_norm as weight_norm
import tqdm 
# All recommendation systems will be structured as:
# 1. An embedding component
#    a. Either initialize a random feature vector, or use feature vector provided
#    b. Some learning process like MLP or GNN that operates on the features
# 2. A decoding component
#    a. Takes the user and item and does something to compare


class RecModel(torch.nn.Module):
    # outer model
    def __init__(
        self,
        dataset,
        num_users,
        num_items,
        model_type,
        hidden_dim=32,
        depth=2,
        device=None,
        act=None,
        user_mag=None,
        item_mag=None,
        loss=None
    ):
        super().__init__()

        self.model_type = model_type
        self.user = dataset.user
        self.item = dataset.item
        self.action = dataset.action
        self.rev_action = dataset.inverse_action

        self.num_users = num_users
        self.num_items = num_items
        self.user_dim = hidden_dim
        self.item_dim = hidden_dim

        self.hidden_dim = hidden_dim
        self.depth = depth
        self.device = device

        # use id-based embeddings by allocating random vectors
        self.user_embedding = Embedding(self.num_users, self.hidden_dim)
        # normalize embeddings
        torch.nn.init.xavier_uniform_(self.user_embedding.weight)
        if user_mag is not None: 
            self.user_embedding.weight.data = F.normalize(self.user_embedding.weight.data, dim=-1)
            user_mag = user_mag.to(self.user_embedding.weight.device)
            self.user_embedding.weight.data = self.user_embedding.weight.data * (user_mag.view(-1, 1))

        self.item_embedding = Embedding(self.num_items, self.hidden_dim)
        # normalize embeddings
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        if item_mag is not None: 
            self.item_embedding.weight.data = F.normalize(self.item_embedding.weight.data, dim=-1)
            item_mag = item_mag.to(self.user_embedding.weight.device)
            self.item_embedding.weight.data = self.item_embedding.weight.data * (item_mag.view(-1, 1))

        
        # need to allocate margin variables
        if loss == 'MAWU':
            self.user_margin = Embedding(self.num_users, 1)
            self.item_margin = Embedding(self.num_items, 1)
            torch.nn.init.xavier_uniform_(self.user_margin.weight)
            torch.nn.init.xavier_uniform_(self.item_margin.weight)

        if loss == 'MACR': 
            # allocate item and user projections that will be used in causal graph regularization
            self.w_item = Embedding(self.hidden_dim, 1)
            self.w_user = Embedding(self.hidden_dim, 1)
            torch.nn.init.xavier_normal_(self.w_item.weight)
            torch.nn.init.xavier_normal_(self.w_user.weight)
            
        if act:
            self.act = eval(act)()
        else:
            self.act = None

        # will be for MLP two tower set up
        if self.model_type == "MLP":
            self.encoder = MLPEncoder(
                self.user_dim,
                self.item_dim,
                self.hidden_dim,
                self.depth,
                act=self.act,
            )
        # set up for more GNN architectures
        # metadata is used for HeteroConv wrapper of lightGCN. enforce it goes user, item in node attrs
        else:
            self.encoder = GNNEncoder(
                model_type,
                self.user_dim,
                self.item_dim,
                self.hidden_dim,
                self.depth,
                metadata=((self.user, self.item), dataset.data.metadata()[1]),
                act=self.act,
                num_users=self.num_users,
                num_items=self.num_items,
                user_attr=self.user,
                item_attr=self.item,
            )
        self.decoder = Decoder(self.user, self.item)

    def preprocess_features(self, x_dict=None, user_id=None, item_id=None):
        # sets up the subsampling for batching, and allocates the embeddings into the x_dict
        if user_id is None:
            user_id = torch.arange(self.num_users)
        if item_id is None:
            item_id = torch.arange(self.num_items)

        x_dict[self.user] = self.user_embedding.weight[user_id, :]
        x_dict[self.item] = self.item_embedding.weight[item_id, :]

        return x_dict

    def forward(
        self,
        x_dict,
        edge_index_dict=None,
        edge_label_index=None,
        user_id=None,
        item_id=None,
    ):

        if self.model_type == "MLP":
            x_dict = self.preprocess_features(x_dict, user_id, item_id)
            x_user, x_item = self.encoder(x_dict[self.user], x_dict[self.item])
            x_dict = {self.user: x_user, self.item: x_item}
            return self.decoder(x_dict, edge_label_index)
        else:
            x_dict = self.preprocess_features(x_dict, user_id, item_id)
            x_dict = self.encoder(x_dict, edge_index_dict)
            return self.decoder(x_dict, edge_label_index)

    def get_embeddings(
        self,
        x_dict,
        edge_index_dict=None,
        user_id=None,
        item_id=None,
    ):

        # this is used to get the full embedding set.
        if self.model_type == "MLP":
            x_dict = self.preprocess_features(x_dict, user_id, item_id)
            x_user, x_item = self.encoder(x_dict[self.user], x_dict[self.item])
            return x_user, x_item
        else:
            # user_id and item_id are the nodes that within the neighborhood of the user and items we will predict for
            x_dict = self.preprocess_features(x_dict, user_id, item_id)
            x_dict = self.encoder(x_dict, edge_index_dict)
            return x_dict[self.user], x_dict[self.item]
    
        
    def get_margins(self, 
        edge_label_index,
    ): 
        return self.user_margin.weight[edge_label_index[0, :]], self.item_margin.weight[edge_label_index[1, :]]
    
    def get_MACR_weights(self):
        return self.w_user.weight, self.w_item.weight


    def clip_margins(self):
        # clip user/item margin
        self.user_margin.weight.data = torch.clamp(self.user_margin.weight.data, min=0.0, max=1.0)
        self.item_margin.weight.data = torch.clamp(self.item_margin.weight.data, min=0.0, max=1.0)


class GNNEncoder(torch.nn.Module):
    def __init__(
        self,
        model_type,
        user_dim,
        item_dim,
        hidden_dim,
        depth,
        act=None,
        metadata=None,
        num_users=None,
        num_items=None,
        user_attr=None,
        item_attr=None,
    ):
        super().__init__()

        self.act = act
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.depth = depth
        self.model_type = model_type
        self.metadata = metadata
        self.num_users = num_users
        self.num_items = num_items
        self.user_attr = user_attr
        self.item_attr = item_attr

        self.convs = torch.nn.ModuleList()

        if model_type in ["LGConv"]:
            alpha = 1.0 / (depth + 1)
            alpha = torch.tensor([alpha] * (depth + 1))
            self.register_buffer("alpha", alpha)
            self.convs.extend([LGConv(normalize=False) for _ in range(depth)])

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):

        # unpack each direction of edges
        user_to_item = edge_index[self.metadata[1][0]].clone()
        item_to_user = edge_index[self.metadata[1][1]].clone()

        # turn into homogeneous edge index by re-indexing based on number of users (add num users to item objects)
        user_to_item[1, :] += x[self.user_attr].shape[0]
        item_to_user[0, :] += x[self.user_attr].shape[0]
        # combine the directional edges, user-to-item direction
        full_edge_index = torch.cat(
            [
                user_to_item,
                item_to_user.flip(0),
            ],
            dim=1,
        )

        # compute deg weighting for the full edge index
        full_edge_index = to_undirected(full_edge_index)
        full_edge_index, deg = gcn_norm(
            (full_edge_index),
            num_nodes=x[self.user_attr].shape[0] + x[self.item_attr].shape[0],
            add_self_loops=False,
        )

        batch_num_users = x[self.user_attr].shape[0]

        # turn x into homogeneous by concatenating into one feat matrix
        x = torch.cat([x[self.user_attr], x[self.item_attr]])
        out_user = x[:batch_num_users] * self.alpha[0]
        out_item = x[batch_num_users:] * self.alpha[0]

        for i in range(self.depth):
            x = self.convs[i](x, full_edge_index, edge_weight=deg)

            # build up representations for LGConv
            out_user = out_user + (x[:batch_num_users] * self.alpha[i + 1])
            out_item = out_item + (x[batch_num_users:] * self.alpha[i + 1])

        if self.model_type == "LGConv":
            x = {self.metadata[0][0]: out_user, self.metadata[0][1]: out_item}
        return x


class MLPEncoder(torch.nn.Module):
    def __init__(
        self,
        user_dim,
        item_dim,
        hidden_dim,
        depth,
        act=None,
    ):
        super().__init__()

        self.act = act
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.depth = depth

        self.user_mlp = []
        for i in range(depth):
            if i == 0:
                self.user_mlp.append(Linear(self.user_dim, hidden_dim))
            else:
                self.user_mlp.append(Linear(hidden_dim, hidden_dim))
        self.user_mlp = ModuleList(self.user_mlp)

        self.item_mlp = []
        for i in range(depth):
            if i == 0:
                self.item_mlp.append(Linear(self.item_dim, hidden_dim))
            else:
                self.item_mlp.append(Linear(hidden_dim, hidden_dim))

        self.item_mlp = ModuleList(self.item_mlp)

    def forward(self, x_user, x_item):

        for i in range(self.depth):
            x_user = self.user_mlp[i](x_user)
            if self.act and (i != (self.depth - 1)):
                self.act(x_user)
            x_item = self.item_mlp[i](x_item)
            if self.act and (i != (self.depth - 1)):
                self.act(x_item)

        return x_user, x_item

# General decoder for embeddings
class Decoder(torch.nn.Module):
    def __init__(self, user, item):
        super().__init__()

        self.user = user
        self.item = item

    def forward(self, x_dict, edge_label_index=None):
        x_src = x_dict[self.user][edge_label_index[0]]
        x_dst = x_dict[self.item][edge_label_index[1]]
        # element-wise product between users and items
        return (x_src * x_dst).sum(dim=-1)
