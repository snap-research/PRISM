import math

import numpy as np
import torch
from sklearn.metrics import ndcg_score


def NDCG(user, num_items):
    if sum(user) == 0:
        # no relevant items in the rec, just return 0
        return 0

    # get ideal NCG based on number of items for user
    ideal_user = np.zeros(user.shape)
    ideal_user[:num_items] = 1

    idcg = 0
    for i in range(1, len(ideal_user) + 1):
        idcg = idcg + (ideal_user[i - 1] / math.log2(i + 1))

    dcg = 0
    # discounted gain will logarithmically decay score for each rec
    for i in range(1, len(user) + 1):
        dcg = dcg + (user[i - 1] / math.log2(i + 1))

    ndcg = dcg / idcg
    return ndcg


def compute_ndcg_at_k(top_k_recs, edge_index):

    ndcg_user = []

    num_edges = edge_index.shape[1]
    start = 0
    end = 1

    binary_matrix = []
    pos_len = []
    while end < num_edges:
        # while we are on the same source node, move end index up one
        if edge_index[0, start] == edge_index[0, end]:
            end += 1
        else:
            # when they are not the same, we will compute NDCG from start to end and move start up
            user = edge_index[0, start]
            items = edge_index[1, start:end].cpu().detach().numpy()
            recs = top_k_recs[user, :].cpu().detach().numpy()

            # convert recs into binary ranking list based on items
            mask = np.isin(recs, items).astype(int)
            ndcg = NDCG(mask, min(top_k_recs.shape[1], len(items)))
            ndcg_user.append(ndcg)
            start = end
            end += 1

    avg_ndcg = sum(ndcg_user) / len(ndcg_user)

    return (
        avg_ndcg,
        np.array(ndcg_user),
    )



def compute_ndcg_at_k_popularity(top_k_recs, edge_index, item_degree):

    ndcg_pop = []
    ndcg_neutral = []
    ndcg_unpop = []

    #user_to_angle = {}
    

    num_edges = edge_index.shape[1]
    start = 0
    end = 1

    # will check popularity bias metric
    sorted_occurrences = sorted(item_degree.items(), key=lambda item: item[1], reverse=True)

    # Calculate the number of elements to select (top 5%)
    # Calculate the number of elements to select for each category
    total_elements = len(sorted_occurrences)
    num_top_5_percent = math.ceil(total_elements * 0.05)
    num_top_20_percent = math.ceil(total_elements * 0.20)

    # Get the top 5% elements as a set
    popular = set(item[0] for item in sorted_occurrences[:num_top_5_percent])

    # Get the elements in the top 5-20% as a set
    neutral = set(item[0] for item in sorted_occurrences[num_top_5_percent:num_top_20_percent])

    # Get the elements in the 20-100% as a set
    unpopular = set(item[0] for item in sorted_occurrences[num_top_20_percent:])

    while end < num_edges:
        # while we are on the same source node, move end index up one
        if edge_index[0, start] == edge_index[0, end]:
            end += 1
        else:
            # when they are not the same, we will compute NDCG from start to end and move start up
            user = edge_index[0, start]
            
            pop_level_angles = []
            for p, pop_level in enumerate([popular, neutral, unpopular]):

                items = edge_index[1, start:end].cpu().detach().numpy()
                orig_num_items = len(items)
                # subsample items just at this popularity level
                items = [item for item in items if item in pop_level]

                recs = top_k_recs[user, :].cpu().detach().numpy()
                # convert recs into binary ranking list based on items
                mask = np.isin(recs, items).astype(int)
                
                ndcg = NDCG(mask, min(top_k_recs.shape[1], orig_num_items))

                if p == 0:
                    ndcg_pop.append(ndcg)
                elif p == 1:
                    ndcg_neutral.append(ndcg)
                elif p == 2:
                    ndcg_unpop.append(ndcg)
            
            #user_to_angle[user.item()] = pop_level_angles
            
            start = end
            end += 1

    avg_pop = sum(ndcg_pop) / len(ndcg_pop)
    avg_neutral = sum(ndcg_neutral) / len(ndcg_neutral)
    avg_unpop = sum(ndcg_unpop) / len(ndcg_unpop)

    return (
        avg_pop,
        avg_neutral, 
        avg_unpop
    )


