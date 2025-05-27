import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from torch.linalg import multi_dot
from utils import groupby_apply



def uniformity(x, k=None):
    if k: 
        x = F.normalize(x, dim=-1)
        uni_indices = np.random.choice(x.shape[0], k, replace=False)
        return torch.cdist(x, x[uni_indices, :]).pow(2).mul(-2).exp().mean().log()
    else: 
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()


    return cdistance

#### IPL Regularization for debiasing
def IPL(item_degrees, edge_label_index, preds, device=None, coeff=1):

    batch_users = edge_label_index[0, :]
    batch_items = edge_label_index[1, :]

    scores = torch.sigmoid(preds)
    # for a particular item, sum the scores of all users who interacted with it
    c_list = groupby_apply(batch_items, scores, bins=item_degrees.shape[0]).to(device=device, dtype=torch.float)

    with np.errstate(invalid='ignore'):
        r_list = c_list/(item_degrees**(2-coeff)) 
    ipl_loss = torch.sqrt(torch.var(r_list))

    return ipl_loss

def ReSN(U, V, device=None):
    # U are user embeddings, V are item embeddings
    e = torch.ones((U.shape[0], 1)).to(device=device, dtype=torch.float)
    return (torch.norm(multi_dot((U,V.T,V,U.T,e))).pow(2))/(torch.norm(multi_dot((V, U.T, e))).pow(2))

# Losses
class AlignmentLoss(_Loss):
    def __init__(self):
        super().__init__()

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    def forward(
        self,
        paired_user,
        paired_item,
        embed_user=None,
        embed_item=None,
        regs=None,
        gammas=None,
        item_degrees=None,
        edge_label_index=None,
        mlp_embeddings=None, 
    ):
        # Assume that paired_user and paired_item are set up such that index i in each corresponds to the i-th user-item interaction
        loss = self.alignment(paired_user, paired_item)
        for r, reg in enumerate(regs):
            if reg == "IPL":
                loss += (gammas[r] * (eval(reg)(item_degrees, edge_label_index, (paired_user * paired_item).sum(dim=-1), \
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), coeff=self.bias_coef)))
            elif reg == 'ReSN':
                loss += gammas[r] * eval(reg)(embed_user, embed_item, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2

        # regularize the mlp embeddings too
        if mlp_embeddings is not None:
            for r, reg in enumerate(regs):
                loss += gammas[r] * (eval(reg)(mlp_embeddings[0]) + eval(reg)(mlp_embeddings[1])) / 2
                
        return loss

class MAWULoss(_Loss):
    def __init__(self):
        super().__init__()
 
    @staticmethod
    def alignment_margin(x, y, margin):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        cos_sim = torch.sum(x * y, dim=-1) # dot product
        angle_ui = torch.arccos(torch.clamp(cos_sim,-1+1e-7,1-1e-7)) # clipping
        angle_ui_plus_margin = angle_ui + (1 - torch.sigmoid(margin))
        angle_ui_plus_margin = torch.clamp(angle_ui_plus_margin, 0., np.pi)
        
        cos_sim_margin = torch.cos(angle_ui_plus_margin)
        
        return -cos_sim_margin.mean()

    @staticmethod
    def uniformity_dot(x, t=2):
        x = F.normalize(x, dim=-1)
        cos_sim = F.cosine_similarity(x[:,:,None], x.t()[None,:,:])
        # take lower triangular matrix
        cos_sim = torch.tril(cos_sim, diagonal=-1)
        # convert cos_sim to distance
        cos_sim = 2 - 2 * cos_sim

        return cos_sim.mul(-t).exp().mean().log()

    def forward(
        self,
        paired_user,
        paired_item,
        margin_user, 
        margin_item,
        embed_user=None,
        embed_item=None,
        margin=1.0,
        gammas=[1.0, 1.0],
        mlp_embeddings=None
    ):

        # adaptive margin
        margin = margin_user + margin_item
        
        # margin-aware alignment and weighted uniformity losses
        align_margin = self.alignment_margin(paired_user, paired_item, margin)
        uniform = gammas[0] * self.uniformity_dot(embed_user) + gammas[1] * self.uniformity_dot(embed_item)
        loss = align_margin + uniform

        return loss


class BPRLoss(_Loss):
    def __init__(self):
        super().__init__(None, None, "sum")

    def forward(
        self,
        preds,
        embed_user=None,
        embed_item=None,
        neg_ratio=1,
        regs=None,
        gammas=None,
        item_degrees=None,
        edge_label_index=None
    ):

        # will assume that preds/labels are ordered such that the ith entry of pos and neg entry are same source
        pred_sets = preds.chunk(neg_ratio + 1)
        # split the pred sets, the first chunk is positives, and then we have self.neg_ratio chunks of negative samples
        pos = pred_sets[0]
        # restack the negative samples
        neg = torch.vstack(pred_sets[1:])

        # note that if there are more than 1 neg samples, this will broadcast pos to neg
        loss = -F.logsigmoid(pos - neg).mean()
        for r, reg in enumerate(regs):
            if reg == "IPL":
                loss += (gammas[r] * (eval(reg)(item_degrees, edge_label_index, preds, \
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), coeff=self.bias_coef)))
            elif reg == 'ReSN':
                loss += gammas[r] * eval(reg)(embed_user, embed_item, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2
     
        return loss
    
class SSMLoss(_Loss):
    def __init__(self):
        super().__init__(None, None, "sum")

        self.tau = 0.2
    def forward(self,  
                preds,
                embed_user=None,
                embed_item=None,
                neg_ratio=1,
                regs=None,
                gammas=None
        ): 
        
        # will assume that preds/labels are ordered such that the ith entry of pos and neg entry are same source
        pred_sets = preds.chunk(neg_ratio + 1)
        # split the pred sets, the first chunk is positives, and then we have self.neg_ratio chunks of negative samples
        pos = pred_sets[0]
        # restack the negative samples
        neg = torch.stack(pred_sets[1:], dim=1)

        numerator = torch.exp(pos / self.tau)
        ratings = torch.cat([pos[:, None], neg], dim=1)
        denominator = torch.sum(torch.exp(ratings / self.tau), dim=1)
        loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        for r, reg in enumerate(regs):
            if reg == "IPL":
                loss += (gammas[r] * (eval(reg)(item_degrees, coeff=self.bias_coef)))
            elif reg == 'ReSN':
                loss += gammas[r] * eval(reg)(embed_user, embed_item, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            else:
                loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2
      
        return loss

class MACRLoss(_Loss): 
    def __init__(self):
        super().__init__(None, None, "sum")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set as default based on paper results 
        self.alpha = 1e-3
        self.beta = 1e-3
        # my guess is c is super sensitive to init level -- using our xavier init makes it harder to set 
        self.c = 1
        self.rubi_c = self.c * torch.ones([1]).cuda(device)

    def forward(self,  
                pairwise_user,
                pairwise_item,
                user_weights,
                item_weights,
                neg_ratio=1,
                embed_user=None,
                embed_item=None, 
                regs=None,
                gammas=None
        ): 
        
        # will assume that preds/labels are ordered such that the ith entry of pos and neg entry are same source
        user_sets = pairwise_user.chunk(neg_ratio + 1, dim=0)
        item_sets = pairwise_item.chunk(neg_ratio + 1, dim=0)
        # split the pred sets, the first chunk is positives, and then we have self.neg_ratio chunks of negative samples
        user_pos = user_sets[0]
        user_neg = torch.stack(user_sets[1:], dim=0).squeeze(0)
        # restack the negative samples
        item_pos = item_sets[0]
        item_neg = torch.stack(item_sets[1:], dim=0).squeeze(0)
     
        # get ranking scores for positive and negative samples 
        pos_scores = torch.sum(user_pos * item_pos, dim=1)
        neg_scores = torch.sum(user_neg * item_neg, dim=1)
        
        # get individual user and item scores 
        pos_item_scores = torch.matmul(item_pos, item_weights)
        neg_item_scores = torch.matmul(item_neg, item_weights)
        pos_user_scores = torch.matmul(user_pos, user_weights)
        neg_user_scores = torch.matmul(user_neg, user_weights)

        # merge pair-wise scores, and invidiual scores, through multiplcation with sigmoid
        pos_scores = pos_scores * torch.sigmoid(pos_item_scores).squeeze(1) * torch.sigmoid(pos_user_scores).squeeze(1)
        neg_scores = neg_scores * torch.sigmoid(neg_item_scores).squeeze(1) * torch.sigmoid(neg_user_scores).squeeze(1)

        # get BCE on the merged scores
        mf_loss_ori = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_scores) + 1e-10)))

        # get BCE on item scores 
        mf_loss_item = torch.mean(
            torch.negative(torch.log(torch.sigmoid(pos_item_scores) + 1e-10)) + torch.negative(
                torch.log(1 - torch.sigmoid(neg_item_scores) + 1e-10)))

        # get BCE on user scores 
        mf_loss_user = torch.mean(torch.negative(torch.log(torch.sigmoid(pos_user_scores) + 1e-10)) + torch.negative(
            torch.log(1 - torch.sigmoid(neg_user_scores) + 1e-10)))

        loss = mf_loss_ori + self.alpha * mf_loss_item + self.beta * mf_loss_user

        if regs: 
            for r, reg in enumerate(regs):
                # wont apply other debiasing methods here 
                loss += gammas[r] * (eval(reg)(embed_user) + eval(reg)(embed_item)) / 2

        return loss
    
    def predict(self, users, items, user_weights, item_weights, method='dot_prod'):
        
        item_scores = torch.matmul(items, item_weights)
        user_scores = torch.matmul(users, user_weights)

        if method == 'dot_prod':
            rate_batch = torch.matmul(users, items.T)
        elif method == 'cosine_sim':
            x_user_norm = users / users.norm(dim=1)[:, None]
            x_item_norm = items / items.norm(dim=1)[:, None]
            rate_batch = torch.mm(x_user_norm, x_item_norm.transpose(0,1))

        #rint(rate_batch[0, :], ((torch.sigmoid(user_scores)) * (torch.sigmoid(item_scores)).T)[0, :])
        rubi_rating_both = (rate_batch - self.rubi_c) * ((torch.sigmoid(user_scores)) * (torch.sigmoid(item_scores)).T)
       
        return rubi_rating_both.cpu().detach()