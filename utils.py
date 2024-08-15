import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd


def broadcast_labels(o, n_broadcast=-1):
    ys_ = torch.nn.functional.one_hot(
        torch.arange(n_broadcast, device=o.device, dtype=torch.long), n_broadcast
    )
    ys = ys_.repeat_interleave(o.size(-2), dim=0)
    new_o = o.repeat(n_broadcast, 1)
    return ys, new_o


def reparameterize(mu, logvar):
    logvar = torch.clip(logvar, min=np.log(1e-3), max=10) # clip for numerical stability
    std = torch.exp(0.5 * logvar)
    sample = mu + torch.randn_like(std) * std
    return mu, logvar, sample


def evaluate(target_array, pred_proba_mat, target_name_mapper=None):
    
    performance = defaultdict(dict)
    
    for ct in np.unique(target_array):
        target = (target_array==ct).astype(int)
        pred_proba = pred_proba_mat[..., ct]
        ct_name = ct if target_name_mapper is None else target_name_mapper[ct]
        performance["auprc"][ct_name] = average_precision_score(target, pred_proba)
        performance["auroc"][ct_name] = roc_auc_score(target, pred_proba)

    return pd.DataFrame(performance)
