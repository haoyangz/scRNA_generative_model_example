import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd


# https://github.com/scverse/scvi-tools/blob/7effcfb21468d272809c3bec85fea05aa9740db9/src/scvi/module/_utils.py#L3
def broadcast_labels(o, n_broadcast=-1):
    """Utility for the semi-supervised setting.

    If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
    If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other
    arguments if not None)
    """
    ys_ = torch.nn.functional.one_hot(
        torch.arange(n_broadcast, device=o.device, dtype=torch.long), n_broadcast
    )
    ys = ys_.repeat_interleave(o.size(-2), dim=0)
    if o.ndim == 2:
        new_o = o.repeat(n_broadcast, 1)
    elif o.ndim == 3:
        new_o = o.repeat(1, n_broadcast, 1)
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
