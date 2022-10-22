import numpy as np
import torch
import pandas as pd

def compute_embedding(backbone, data_loader, lb):
    """Compute CLS embedding and prepare for TensorBoard.
    Parameters
    ----------
    backbone : timm.models.vision_transformer.VisionTransformer
        Vision transformer. The head should be an identity mapping.
    data_loader : torch.utils.data.DataLoader
        Validation dataloader that does not apply any augmentations. Just
        casting to tensor and then normalizing.
    Returns
    -------
    embs : torch.Tensor
        Embeddings of shape `(n_samples, out_dim)`.
    imgs : torch.Tensor
        Images of shape `(n_samples, 3, height, width)`.
    labels : list
        List of strings representing the classes.
    """
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(img.cpu()[:, :, :224, :224])
        labels.extend([list(lb.classes_)[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels