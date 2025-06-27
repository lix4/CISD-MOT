import torch
import torch.nn.functional as F

def masked_cross_entropy_loss(logits, labels, mask):
    """
    logits: [BN, L, C] – model output logits
    labels: [BN, L]    – ground truth class indices (0, 1, 2, ...)
    mask:   [BN, L]    – boolean mask (True for valid positions)

    Returns:
        Scalar loss: averaged over valid (mask=True) positions
    """
    # print(logits.shape, labels.shape, mask.shape)
    # print(logits.shape)
    BN, L, C = logits.shape
    if BN == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # labels = labels[:,-1:]
    # mask = mask[:,-1:]

    # Flatten everything
    logits_flat = logits.view(-1, C)        # [BN*L, C]
    labels_flat = labels.view(-1)           # [BN*L]
    mask_flat = mask.view(-1).bool()        # [BN*L]

    # Filter valid positions
    logits_valid = logits_flat[mask_flat]   # [N_valid, C]
    labels_valid = labels_flat[mask_flat]   # [N_valid]
    # print(logits_valid.shape)
    # Compute cross-entropy
    loss = F.cross_entropy(logits_valid, labels_valid)
    return loss
