import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from yawareloss import GeneralizedSupervisedNTXenLoss


# ----- 1) Fixed embeddings & labels -----
emb_init = torch.tensor([
    [ 2.0,  0.0,  0.1,  0.6],
    [ 1.8,  0.2,  0.0,  0.5],
    [ 5.0,  2.0,  3.0,  4.0],
    [ 1.0,  4.0,  8.0,  3.0],
    [-1.0,  2.0, 12.0,  5.5],
    [ 0.0, -2.0,  6.0,  7.0],
], dtype=torch.float32)

labels = torch.tensor([
    [1.0, 0.0,  0.0, 0.0],
    [1.0, 0.0,  0.0, 0.0],
    [1.0, 2.0,  3.0, 1.0],
    [1.0, 2.2,  3.0, 1.1],
    [6.0, 1.0,  2.5, 0.0],
    [6.5, 1.0,  2.7, 0.0],
], dtype=torch.float32)

# ----- 2) Compute initial 6×6 cosine similarity -----
with torch.no_grad():
    norm0 = F.normalize(emb_init, dim=1)
    sim0 = norm0 @ norm0.T
print("Initial similarity matrix:\n", sim0.numpy())

# ----- 3) Train with y‑aware loss -----
emb = emb_init.clone().requires_grad_(True)
opt = torch.optim.SGD([emb], lr=0.5)
loss_fn = GeneralizedSupervisedNTXenLoss(temperature=0.5, sigma=1.0)

for _ in range(50):
    zi = emb
    zj = emb + 0.05*torch.randn_like(emb)
    loss = loss_fn(zi, zj, labels)
    opt.zero_grad(); loss.backward(); opt.step()

emb_final = emb.detach()

# ----- 4) Compute final similarity -----
with torch.no_grad():
    norm1 = F.normalize(emb_final, dim=1)
    sim1 = norm1 @ norm1.T
print("\nTrained similarity matrix:\n", sim1.numpy())

# ----- 5) Plot and save heatmaps -----
fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,4))

im0 = ax0.imshow(sim0.numpy(), vmin=-1, vmax=1, cmap='coolwarm')
ax0.set_title("Before Training")
fig.colorbar(im0, ax=ax0, fraction=0.046)

im1 = ax1.imshow(sim1.numpy(), vmin=-1, vmax=1, cmap='coolwarm')
ax1.set_title("After Training")
fig.colorbar(im1, ax=ax1, fraction=0.046)

plt.tight_layout()
plt.savefig("yaware_similarity2.png", dpi=150)

