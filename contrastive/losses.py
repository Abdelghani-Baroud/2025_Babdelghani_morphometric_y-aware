#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import check_array
try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = tuple
from contrastive.utils import logs
import os
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import ListConfig
from contrastive.utils.logs import set_file_logger
import logging



def mean_off_diagonal(a):
    """Computes the mean of off-diagonal elements"""
    n = a.shape[0]
    return ((a.sum() - a.trace()) / (n * n - n))


def quantile_off_diagonal(a):
    """Computes the quantile of off-diagonal elements
    TODO: it is here the quantile of the whole a"""
    return a.quantile(0.75)


def print_info(z_i, z_j, sim_zij, sim_zii, sim_zjj, temperature):
    """prints useful info over correlations"""

    log.info("histogram of z_i after normalization:")
    log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

    log.info("histogram of z_j after normalization:")
    log.info(np.histogram(z_j.detach().cpu().numpy() * 100, bins='auto'))

    # Gives histogram of sim vectors
    log.info("histogram of sim_zij:")
    log.info(
        np.histogram(
            sim_zij.detach().cpu().numpy() *
            temperature *
            100,
            bins='auto'))

    # Diagonals as 1D tensor
    diag_ij = sim_zij.diagonal()

    # Prints quantiles of positive pairs (views from the same image)
    quantile_positive_pairs = diag_ij.quantile(0.75)
    log.info(
        f"quantile of positives ij = "
        f"{quantile_positive_pairs.cpu()*temperature*100}")

    # Computes quantiles of negative pairs
    quantile_negative_ii = quantile_off_diagonal(sim_zii)
    quantile_negative_jj = quantile_off_diagonal(sim_zjj)
    quantile_negative_ij = quantile_off_diagonal(sim_zij)

    # Prints quantiles of negative pairs
    log.info(
        f"quantile of negatives ii = "
        f"{quantile_negative_ii.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives jj = "
        f"{quantile_negative_jj.cpu()*temperature*100}")
    log.info(
        f"quantile of negatives ij = "
        f"{quantile_negative_ij.cpu()*temperature*100}")
    

class BarlowTwinsLoss(nn.Module):

    def __init__(self, device, correlation='cross', lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device
        self.correlation = correlation

    def forward(self, z_a, z_b):
        # normalize repr. along the batch dimension
        # beware: normalization is not robust to batch of size 1
        # if it happens, it will return a nan loss
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD

        N = z_a.size(0)
        D = z_a.size(1)
        lbd = self.lambda_param / D

        if self.correlation=='cross':
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            # multiply off-diagonal elems of c_diff by lambda
            c_diff[~torch.eye(D, dtype=bool)] *= lbd
            loss_invariance = c_diff[torch.eye(D, dtype=bool)].sum()
            loss_redundancy = c_diff[~torch.eye(D, dtype=bool)].sum()
            loss = loss_invariance + loss_redundancy
        elif self.correlation=='auto':
            # auto-correlation matrix
            c1 = torch.mm(z_a_norm.T, z_a_norm) / N # DxD
            c2 = torch.mm(z_b_norm.T, z_b_norm) / N # DxD
            c = (c1.pow(2) + c2.pow(2)) / 2
            c[torch.eye(D, dtype=bool)]=0
            redundancy_loss = c.sum()
            # cross-correlation matrix
            c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
            # loss
            c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
            c_diff[~torch.eye(D, dtype=bool)]=0
            loss_invariance = c_diff.sum()
            loss_redundancy = self.lbd*redundancy_loss
            loss = loss_invariance + loss_redundancy
        else:
            raise ValueError("Wrong correlation specified in BarlowTwins\
                             config: use cross or auto.")

        return(loss, loss_invariance, loss_redundancy)


class VicRegLoss(nn.Module):

    def __init__(self, device=None, lmbd=5e-3, u=1, v=1, epsilon=1e-3):
        super(VicRegLoss, self).__init__()
        self.lmbd = lmbd
        self.device = device
        self.u = u
        self.v = v
        self.epsilon = epsilon

    def forward(self, x, y):
        
        bs = x.size(0)
        emb = x.size(1)

        std_x = torch.sqrt(x.var(dim=0) + self.epsilon)
        std_y = torch.sqrt(y.var(dim=0) + self.epsilon)
        var_loss = torch.mean(func.relu(1 - std_x)) + torch.mean(func.relu(1 - std_y))

        invar_loss = func.mse_loss(x, y)

        xNorm = (x - x.mean(0)) / x.std(0)
        yNorm = (y - y.mean(0)) / y.std(0)
        crossCorMat = (xNorm.T@yNorm) / bs
        cross_loss = (crossCorMat*self.lmbd - torch.eye(emb, device=torch.device('cuda'))*self.lmbd).pow(2).sum()
        
        loss = self.u*var_loss + self.v*invar_loss + cross_loss

        return loss


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.utils.validation import check_array
import logging
from contrastive.utils.logs import set_file_logger

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = tuple


def _offdiag_np(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return M[mask]


def _entropy_rows(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P = np.clip(P, eps, 1.0)
    return -(P * np.log(P)).sum(axis=1)


def _neff(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # effective neighbors per row = exp(entropy)
    return np.exp(_entropy_rows(P, eps=eps))


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import logging

import pandas as pd
from sklearn.utils.validation import check_array

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = tuple

from contrastive.utils.logs import set_file_logger





import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as func

from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import rbf_kernel

try:
    from omegaconf import ListConfig
except Exception:
    ListConfig = tuple  # fallback


class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(
        self,
        label_csv: str,
        label_cols: list[str],
        kernel="rbf",
        temperature=0.1,
        return_logits=False,
        sigma=None,
        sigma_factor=1.0,
        sigma_mode="diag",            # "scalar" | "diag" | "full" | "cosine"
        standardize_y=False,          # ONLY applied for FULL mode
        shrinkage=0.1,                # ONLY used for FULL mode
        cosine_temperature=None       # ONLY used for cosine mode; if None uses sigma_factor
    ):
        super().__init__()
        self.original_kernel = kernel
        self.temperature = float(temperature)
        self.return_logits = return_logits
        self.INF = 1e8

        self.log = set_file_logger(__file__)
        self.log.setLevel(logging.INFO)
        self.log.propagate = True

        self.label_csv = label_csv
        self.label_cols = label_cols

        if isinstance(sigma, ListConfig):
            sigma = list(sigma)

        self.sigma = sigma
        self.sigma_factor = float(sigma_factor)

        self.sigma_mode = str(sigma_mode).lower()
        assert self.sigma_mode in ("scalar", "diag", "full", "cosine"), f"Wrong sigma_mode={sigma_mode}"

        self.standardize_y = bool(standardize_y)
        self.shrinkage = float(shrinkage)
        self.cosine_temperature = cosine_temperature

        self._y_mean = None
        self._y_std = None
        self._precision = None  # (d,d) for FULL mode

        if kernel == "rbf":
            self.kernel = self._kernel_dispatch
        else:
            assert hasattr(kernel, "__call__"), "kernel must be callable"
            self.kernel = kernel

        self._did_global_log = False
        self._logged_weight_stats = False  # <-- ADDED: log-once guard (safer than hasattr)

        self._fit_once_global()

    def _fit_once_global(self):
        df = pd.read_csv(self.label_csv)
        df = df.dropna(subset=self.label_cols)
        X_raw = check_array(df[self.label_cols].values).astype(np.float64)
        n, d = X_raw.shape# ---- SAFETY CHECKS (eval often changes label_cols) ----
        if isinstance(self.label_cols, str):
            self.label_cols = [self.label_cols]  # prevent pandas Series -> 1D values

        if n < 2:
            raise ValueError(f"[FULL FIT] Not enough samples after dropna: n={n} csv={self.label_csv}")

        if d < 1:
            raise ValueError(f"[FULL FIT] No label columns (d={d}) csv={self.label_csv}")

        self.log.info(f"[FULL FIT DEBUG] X_raw shape={X_raw.shape} | n={n} d={d}")



        self._y_mean = X_raw.mean(axis=0, keepdims=True)
        self._y_std = np.maximum(X_raw.std(axis=0, keepdims=True), 1e-8)

        if self.sigma is not None:
            if self.sigma_mode == "full":
                Sigma = np.asarray(self.sigma, dtype=np.float64)
                Sigma = np.atleast_2d(Sigma)
                if Sigma.ndim != 2 or Sigma.shape != (d, d):
                    raise ValueError(f"sigma_mode=full expects sigma as DxD matrix with D={d}, got {Sigma.shape}")

                a = self.shrinkage
                scale = float(np.trace(Sigma) / d)
                Sigma_reg = (1.0 - a) * Sigma + a * scale * np.eye(d)
                Sigma_reg = (self.sigma_factor ** 2) * Sigma_reg
                self._precision = np.linalg.pinv(Sigma_reg)

            elif self.sigma_mode == "scalar":
                self.sigma = float(self.sigma)

            elif self.sigma_mode == "diag":
                self.sigma = np.asarray(self.sigma, dtype=np.float64)
                if self.sigma.ndim != 1 or self.sigma.shape[0] != d:
                    raise ValueError(f"sigma_mode=diag expects sigma as vector length {d}, got {self.sigma.shape}")

            elif self.sigma_mode == "cosine":
                pass

            self._global_log_once(n, d)
            return

        if self.sigma_mode in ("scalar", "diag"):
            sigma_vec = self._estimate_sigma_scott(X_raw)
            self.sigma = float(np.mean(sigma_vec)) if self.sigma_mode == "scalar" else sigma_vec

        elif self.sigma_mode == "full":
            X_full = (X_raw - self._y_mean) / self._y_std if self.standardize_y else X_raw
            C = np.cov(X_full, rowvar=False, ddof=1)
            C = np.atleast_2d(C)
            a = self.shrinkage
            scale = float(np.trace(C) / d)
            
            C_reg = (1.0 - a) * C + a * scale * np.eye(d)

            Sigma = (self.sigma_factor ** 2) * C_reg
            self._precision = np.linalg.pinv(Sigma)

        elif self.sigma_mode == "cosine":
            pass

        self._global_log_once(n, d)

    def _global_log_once(self, n, d):
        if self._did_global_log:
            return

        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            self._did_global_log = True
            return

        tau = self.cosine_temperature if self.cosine_temperature is not None else self.sigma_factor

        block = [
            "",
            "================ GLOBAL KERNEL CFG ================",
            f"sigma_mode: {self.sigma_mode}",
            f"kernel: {self.original_kernel}",
            f"label_csv: {self.label_csv}",
            f"label_cols: {self.label_cols}",
            f"n_samples: {n}",
            f"label_dim: {d}",
            f"sigma_factor: {self.sigma_factor}",
            f"standardize_y (FULL only): {self.standardize_y}",
            f"shrinkage (FULL only): {self.shrinkage}",
            f"cosine_temperature (COSINE only): {tau}",
            "",
        ]

        if self.sigma_mode in ("diag", "scalar"):
            block += ["--- sigma used ---", f"{np.asarray(self.sigma)}", ""]
        elif self.sigma_mode == "full":
            # <-- CHANGED: only log shape, not the full matrix
            shp = None if self._precision is None else tuple(self._precision.shape)
            block += ["--- precision used (Sigma^{-1}) ---", f"precision_shape: {shp} (should be ({d},{d}))", ""]
        elif self.sigma_mode == "cosine":
            block += ["--- cosine mode: no sigma/precision ---", ""]

        block += ["====================================================", ""]
        self.log.info("\n".join(block))
        self._did_global_log = True

    def _kernel_dispatch(self, y1, y2):
        y1 = np.asarray(y1, dtype=np.float64)
        y2 = np.asarray(y2, dtype=np.float64)

        if self.sigma_mode == "cosine":
            y1n = y1 / (np.linalg.norm(y1, axis=1, keepdims=True) + 1e-12)
            y2n = y2 / (np.linalg.norm(y2, axis=1, keepdims=True) + 1e-12)
            cos = y1n @ y2n.T
            tau = self.cosine_temperature if self.cosine_temperature is not None else float(self.sigma_factor)
            tau = max(float(tau), 1e-8)
            logits = cos / tau
            logits = logits - logits.max(axis=1, keepdims=True)
            return np.exp(logits)

        if self.sigma_mode == "full":
            if self._precision is None:
                raise RuntimeError("sigma_mode=full but precision is None (fit_once_global failed).")

            if self.standardize_y:
                y1 = (y1 - self._y_mean) / self._y_std
                y2 = (y2 - self._y_mean) / self._y_std

            diff = y1[:, None, :] - y2[None, :, :]
            d2 = np.einsum("...i,ij,...j->...", diff, self._precision, diff)
            return np.exp(-0.5 * d2)

        if self.sigma_mode == "scalar":
            s = float(self.sigma)
            gamma = 1.0 / (2.0 * s ** 2)
            return rbf_kernel(y1, y2, gamma=gamma)

        if self.sigma_mode == "diag":
            sigma_vec = np.asarray(self.sigma, dtype=np.float64)
            if sigma_vec.ndim != 1 or sigma_vec.shape[0] != y1.shape[1]:
                raise ValueError(f"sigma_mode=diag expects sigma vector length {y1.shape[1]}, got {sigma_vec.shape}")

            diff = y1[:, None, :] - y2[None, :, :]
            gamma_vec = 1.0 / (2.0 * sigma_vec ** 2)
            dists = np.sum(diff ** 2 * gamma_vec, axis=2)
            return np.exp(-dists)

        raise ValueError(f"Unsupported sigma_mode={self.sigma_mode}")

    def _estimate_sigma_scott(self, X):
        n, d = X.shape
        factor = n ** (-1.0 / (d + 4))
        variances = np.var(X, axis=0, ddof=1)
        sigma_vec = np.sqrt(variances) * factor
        return sigma_vec * self.sigma_factor

    def forward(self, z_i, z_j, labels, return_kernel=False, step=None, writer=None):
        N = len(z_i)
        assert N == len(labels), f"Unexpected labels length: {len(labels)}"

        z_i = func.normalize(z_i, p=2, dim=-1)
        z_j = func.normalize(z_j, p=2, dim=-1)

        sim_zii = (z_i @ z_i.T) / self.temperature
        sim_zjj = (z_j @ z_j.T) / self.temperature
        sim_zij = (z_i @ z_j.T) / self.temperature

        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels_np = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()
        weights_np = self.kernel(all_labels_np, all_labels_np)  # <-- ALWAYS numpy here

        if isinstance(weights_np, torch.Tensor):
            weights_np = weights_np.detach().cpu().numpy()

        weights_np = weights_np * (1 - np.eye(2 * N))
        weights_np = weights_np / (weights_np.sum(axis=1, keepdims=True) + 1e-8)

        # -------- LOG ONCE (but do NOT change types / flow) --------
        if (not self._logged_weight_stats) and int(os.environ.get("RANK", "0")) == 0:
            self._logged_weight_stats = True
            w_min = float(weights_np.min())
            w_mean = float(weights_np.mean())
            w_max = float(weights_np.max())
            entropy = float(-(weights_np * np.log(weights_np + 1e-12)).sum(axis=1).mean())
            eff = 1.0 / (weights_np**2).sum(axis=1)

            self.log.info(
                "\n========== KERNEL WEIGHT STATS ==========\n"
                f"min={w_min:.6e}\n"
                f"mean={w_mean:.6e}\n"
                f"max={w_max:.6e}\n"
                f"avg_entropy_per_row={entropy:.6f}\n"
                "========================================"
            )
            self.log.info(
                f"effective_neighbors mean/min/max = {eff.mean():.2f} / {eff.min():.2f} / {eff.max():.2f}"
            )

        # <-- CRITICAL FIX: ALWAYS convert to torch here
        weights = torch.from_numpy(weights_np).float().to(z_i.device)  # <-- FIXED (always happens)

        sim_Z = torch.cat([
            torch.cat([sim_zii, sim_zij], dim=1),
            torch.cat([sim_zij.T, sim_zjj], dim=1)
        ], dim=0)

        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = -1.0 / N * (weights * log_sim_Z).sum()

        if self.return_logits:
            return loss, sim_zij, sim_zii, sim_zjj

        return loss

    def __str__(self):
        return f"{type(self).__name__}(temp={self.temperature}, kernel={self.original_kernel}, sigma_mode={self.sigma_mode})"




# import os
# import logging
# from typing import Optional

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as func

# from sklearn.utils.validation import check_array
# from sklearn.metrics.pairwise import rbf_kernel

# try:
#     from omegaconf import ListConfig
# except Exception:
#     ListConfig = tuple  # fallback


# class GeneralizedSupervisedNTXenLoss(nn.Module):
#     """
#     SimCLR-pure positive + Y-aware neighbor weighting.

#     - Positive pair (i <-> i+N) is handled explicitly with fixed mass (1 - kernel_mass).
#     - Kernel weights + TopK are applied ONLY to "neighbors" (excluding diagonal and positive edge).
#     - The neighbor kernel mass is kernel_mass (alpha) and is distributed according to the kernel.
#     """

#     def __init__(
#         self,
#         label_csv: str,
#         label_cols: list[str],
#         kernel="rbf",
#         temperature=0.1,
#         return_logits=False,
#         sigma=None,
#         sigma_factor=1.0,
#         sigma_mode="diag",            # "scalar" | "diag" | "full" | "cosine"
#         standardize_y=False,          # ONLY applied for FULL mode
#         shrinkage=0.1,                # ONLY used for FULL mode
#         cosine_temperature=None,      # ONLY used for cosine mode; if None uses sigma_factor

#         # Top-k on neighbors ONLY
#         topk_frac: Optional[float] = None,   # e.g. 0.10 ; None disables
#         topk_min: int = 1,

#         # Mix: positive mass vs kernel-neighbor mass
#         kernel_mass: float = 0.2,            # alpha in [0,1]; pos mass = 1-alpha

#         # Optional calibration
#         calibrate_sigma: bool = False,
#         target_median_weight: float = 0.2,
#         calib_pairs: int = 200_000,
#         calib_seed: int = 0,
#     ):
#         super().__init__()
#         self.original_kernel = kernel
#         self.temperature = float(temperature)
#         self.return_logits = return_logits
#         self.INF = 1e8

#         self.log = set_file_logger(__file__)
#         self.log.setLevel(logging.INFO)
#         self.log.propagate = True

#         self.label_csv = label_csv
#         self.label_cols = label_cols if not isinstance(label_cols, str) else [label_cols]

#         if isinstance(sigma, ListConfig):
#             sigma = list(sigma)

#         self.sigma = sigma

#         if sigma_factor is None:
#             sigma_factor = 1.0
#         self.sigma_factor = float(sigma_factor)

#         self.sigma_mode = str(sigma_mode).lower()
#         assert self.sigma_mode in ("scalar", "diag", "full", "cosine"), f"Wrong sigma_mode={sigma_mode}"

#         self.standardize_y = bool(standardize_y)
#         self.shrinkage = float(shrinkage)
#         self.cosine_temperature = cosine_temperature

#         self.topk_frac = topk_frac
#         if self.topk_frac is not None:
#             self.topk_frac = float(self.topk_frac)
#             if not (0.0 < self.topk_frac <= 1.0):
#                 raise ValueError(f"topk_frac must be in (0,1], got {self.topk_frac}")
#         self.topk_min = int(topk_min)
#         if self.topk_min < 1:
#             raise ValueError(f"topk_min must be >= 1, got {self.topk_min}")

#         self.kernel_mass = float(kernel_mass)
#         if not (0.0 <= self.kernel_mass <= 1.0):
#             raise ValueError(f"kernel_mass must be in [0,1], got {self.kernel_mass}")

#         self.calibrate_sigma = bool(calibrate_sigma)
#         self.target_median_weight = float(target_median_weight)
#         self.calib_pairs = int(calib_pairs)
#         self.calib_seed = int(calib_seed)

#         self._y_mean = None
#         self._y_std = None
#         self._precision = None

#         if kernel == "rbf":
#             self.kernel = self._kernel_dispatch
#         else:
#             assert hasattr(kernel, "__call__"), "kernel must be callable"
#             self.kernel = kernel

#         self._did_global_log = False
#         self._logged_weight_stats_pre = False
#         self._logged_weight_stats_post = False

#         self._fit_once_global()

#     # ---------- calibration (same idea as before, optional) ----------
#     def _calibrate_from_csv_pairs(self, X_raw: np.ndarray):
#         rank = int(os.environ.get("RANK", "0"))
#         if rank != 0:
#             return
#         n, d = X_raw.shape
#         rng = np.random.default_rng(self.calib_seed)
#         P = min(self.calib_pairs, max(10_000, n * 5))

#         i = rng.integers(0, n, size=P)
#         j = rng.integers(0, n, size=P)
#         same = (i == j)
#         while same.any():
#             j[same] = rng.integers(0, n, size=same.sum())
#             same = (i == j)

#         Xi = X_raw[i]
#         Xj = X_raw[j]
#         target = self.target_median_weight

#         if self.sigma_mode == "cosine":
#             Xi_n = Xi / (np.linalg.norm(Xi, axis=1, keepdims=True) + 1e-12)
#             Xj_n = Xj / (np.linalg.norm(Xj, axis=1, keepdims=True) + 1e-12)
#             cos = np.sum(Xi_n * Xj_n, axis=1)
#             m = float(np.median(1.0 - cos))
#             tau = m / (-np.log(target) + 1e-12)
#             tau = float(np.clip(tau, 1e-6, 1e6))
#             self.cosine_temperature = tau
#             self.log.info(f"[CALIB COSINE] target={target:.3f} | median(1-cos)={m:.4g} -> tau={tau:.4g}")
#             return

#         if self.sigma_mode in ("diag", "scalar"):
#             sigma_vec_base = self._estimate_sigma_scott(X_raw, sigma_factor_override=1.0)
#             sigma_vec_base = np.maximum(sigma_vec_base, 1e-12)
#             if self.sigma_mode == "scalar":
#                 s = float(np.mean(sigma_vec_base))
#                 diff = (Xi - Xj) / s
#                 d2_base = np.sum(diff * diff, axis=1)
#             else:
#                 diff = (Xi - Xj) / sigma_vec_base
#                 d2_base = np.sum(diff * diff, axis=1)
#         elif self.sigma_mode == "full":
#             mu = X_raw.mean(axis=0, keepdims=True)
#             std = np.maximum(X_raw.std(axis=0, keepdims=True), 1e-8)
#             Xf = (X_raw - mu) / std if self.standardize_y else X_raw
#             C = np.cov(Xf, rowvar=False, ddof=1)
#             C = np.atleast_2d(C)
#             a = self.shrinkage
#             scale = float(np.trace(C) / d)
#             C_reg = (1.0 - a) * C + a * scale * np.eye(d)
#             Pmat = np.linalg.pinv(C_reg)
#             Xi_f = (Xi - mu) / std if self.standardize_y else Xi
#             Xj_f = (Xj - mu) / std if self.standardize_y else Xj
#             diff = Xi_f - Xj_f
#             d2_base = np.einsum("bi,ij,bj->b", diff, Pmat, diff)
#         else:
#             raise ValueError(self.sigma_mode)

#         q = float(np.median(d2_base))
#         sf = np.sqrt(q / (-2.0 * np.log(target) + 1e-12))
#         sf = float(np.clip(sf, 1e-6, 1e6))
#         self.sigma_factor = sf
#         achieved = float(np.median(np.exp(-0.5 * d2_base / (sf ** 2))))
#         self.log.info(
#             f"[CALIB {self.sigma_mode.upper()}] target={target:.3f} | median(d2_base)={q:.4g} "
#             f"-> sigma_factor={sf:.4g} | achieved={achieved:.3f}"
#         )

#     def _fit_once_global(self):
#         df = pd.read_csv(self.label_csv)
#         df = df.dropna(subset=self.label_cols)
#         X_raw = check_array(df[self.label_cols].values).astype(np.float64)
#         n, d = X_raw.shape

#         if n < 2:
#             raise ValueError(f"[FULL FIT] Not enough samples after dropna: n={n} csv={self.label_csv}")
#         if d < 1:
#             raise ValueError(f"[FULL FIT] No label columns (d={d}) csv={self.label_csv}")

#         self._y_mean = X_raw.mean(axis=0, keepdims=True)
#         self._y_std = np.maximum(X_raw.std(axis=0, keepdims=True), 1e-8)

#         if self.calibrate_sigma and self.sigma is None:
#             self._calibrate_from_csv_pairs(X_raw)

#         if self.sigma is not None:
#             if self.sigma_mode == "full":
#                 Sigma = np.asarray(self.sigma, dtype=np.float64)
#                 Sigma = np.atleast_2d(Sigma)
#                 if Sigma.shape != (d, d):
#                     raise ValueError(f"full expects DxD, got {Sigma.shape}")
#                 a = self.shrinkage
#                 scale = float(np.trace(Sigma) / d)
#                 Sigma_reg = (1.0 - a) * Sigma + a * scale * np.eye(d)
#                 Sigma_reg = (self.sigma_factor ** 2) * Sigma_reg
#                 self._precision = np.linalg.pinv(Sigma_reg)
#             elif self.sigma_mode == "scalar":
#                 self.sigma = float(self.sigma)
#             elif self.sigma_mode == "diag":
#                 self.sigma = np.asarray(self.sigma, dtype=np.float64)
#                 if self.sigma.ndim != 1 or self.sigma.shape[0] != d:
#                     raise ValueError(f"diag expects len {d}, got {self.sigma.shape}")
#             elif self.sigma_mode == "cosine":
#                 pass
#             self._global_log_once(n, d)
#             return

#         if self.sigma_mode in ("scalar", "diag"):
#             sigma_vec = self._estimate_sigma_scott(X_raw, sigma_factor_override=None)
#             self.sigma = float(np.mean(sigma_vec)) if self.sigma_mode == "scalar" else sigma_vec
#         elif self.sigma_mode == "full":
#             X_full = (X_raw - self._y_mean) / self._y_std if self.standardize_y else X_raw
#             C = np.cov(X_full, rowvar=False, ddof=1)
#             C = np.atleast_2d(C)
#             a = self.shrinkage
#             scale = float(np.trace(C) / d)
#             C_reg = (1.0 - a) * C + a * scale * np.eye(d)
#             Sigma = (self.sigma_factor ** 2) * C_reg
#             self._precision = np.linalg.pinv(Sigma)
#         elif self.sigma_mode == "cosine":
#             pass

#         self._global_log_once(n, d)

#     def _global_log_once(self, n, d):
#         if self._did_global_log:
#             return
#         rank = int(os.environ.get("RANK", "0"))
#         if rank != 0:
#             self._did_global_log = True
#             return

#         tau = self.cosine_temperature if self.cosine_temperature is not None else self.sigma_factor
#         self.log.info(
#             "\n".join([
#                 "",
#                 "================ GLOBAL KERNEL CFG ================",
#                 f"sigma_mode: {self.sigma_mode}",
#                 f"kernel: {self.original_kernel}",
#                 f"label_csv: {self.label_csv}",
#                 f"label_cols: {self.label_cols}",
#                 f"n_samples: {n}",
#                 f"label_dim: {d}",
#                 f"sigma_factor: {self.sigma_factor}",
#                 f"calibrate_sigma: {self.calibrate_sigma}",
#                 f"target_median_weight: {self.target_median_weight}",
#                 f"kernel_mass(alpha): {self.kernel_mass}  (pos mass = {1.0 - self.kernel_mass})",
#                 f"topk_frac: {self.topk_frac}",
#                 f"topk_min : {self.topk_min}",
#                 f"standardize_y (FULL): {self.standardize_y}",
#                 f"shrinkage (FULL): {self.shrinkage}",
#                 f"cosine_temperature (COSINE): {tau}",
#                 "====================================================",
#                 "",
#             ])
#         )
#         self._did_global_log = True

#     def _kernel_dispatch(self, y1, y2):
#         y1 = np.asarray(y1, dtype=np.float64)
#         y2 = np.asarray(y2, dtype=np.float64)

#         if self.sigma_mode == "cosine":
#             y1n = y1 / (np.linalg.norm(y1, axis=1, keepdims=True) + 1e-12)
#             y2n = y2 / (np.linalg.norm(y2, axis=1, keepdims=True) + 1e-12)
#             cos = y1n @ y2n.T
#             tau = self.cosine_temperature if self.cosine_temperature is not None else float(self.sigma_factor)
#             tau = max(float(tau), 1e-8)
#             logits = cos / tau
#             logits = logits - logits.max(axis=1, keepdims=True)
#             return np.exp(logits)

#         if self.sigma_mode == "full":
#             if self._precision is None:
#                 raise RuntimeError("sigma_mode=full but precision is None.")
#             if self.standardize_y:
#                 y1 = (y1 - self._y_mean) / self._y_std
#                 y2 = (y2 - self._y_mean) / self._y_std
#             diff = y1[:, None, :] - y2[None, :, :]
#             d2 = np.einsum("...i,ij,...j->...", diff, self._precision, diff)
#             return np.exp(-0.5 * d2)

#         if self.sigma_mode == "scalar":
#             s = float(self.sigma)
#             gamma = 1.0 / (2.0 * s ** 2)
#             return rbf_kernel(y1, y2, gamma=gamma)

#         if self.sigma_mode == "diag":
#             sigma_vec = np.asarray(self.sigma, dtype=np.float64)
#             diff = y1[:, None, :] - y2[None, :, :]
#             gamma_vec = 1.0 / (2.0 * sigma_vec ** 2)
#             dists = np.sum(diff ** 2 * gamma_vec, axis=2)
#             return np.exp(-dists)

#         raise ValueError(self.sigma_mode)

#     def _estimate_sigma_scott(self, X, sigma_factor_override: Optional[float] = None):
#         n, d = X.shape
#         factor = n ** (-1.0 / (d + 4))
#         variances = np.var(X, axis=0, ddof=1)
#         sigma_vec = np.sqrt(variances) * factor
#         sf = self.sigma_factor if sigma_factor_override is None else float(sigma_factor_override)
#         return sigma_vec * sf

#     def forward(self, z_i, z_j, labels, return_kernel=False, step=None, writer=None):
#         N = len(z_i)
#         assert N == len(labels), f"Unexpected labels length: {len(labels)}"

#         z_i = func.normalize(z_i, p=2, dim=-1)
#         z_j = func.normalize(z_j, p=2, dim=-1)

#         sim_zii = (z_i @ z_i.T) / self.temperature
#         sim_zjj = (z_j @ z_j.T) / self.temperature
#         sim_zij = (z_i @ z_j.T) / self.temperature

#         sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
#         sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

#         # ---- kernel on labels (2N x 2N) ----
#         all_labels_np = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()
#         K = self.kernel(all_labels_np, all_labels_np)
#         if isinstance(K, torch.Tensor):
#             K = K.detach().cpu().numpy()

#         M = 2 * N
#         rows = np.arange(M)
#         pos_cols = np.concatenate([np.arange(N) + N, np.arange(N)])  # positive per row

#         # candidates for kernel-neighbors: exclude diag and exclude positive edge
#         K = K * (1 - np.eye(M))
#         K[rows, pos_cols] = 0.0

#         # PRE stats
#         if (not self._logged_weight_stats_pre) and int(os.environ.get("RANK", "0")) == 0:
#             self._logged_weight_stats_pre = True
#             Ptmp = K / (K.sum(axis=1, keepdims=True) + 1e-8)
#             entropy = float(-(Ptmp * np.log(Ptmp + 1e-12)).sum(axis=1).mean())
#             eff = 1.0 / (Ptmp ** 2).sum(axis=1)
#             self.log.info(
#                 "\n========== PRE-TOPK (neighbors-only kernel) ==========\n"
#                 f"kernel_mass(alpha)={self.kernel_mass}\n"
#                 f"avg_entropy={entropy:.6f}\n"
#                 f"effective_neighbors mean/min/max = {eff.mean():.2f} / {eff.min():.2f} / {eff.max():.2f}\n"
#                 "====================================================="
#             )

#         # Top-k on neighbors ONLY
#         if self.topk_frac is not None:
#             k = int(np.ceil(self.topk_frac * (M - 2)))  # exclude diag + pos
#             k = max(k, self.topk_min)
#             k = min(k, M - 2)

#             topk_idx = np.argpartition(K, -k, axis=1)[:, -k:]
#             mask = np.zeros_like(K, dtype=bool)
#             mask[rows[:, None], topk_idx] = True
#             np.fill_diagonal(mask, False)
#             mask[rows, pos_cols] = False
#             K = np.where(mask, K, 0.0)

#         # Build final weights: positive fixed mass, neighbors share kernel_mass
#         alpha = self.kernel_mass
#         W = np.zeros_like(K, dtype=np.float64)
#         W[rows, pos_cols] = (1.0 - alpha)

#         Ksum = K.sum(axis=1, keepdims=True) + 1e-8
#         W += alpha * (K / Ksum)

#         # POST stats
#         if (not self._logged_weight_stats_post) and int(os.environ.get("RANK", "0")) == 0:
#             self._logged_weight_stats_post = True
#             kept = (K > 0).sum(axis=1)
#             Pn = (K / Ksum)
#             entropy = float(-(Pn * np.log(Pn + 1e-12)).sum(axis=1).mean())
#             eff = 1.0 / (Pn ** 2).sum(axis=1)
#             self.log.info(
#                 "\n========== POST-TOPK (neighbors-only kernel) ==========\n"
#                 f"kept_neighbors mean/min/max = {kept.mean():.1f} / {kept.min()} / {kept.max()}\n"
#                 f"avg_entropy={entropy:.6f}\n"
#                 f"effective_neighbors mean/min/max = {eff.mean():.2f} / {eff.min():.2f} / {eff.max():.2f}\n"
#                 "======================================================"
#             )

#         weights = torch.from_numpy(W).float().to(z_i.device)

#         sim_Z = torch.cat(
#             [
#                 torch.cat([sim_zii, sim_zij], dim=1),
#                 torch.cat([sim_zij.T, sim_zjj], dim=1),
#             ],
#             dim=0,
#         )

#         log_sim_Z = func.log_softmax(sim_Z, dim=1)
#         loss = -1.0 / N * (weights * log_sim_Z).sum()

#         if return_kernel:
#             return loss, weights

#         if self.return_logits:
#             return loss, sim_zij, sim_zii, sim_zjj

#         return loss










# Optional utility kernel if needed
# def rbf_kernel(Y1, Y2, gamma=1.0):
#     if isinstance(Y1, np.ndarray):
#         Y1 = torch.from_numpy(Y1)
#     if isinstance(Y2, np.ndarray):
#         Y2 = torch.from_numpy(Y2)

#     Y1 = Y1.to(dtype=torch.float32)
#     Y2 = Y2.to(dtype=torch.float32)
    
#     diff = Y1.unsqueeze(1) - Y2.unsqueeze(0)
#     dist_sq = torch.sum(diff ** 2, dim=-1)
#     return torch.exp(-gamma * dist_sq)

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_if_ddp(x: torch.Tensor) -> torch.Tensor:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return x
    world_size = torch.distributed.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


class CLIPLoss(nn.Module):
    """
    Symmetric CLIP / InfoNCE loss.
    Expects L2-normalized embeddings.
    """

    def __init__(self, temperature: float = 0.07, gather_ddp: bool = True, return_logits: bool = False):
        super().__init__()
        self.temperature = float(temperature)
        self.gather_ddp = bool(gather_ddp)
        self.return_logits = bool(return_logits)

    def forward(self, z_img: torch.Tensor, z_tab: torch.Tensor):
        """
        z_img: [B, D]
        z_tab: [B, D]
        """

        if self.gather_ddp:
            z_img_all = _gather_if_ddp(z_img)
            z_tab_all = _gather_if_ddp(z_tab)
        else:
            z_img_all, z_tab_all = z_img, z_tab

        logits = (z_img @ z_tab_all.t()) / self.temperature
        logits_t = (z_tab @ z_img_all.t()) / self.temperature

        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.gather_ddp:
            rank = torch.distributed.get_rank()
            b = z_img.shape[0]
            targets = torch.arange(b, device=z_img.device) + rank * b
        else:
            targets = torch.arange(z_img.shape[0], device=z_img.device)
        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits_t, targets)
        loss = 0.5 * (loss_i2t + loss_t2i)

        if self.return_logits:
            # sims for plotting/debug (local batch only)
            sim_zij = (z_img @ z_tab.t()).detach()
            sim_zii = (z_img @ z_img.t()).detach()
            sim_zjj = (z_tab @ z_tab.t()).detach()
            return loss, sim_zij, sim_zii, sim_zjj

        return loss




class NTXenLoss_WithoutHardNegative(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zii > quantile_negative_jj] = -self.INF

        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class NTXenLoss_Mixed(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations,
    arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward_NearestNeighbours_OtherView(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        z_nn_i = z_j[max_ij.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        z_nn_j = z_i[max_ji.indices]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_NearestNeighbours(self, z_i, z_j):
        N = len(z_i)
        diag_inf = self.INF * torch.eye(N, device=z_i.device)

        #####################################################
        # Computes the classical terms for NTXenLoss
        #####################################################

        log.info("histogram of z_i before normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        log.info("histogram of z_i after normalization:")
        log.info(np.histogram(z_i.detach().cpu().numpy() * 100, bins='auto'))

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature
        sim_zji = sim_zij.T

        log.info("histogram of sim_zij:")
        log.info(
            np.histogram(
                sim_zij.detach().cpu().numpy() *
                self.temperature *
                100,
                bins='auto'))

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_i
        #####################################################

        max_ii = torch.max(sim_zii - diag_inf, dim=1)
        max_ij = torch.max(sim_zij - diag_inf, dim=1)

        # Computes nearest-neighbour of z_i
        z_nn_i = torch.zeros(z_i.shape, device=z_i.device)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                z_nn_i[i] = z_i[max_ii.indices[i]]
            else:
                z_nn_i[i] = z_j[max_ij.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zii = (z_nn_i @ z_i.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zij = (z_nn_i @ z_j.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_ii.values[i] > max_ij.values[i]:
                sim_nn_zii[i, max_ii.indices[i]] = -self.INF
            else:
                sim_nn_zij[i, max_ij.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zii = sim_nn_zii - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_nn_zij, sim_nn_zii], dim=1),
                                    correct_pairs)

        #####################################################
        # Computes the terms for NearestNeighbour NTXenLoss
        # loss_j
        #####################################################

        max_jj = torch.max(sim_zjj - diag_inf, dim=1)
        max_ji = torch.max(sim_zji - diag_inf, dim=1)

        # Computes nearest-neighbour of z_j
        z_nn_j = torch.zeros(z_j.shape, device=z_j.device)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                z_nn_j[i] = z_j[max_jj.indices[i]]
            else:
                z_nn_j[i] = z_i[max_ji.indices[i]]

        # dim [N, N] => Upper triangle contains incorrect pairs (nn(i),i+)
        sim_nn_zjj = (z_nn_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (nn(i),j)
        sim_nn_zji = (z_nn_j @ z_i.T) / self.temperature

        # 'Remove' the covariant vectors by penalizing it (exp(-inf) = 0)
        for i in range(N):
            if max_jj.values[i] > max_ji.values[i]:
                sim_nn_zjj[i, max_jj.indices[i]] = -self.INF
            else:
                sim_nn_zji[i, max_ji.indices[i]] = -self.INF

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_nn_zjj = sim_nn_zjj - diag_inf

        # Computes nearest neighbour contrastive loss for first view i
        loss_j = func.cross_entropy(torch.cat([sim_nn_zji, sim_nn_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward_WithoutHardNegative(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)  # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1)  # dim [N, D]

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zii = (z_i @ z_i.T) / self.temperature

        # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature

        # dim [N, N] => the diag contains the correct pairs (i,j)
        # (x transforms via T_i and T_j)
        sim_zij = (z_i @ z_j.T) / self.temperature

        # Diagonals as 1D tensor
        diag_ij = sim_zij.diagonal()

        # Prints quantiles of positive pairs (views from the same image)
        quantile_positive_pairs = diag_ij.quantile(0.75)
        log.info(
            f"quantile of positives ij = "
            f"{quantile_positive_pairs.cpu()*self.temperature*100}")

        # Computes quantiles of negative pairs
        quantile_negative_ii = quantile_off_diagonal(sim_zii)
        quantile_negative_jj = quantile_off_diagonal(sim_zjj)
        quantile_negative_ij = quantile_off_diagonal(sim_zij)

        # Prints quantiles of negative pairs
        log.info(
            f"quantile of negatives ii = "
            f"{quantile_negative_ii.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives jj = "
            f"{quantile_negative_jj.cpu()*self.temperature*100}")
        log.info(
            f"quantile of negatives ij = "
            f"{quantile_negative_ij.cpu()*self.temperature*100}")

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        # 'Remove' the parts that are hard negatives to promote clustering
        sim_zii[sim_zii > quantile_negative_ii] = -self.INF
        sim_zjj[sim_zjj > quantile_negative_jj] = -self.INF

        # 'Remove' the parts that are hard negatives to promote clustering
        # We keep the positive element j (second view)
        negative_ij = sim_zij - diag_ij.diag()
        negative_ij[negative_ij > quantile_negative_ij] = -self.INF
        negative_ij.fill_diagonal_(0.)
        sim_zij = negative_ij + diag_ij.diag()

        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1),
                                    correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1),
                                    correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def forward(self, z_i, z_j):
        loss_NN, _, _ = self.forward_NearestNeighbours_OtherView(z_i, z_j)
        loss_WHN, sim_zij, correct_pairs = self.forward_WithoutHardNegative(
            z_i, z_j)

        if self.return_logits:
            return ((loss_NN + loss_WHN) / 2), sim_zij, correct_pairs

        return ((loss_NN + loss_WHN) / 2)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)
