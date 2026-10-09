"""Deep learning for tabular data: an entity-embedding MLP in PyTorch.

Categorical features are mapped to learned dense embeddings (Guo & Berkhahn,
2016) instead of sparse one-hot vectors; numeric features are standardised.
The network is wrapped as a scikit-learn estimator so it plugs into the same
cross-validation, Optuna tuning, stacking and persistence code as every other
model in the project.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn


class _TabularNet(nn.Module):
    def __init__(self, cardinalities, emb_dims, n_num, hidden, dropout):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims))
        self.emb_dropout = nn.Dropout(dropout / 2)
        layers: list[nn.Module] = []
        width = sum(emb_dims) + n_num
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        emb = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat([self.emb_dropout(torch.cat(emb, dim=1)), x_num], dim=1)
        return self.mlp(x).squeeze(1)


class TabularNetClassifier(ClassifierMixin, BaseEstimator):
    """Entity-embedding MLP with early stopping on a stratified validation split."""

    def __init__(
        self,
        cat_cols: tuple[str, ...] = (),
        num_cols: tuple[str, ...] = (),
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        patience: int = 15,
        val_fraction: float = 0.15,
        emb_scale: float = 1.0,
        random_state: int = 42,
        n_threads: int = 4,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.emb_scale = emb_scale
        self.random_state = random_state
        self.n_threads = n_threads

    # -- encoding -----------------------------------------------------------
    def _encode(self, X: pd.DataFrame):
        cats = np.column_stack(
            [
                X[c].astype(str).map(self.vocab_[c]).fillna(0).astype(np.int64).to_numpy()
                for c in self.cat_cols
            ]
        )
        nums = (X[list(self.num_cols)].to_numpy(dtype=np.float32) - self.mean_) / self.std_
        return torch.from_numpy(cats), torch.from_numpy(nums.astype(np.float32))

    # -- sklearn API --------------------------------------------------------
    def fit(self, X: pd.DataFrame, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(self.n_threads)
        y = np.asarray(y).astype(np.float32)
        self.classes_ = np.array([0, 1])

        # index 0 is reserved for categories unseen during training
        self.vocab_ = {
            c: {v: i + 1 for i, v in enumerate(sorted(X[c].astype(str).unique()))}
            for c in self.cat_cols
        }
        num = X[list(self.num_cols)].to_numpy(dtype=np.float32)
        self.mean_ = num.mean(axis=0)
        self.std_ = num.std(axis=0) + 1e-6

        cardinalities = [len(self.vocab_[c]) + 1 for c in self.cat_cols]
        emb_dims = [max(2, int(round(self.emb_scale * min(12, (c + 1) // 2)))) for c in cardinalities]
        self.net_ = _TabularNet(cardinalities, emb_dims, len(self.num_cols), self.hidden, self.dropout)

        idx_tr, idx_va = train_test_split(
            np.arange(len(y)), test_size=self.val_fraction, stratify=y, random_state=self.random_state
        )
        xc, xn = self._encode(X)
        yt = torch.from_numpy(y)
        tr = (xc[idx_tr], xn[idx_tr], yt[idx_tr])
        va = (xc[idx_va], xn[idx_va], yt[idx_va])

        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=max(2, self.patience // 3))
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.random_state)

        best_loss, best_state, bad_epochs = np.inf, None, 0
        self.history_ = {"train_loss": [], "val_loss": [], "val_auc": []}
        n = len(idx_tr)
        for _epoch in range(self.max_epochs):
            self.net_.train()
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start : start + self.batch_size]
                if len(b) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net_(tr[0][b], tr[1][b]), tr[2][b])
                loss.backward()
                opt.step()
                total += loss.item() * len(b)
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(va[0], va[1])
                val_loss = loss_fn(logits, va[2]).item()
                val_auc = roc_auc_score(va[2].numpy(), torch.sigmoid(logits).numpy())
            self.history_["train_loss"].append(total / n)
            self.history_["val_loss"].append(val_loss)
            self.history_["val_auc"].append(val_auc)
            sched.step(val_loss)
            if val_loss < best_loss - 1e-5:
                best_loss, best_state, bad_epochs = val_loss, copy.deepcopy(self.net_.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        self.net_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        self.net_.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        xc, xn = self._encode(X)
        with torch.no_grad():
            p = torch.sigmoid(self.net_(xc, xn)).numpy().astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
