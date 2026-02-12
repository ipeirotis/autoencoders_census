# chow_liu_rank.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from alembic.ddl.base import drop_column
from gitdb.util import exists

from dataset.loader import DataLoader
from utils import define_necessary_elements


# -------------------------
# Schema for categorical-only tables
# -------------------------

@dataclass
class ColumnSpec:
    name: str
    cats: List[str]
    cat2idx: Dict[str, int]
    idx2cat: List[str]
    offset: int
    K: int

class CLTree:
    """
    Chow–Liu tree for categorical data.
    - Treats literal 'nan' as a category (kept as string "nan").
    - Uses Laplace smoothing (alpha) for robust probabilities.
    - Provides per-row log-likelihood and convenience scores.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        # schema
        self.columns: List[str] = []
        self.specs: Dict[str, ColumnSpec] = {}
        self.total_dim: int = 0
        # structure
        self.root: Optional[str] = None
        self.parent: Dict[str, Optional[str]] = {}
        self.children: Dict[str, List[str]] = {}
        self.mi_edges: List[Tuple[str, str, float]] = []  # (u, v, MI)
        # params
        self.p_root: Dict[str, np.ndarray] = {}
        self.cpt: Dict[Tuple[str, str], np.ndarray] = {}  # (child,parent) -> [Kp, Kc]

    # ---------- utils ----------
    @staticmethod
    def _as_str(s: pd.Series) -> pd.Series:
        return s.astype("object").where(pd.notnull(s), "nan").astype(str)

    def _build_schema(self, df: pd.DataFrame):
        self.columns = list(df.columns)
        offset = 0
        for c in self.columns:
            s = self._as_str(df[c])
            cats = sorted(pd.Index(s).value_counts().index.tolist())
            if "nan" not in cats:
                cats = ["nan"] + cats
            K = len(cats)
            cat2idx = {v: i for i, v in enumerate(cats)}
            spec = ColumnSpec(
                name=c, cats=cats, cat2idx=cat2idx, idx2cat=list(cats),
                offset=offset, K=K
            )
            self.specs[c] = spec
            offset += K
        self.total_dim = offset

    def _df_to_ints(self, df: pd.DataFrame) -> np.ndarray:
        X = np.zeros((len(df), len(self.columns)), dtype=np.int64)
        for j, c in enumerate(self.columns):
            s = self._as_str(df[c])
            m = self.specs[c].cat2idx
            X[:, j] = s.map(lambda v: m.get(v, m["nan"])).values
        return X

    # ---------- MI & tree ----------
    def _mutual_info(self, X: np.ndarray) -> Dict[Tuple[str, str], float]:
        N, d = X.shape
        mi = {}
        for i in range(d):
            ci = self.columns[i]
            Ki = self.specs[ci].K
            xi = X[:, i]
            for j in range(i + 1, d):
                cj = self.columns[j]
                Kj = self.specs[cj].K
                xj = X[:, j]
                cont = np.zeros((Ki, Kj), dtype=np.float64)
                np.add.at(cont, (xi, xj), 1.0)
                cont += self.alpha
                pxy = cont / cont.sum()
                px = pxy.sum(1, keepdims=True)
                py = pxy.sum(0, keepdims=True)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = pxy / (px * py)
                    term = np.where(pxy > 0, pxy * np.log(ratio + 1e-12), 0.0)
                val = float(np.nansum(term))
                mi[(ci, cj)] = val
                mi[(cj, ci)] = val
        return mi

    def _choose_root(self, mi: Dict[Tuple[str, str], float]) -> str:
        # pick the most "central" node by MI degree
        centrality = {c: 0.0 for c in self.columns}
        for u in self.columns:
            centrality[u] = sum(mi.get((u, v), 0.0) for v in self.columns if v != u)
        return max(self.columns, key=lambda c: centrality[c])

    def _build_tree(self, mi: Dict[Tuple[str, str], float], root: str):
        cols = set(self.columns)
        U = {root}
        V = cols - U
        parent = {root: None}
        children = {c: [] for c in self.columns}
        edges = []
        # Prim's algorithm on complete graph with weights MI(u,v)
        while V:
            best = None; best_w = -1.0
            for u in U:
                for v in V:
                    w = mi.get((u, v), 0.0)
                    if w > best_w:
                        best = (u, v); best_w = w
            u, v = best
            parent[v] = u
            children[u].append(v)
            U.add(v); V.remove(v)
            edges.append((u, v, best_w))
        self.parent, self.children, self.root = parent, children, root
        self.mi_edges = edges  # store (u -> v) with MI weight

    # ---------- parameter estimation ----------
    def _estimate_params(self, X: np.ndarray):
        # Root marginal
        r = self.root
        rj = self.columns.index(r)
        Kr = self.specs[r].K
        counts = np.bincount(X[:, rj], minlength=Kr).astype(np.float64) + self.alpha
        self.p_root[r] = counts / counts.sum()
        # Conditionals
        for c in self.columns:
            p = self.parent.get(c, None)
            if p is None:  # root handled above
                continue
            cj = self.columns.index(c)
            pj = self.columns.index(p)
            Kc = self.specs[c].K
            Kp = self.specs[p].K
            cont = np.zeros((Kp, Kc), dtype=np.float64)
            np.add.at(cont, (X[:, pj], X[:, cj]), 1.0)
            cont += self.alpha
            self.cpt[(c, p)] = cont / cont.sum(axis=1, keepdims=True)

    # ---------- public API ----------
    def fit(self, df: pd.DataFrame, root: Optional[str] = None, mi_subsample: Optional[int] = None):
        """
        Fit tree on df. If mi_subsample is set (e.g., 10000), use a random subset
        of rows for MI to speed up on very large datasets.
        """
        self._build_schema(df)
        if mi_subsample is not None and len(df) > mi_subsample:
            samp = df.sample(mi_subsample, random_state=42)
        else:
            samp = df
        X_sub = self._df_to_ints(samp)
        mi = self._mutual_info(X_sub)
        if root is None:
            root = self._choose_root(mi)
        self._build_tree(mi, root)
        X = self._df_to_ints(df)
        self._estimate_params(X)
        return self

    def log_likelihood(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized per-row log-likelihood under the learned tree.
        """
        X = self._df_to_ints(df)
        N, d = X.shape
        logp = np.zeros(N, dtype=np.float64)
        # root term
        r = self.root; rj = self.columns.index(r)
        pr = np.clip(self.p_root[r], 1e-12, None)
        logp += np.log(pr[X[:, rj]])
        # child terms
        for c in self.columns:
            p = self.parent.get(c, None)
            if p is None:
                continue
            cj = self.columns.index(c)
            pj = self.columns.index(p)
            table = np.clip(self.cpt[(c, p)], 1e-12, None)  # [Kp, Kc]
            logp += np.log(table[X[:, pj], X[:, cj]])
        return logp

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy with scoring columns:
        - logp: total log-likelihood
        - avg_logp: per-column average log-prob (helps interpretability)
        - gmean_prob: geometric-mean probability per column (exp(avg_logp))
        - rank_desc: 1=most typical (highest logp)
        - pct: percentile of logp among rows
        - z: z-score of logp among rows
        """
        logp = self.log_likelihood(df)
        d = len(self.columns)
        avg_logp = logp / d
        gmean_prob = np.exp(avg_logp)  # in [0,1], interpretable
        out = df.copy()
        out["logp"] = logp
        out["avg_logp"] = avg_logp
        out["gmean_prob"] = gmean_prob
        # ranks / percentiles
        order = np.argsort(-logp)  # descending (most typical first)
        rank = np.empty_like(order); rank[order] = np.arange(1, len(df)+1)
        out["rank_desc"] = rank
        pct = (rank - 1) / (len(df) - 1 + 1e-9)
        out["pct"] = 1.0 - pct  # 1.0 = most typical
        # z-score
        mu, sd = logp.mean(), logp.std(ddof=0) + 1e-12
        out["z"] = (logp - mu) / sd
        return out#.sort_values("logp", ascending=False).reset_index(drop=True)

    def edges(self) -> List[Tuple[str, str, float]]:
        """
        Return directed edges (parent, child, mutual information weight).
        """
        return list(self.mi_edges)

# --------------- convenience function ---------------

def rank_rows_by_chow_liu(df: pd.DataFrame,
                          alpha: float = 1.0,
                          mi_subsample: Optional[int] = None):
    """
    Fit a Chow–Liu tree on 'df' and return:
      - ranked_df: df with scoring columns, sorted by logp DESC
      - model: the fitted CLTree (exposes edges() and parameters)
    """
    cl = CLTree(alpha=alpha).fit(df, mi_subsample=mi_subsample)
    ranked = cl.score_dataframe(df)
    return ranked, cl


import pandas as pd
from chow_liu_rank import rank_rows_by_chow_liu

path = "."
data = "pennycook_1"
drop_columns = None
rename_columns = None
interest_columns = None

(
        drop_columns,
        rename_columns,
        interest_columns,
        additional_drop_columns,
        additional_rename_columns,
        additional_interest_columns,
    ) = define_necessary_elements(data, drop_columns, rename_columns, interest_columns)

data_loader = DataLoader(
    drop_columns,
    rename_columns,
    interest_columns,
    additional_drop_columns=additional_drop_columns,
    additional_rename_columns=additional_rename_columns,
    additional_columns_of_interest=additional_interest_columns,
)
df, variable_types = data_loader.load_data(data)

# df: your DataFrame, all columns categorical (strings). Keep 'nan' as literal if needed.
ranked_df, cl_model = rank_rows_by_chow_liu(df, alpha=1.0, mi_subsample=10000)

ranked_df.rename(columns={
    "pct": "error",
}, inplace=True)

import os
# create the directory if it doesn't exist
os.makedirs(f"{path}/{data}_cl", exist_ok=True)

ranked_df.to_csv(f"{path}/{data}_cl/errors.csv", index=False)