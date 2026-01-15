# modules/analysis/coauthor/compute.py
from __future__ import annotations
import itertools
import pandas as pd
import streamlit as st

# Optional deps
try:
    import networkx as nx  # type: ignore
    HAS_NX = True
except Exception:
    HAS_NX = False

from .filters_adapter import split_authors

@st.cache_data(ttl=600, show_spinner=False)
def author_total_counts(df: pd.DataFrame) -> pd.Series:
    if "著者" not in df.columns:
        return pd.Series(dtype=int)
    bags = []
    for a in df["著者"].fillna(""):
        names = list(dict.fromkeys(split_authors(a)))
        bags += names
    if not bags:
        return pd.Series(dtype=int)
    s = pd.Series(bags, dtype="object")
    return s.value_counts().sort_values(ascending=False)

@st.cache_data(ttl=600, show_spinner=False)
def yearly_author_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "著者" not in df.columns or "発行年" not in df.columns:
        return pd.DataFrame(columns=["発行年", "著者", "count"])
    rows = []
    for _, r in df.iterrows():
        y = pd.to_numeric(r.get("発行年"), errors="coerce")
        if pd.isna(y):
            continue
        names = list(dict.fromkeys(split_authors(r.get("著者", ""))))
        for n in names:
            rows.append((int(y), n))
    if not rows:
        return pd.DataFrame(columns=["発行年", "著者", "count"])
    c = pd.DataFrame(rows, columns=["発行年","著者"]).value_counts().reset_index(name="count")
    return c.sort_values(["発行年","count"], ascending=[True, False]).reset_index(drop=True)

@st.cache_data(ttl=600, show_spinner=False)
def build_coauthor_edges(df: pd.DataFrame) -> pd.DataFrame:
    """前段で年/対象物/タイプのフィルタ済みDFを受け取り、共著ペアを数える。"""
    rows = []
    for a in df.get("著者", pd.Series(dtype=str)).fillna(""):
        names = sorted(set(split_authors(a)))
        for s, t in itertools.combinations(names, 2):
            rows.append((s, t))
    if not rows:
        return pd.DataFrame(columns=["src", "dst", "weight"])
    edges = pd.DataFrame(rows, columns=["src", "dst"])
    edges["pair"] = edges.apply(lambda r: tuple(sorted([r["src"], r["dst"]])), axis=1)
    edges = edges.groupby("pair").size().reset_index(name="weight")
    edges[["src", "dst"]] = pd.DataFrame(edges["pair"].tolist(), index=edges.index)
    edges = edges.drop(columns=["pair"]).sort_values("weight", ascending=False).reset_index(drop=True)
    return edges[["src", "dst", "weight"]]

def centrality_from_edges(edges: pd.DataFrame, metric: str = "degree") -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["著者", "共著数", "つながりスコア"])

    deg_simple = pd.concat([
        edges.groupby("src")["weight"].sum(),
        edges.groupby("dst")["weight"].sum(),
    ], axis=1).fillna(0)
    deg_simple["coauth_count"] = deg_simple["weight"].sum(axis=1)
    deg_simple = deg_simple["coauth_count"].reset_index().rename(columns={"index": "著者", "coauth_count": "共著数"})

    if not HAS_NX:
        out = deg_simple.rename(columns={"共著数": "つながりスコア"})
        return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)

    import networkx as nx
    G = nx.Graph()
    for _, r in edges.iterrows():
        G.add_edge(str(r["src"]), str(r["dst"]), weight=float(r["weight"]))

    if metric == "betweenness":
        cen = nx.betweenness_centrality(G, weight="weight", normalized=True)
    elif metric == "eigenvector":
        try:
            cen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            cen = nx.degree_centrality(G)
    else:
        cen = nx.degree_centrality(G)

    cen_df = pd.Series(cen, name="つながりスコア").reset_index().rename(columns={"index": "著者"})
    out = pd.merge(cen_df, deg_simple, on="著者", how="left")
    out["共著数"] = out["共著数"].fillna(0).astype(float).round().astype(int)
    return out[["著者", "共著数", "つながりスコア"]].sort_values("つながりスコア", ascending=False).reset_index(drop=True)