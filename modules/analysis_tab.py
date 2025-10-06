# modules/analysis_tab.py
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def render_analysis_tab(df: pd.DataFrame):
    st.header("📊 分析ツール")

    st.markdown("### 著者共起ネットワーク（中心性分析付き）")

    # 共起処理
    edges = []
    for authors in df["著者"].dropna():
        names = [a.strip() for a in re.split(r"[;；,、，/／|｜]+", authors) if a.strip()]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                edges.append(tuple(sorted((names[i], names[j]))))

    edges_df = pd.DataFrame(edges, columns=["a", "b"]).value_counts().reset_index(name="count")
    edges_df = edges_df[edges_df["count"] >= 2]

    # ネットワーク構築
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["a"], row["b"], weight=row["count"])

    # 中心性ランキング
    centrality = nx.degree_centrality(G)
    top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    st.dataframe(pd.DataFrame(top_centrality, columns=["著者", "中心性"]))

    # ネットワーク描画
    pos = nx.spring_layout(G, k=0.3)
    edge_x, edge_y, node_x, node_y, node_text = [], [], [], [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node} ({centrality[node]:.3f})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#aaa')))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             text=node_text, textposition="top center",
                             marker=dict(size=10, color='skyblue', line_width=1)))
    fig.update_layout(showlegend=False, height=700, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)