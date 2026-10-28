# -*- coding: utf-8 -*-
"""
§3.5 커뮤니티 탐지 기반 주제 구조 분석 — Table 3, Figure 6.

입력: config.FILTERED_BLOCKS_JSON
출력:
  - config.COMMUNITY_CSV (Table 3 원본: 상위 n_nodes 기준 39개 커뮤니티가
    final_community_graph/1112 clusters_full_keywords_명은.csv와 거의 일치함을 재실행으로 확인함)
  - config.FIGURE6_PNG (Figure 6: 커뮤니티별 네트워크 시각화)

주의: 이 단계는 PPMI(Positive PMI) 가중치 + 큰 커뮤니티 재귀 분할을 사용한다.
논문 §3.5 본문은 "keyword co-occurrence 그래프"라고만 서술하고 PPMI를 명시하지 않지만,
실제 재실행 비교(상위 커뮤니티 n_nodes 일치, 최대 커뮤니티 키워드 집합 Jaccard=1.0)로 확인된 방법이다.
"""
import os

import config


# -*- coding: utf-8 -*-
"""
연도 컷오프 리스트 기반 그룹화 → 그룹별 PPMI co-occurrence 그래프 → (옵션)엣지 희소화 →
커뮤니티 탐지 → (옵션)큰 커뮤니티 재귀 분할 → 각 그룹별 CSV 저장(클러스터별 '모든' 키워드 포함)

출력(그룹 g마다):
  ./temporal_communities_ppmi/group_{g}/clusters_full_keywords.csv
    - columns: [group_id, years_range, community_id, n_nodes, n_edges,
                density, total_edge_weight, avg_degree, keywords]
    - keywords: strength(가중 차수합) 내림차순으로 정렬된 모든 키워드(콤마로 연결)
"""

import os, json, math
from collections import Counter, defaultdict
from itertools import combinations
from typing import List, Dict

import numpy as np
import pandas as pd
import networkx as nx

# 선택: Louvain / Leiden
try:
    import community as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

try:
    import igraph as ig
    import leidenalg as la
    HAS_LEIDEN = True
except Exception:
    HAS_LEIDEN = False


# ---------------------------
# 0) 연도 컷오프 → 그룹 분할
# ---------------------------
def split_years_into_groups(year_cutoffs: List[int], all_years: List[int]) -> Dict[int, List[int]]:
    """
    year_cutoffs = [2010, 2013, 2016, 2019, 2025]라면
      G1: ≤2010, G2: 2011–2013, G3: 2014–2016, G4: 2017–2019, G5: 2020–2025
    """
    year_cutoffs = sorted(year_cutoffs)
    all_years = sorted(set(int(y) for y in all_years))
    groups = {}
    prev = min(all_years)
    gid = 1
    for cutoff in year_cutoffs:
        yrs = [y for y in all_years if prev <= y <= cutoff]
        if yrs:
            groups[gid] = yrs
            gid += 1
        prev = cutoff + 1
    return groups


# ---------------------------
# 1) 그룹 그래프 (PPMI 가중치)
# ---------------------------
def build_group_graph_ppmi(
    json_path: str,
    years: List[int],
    blocks_field: str = "keywords_filtered_blocks",  # 또는 "keywords_cleaned_blocks"
    min_token_len: int = 1,
    lowercase: bool = False,
    min_cooc: int = 1,          # 동시출현 최소 블록 수
    min_edge_ppmi: float = 0.0, # 엣지 필터 (PPMI 기준)
    alpha: float = 0.0,         # 라플라스 스무딩(필요 시 >0)
    eps: float = 1e-12,
) -> nx.Graph:
    with open(json_path, "r", encoding="utf-8-sig") as f:
        records = json.load(f)

    Y = set(int(y) for y in years)

    def norm(t: str) -> str:
        s = str(t).strip()
        return s.lower() if lowercase else s

    fi = Counter()   # 단어의 블록 수
    fij = Counter()  # (u,v) 동시 블록 수
    W = 0            # 총 블록 수

    for rec in records:
        y = int(rec.get("year", -1))
        if y not in Y:
            continue
        blocks = rec.get(blocks_field, [])
        for blk in blocks:
            if not isinstance(blk, list):
                continue
            terms = sorted({ norm(t) for t in blk if isinstance(t, str) and len(str(t)) >= min_token_len })
            if not terms:
                continue
            W += 1
            for t in terms:
                fi[t] += 1
            for a, b in combinations(terms, 2):
                if a == b:
                    continue
                u, v = (a, b) if a < b else (b, a)
                fij[(u, v)] += 1

    G = nx.Graph()
    for t, f in fi.items():
        G.add_node(t, freq=int(f))

    if W == 0:
        return G

    for (u, v), cij in fij.items():
        if cij < min_cooc:
            continue
        fu, fv = fi.get(u, 0), fi.get(v, 0)

        # 스무딩
        cij_s = cij + alpha
        fu_s  = fu  + alpha
        fv_s  = fv  + alpha
        W_s   = W   + alpha

        # PMI = log( (c_ij * W) / (f_u * f_v) )
        num = (cij_s * W_s)
        den = max(fu_s * fv_s, eps)
        pmi  = np.log(max(num / den, eps))
        ppmi = max(pmi, 0.0)

        if ppmi >= min_edge_ppmi:
            # weight=PPMI, pmi 원값/동시출현 수도 보관
            G.add_edge(u, v, weight=float(ppmi), pmi=float(pmi), cooc=int(cij))
    return G


# ---------------------------
# 1.5) (옵션) 엣지 희소화 (노드 보존)
# ---------------------------
def sparsify_pernode_topk(G: nx.Graph, k: int = 8, weight: str = "weight") -> nx.Graph:
    """
    노드 보존, 엣지만 per-node top-k(가중치 기준) 유지.
    - 양방향 보장: (u,v)는 u에서 top-k이거나 v에서 top-k이면 유지
    """
    keep = set()
    for u in G.nodes():
        nbrs = [(v, float(G[u][v].get(weight, 1.0))) for v in G.neighbors(u)]
        nbrs.sort(key=lambda x: x[1], reverse=True)
        for v,_ in nbrs[:k]:
            a,b = (u,v) if u < v else (v,u)
            keep.add((a,b))
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for (u,v) in keep:
        if G.has_edge(u,v):
            H.add_edge(u, v, **G[u][v])
    return H

def sparsify_disparity_filter(G: nx.Graph, alpha: float = 0.05, weight: str = "weight") -> nx.Graph:
    """
    Serrano et al. (2009) Disparity Filter. 엣지( u-v )의 유의성 p값 < alpha 만 유지.
    노드 보존. 무방향 그래프 기준 양끝 노드 중 더 엄격한 기준 사용.
    """
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))

    for u,v,data in G.edges(data=True):
        w = float(data.get(weight, 1.0))
        # u 쪽 검정
        deg_u = G.degree(u)
        if deg_u <= 1:
            p_u = 0.0
        else:
            su = sum(float(G[u][x].get(weight,1.0)) for x in G.neighbors(u))
            p_u = 1.0 - (1.0 - (w/su))**(deg_u - 1)

        # v 쪽 검정
        deg_v = G.degree(v)
        if deg_v <= 1:
            p_v = 0.0
        else:
            sv = sum(float(G[v][x].get(weight,1.0)) for x in G.neighbors(v))
            p_v = 1.0 - (1.0 - (w/sv))**(deg_v - 1)

        p = max(p_u, p_v)  # 양끝 중 더 보수적으로
        if p < alpha:
            H.add_edge(u, v, **data)
    return H


# ---------------------------
# 2) 커뮤니티 탐지 (Louvain / Leiden)
# ---------------------------
def louvain_partition(G: nx.Graph, resolution: float = 1.0, random_state: int = 42, weight: str = "weight"):
    if not HAS_LOUVAIN:
        raise RuntimeError("python-louvain 미설치: pip install python-louvain")
    return community_louvain.best_partition(G, weight=weight, random_state=random_state, resolution=resolution)

def leiden_partition(
    G: nx.Graph,
    quality: str = "cpm",           # "cpm" | "rb" | "modularity"
    resolution: float = 1.0,
    random_state: int = 42,
    weight: str = "weight",
):
    if not HAS_LEIDEN:
        raise RuntimeError("Leiden 미설치: pip install igraph leidenalg")

    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    igG = ig.Graph(n=len(nodes))
    igG.vs["name"] = nodes
    if G.number_of_edges() > 0:
        igG.add_edges([(idx[u], idx[v]) for u,v in G.edges()])
        igG.es["weight"] = [float(G[u][v].get(weight,1.0)) for u,v in G.edges()]
    else:
        igG.add_edges([])

    if quality == "cpm":
        part = la.find_partition(
            igG, la.CPMVertexPartition, weights="weight",
            seed=random_state, resolution_parameter=resolution
        )
    elif quality == "rb":
        part = la.find_partition(
            igG, la.RBConfigurationVertexPartition, weights="weight",
            seed=random_state, resolution_parameter=resolution
        )
    else:  # modularity
        part = la.find_partition(
            igG, la.ModularityVertexPartition, weights="weight",
            seed=random_state
        )
    return { n: int(c) for n,c in zip(nodes, part.membership) }


# ---------------------------
# 2.5) 큰 커뮤니티 재귀 분할 (노드 보존)
# ---------------------------
def relabel_by_size(comm_map: dict) -> dict:
    size_by_c = Counter(comm_map.values())
    old2new = {old: i+1 for i,(old,_) in enumerate(sorted(size_by_c.items(), key=lambda kv: (-kv[1], kv[0])))}
    return {n: old2new[c] for n,c in comm_map.items()}

def recursive_refine_communities(
    G: nx.Graph,
    base_comm: dict,
    max_size: int = 80,              # 목표 최대 커뮤니티 크기
    quality: str = "cpm",            # Leiden 품질함수
    base_resolution: float = 1.0,    # 시작 resolution
    resolution_step: float = 0.5,    # 재귀 시 올리는 간격
    max_depth: int = 3,
    random_state: int = 42,
    weight: str = "weight",
):
    comm = dict(base_comm)
    depth = 0
    while depth < max_depth:
        buckets = defaultdict(list)
        for n,c in comm.items(): buckets[c].append(n)
        targets = [c for c, nodes in buckets.items() if len(nodes) > max_size]
        if not targets:
            break
        for c in targets:
            nodes = buckets[c]
            H = G.subgraph(nodes).copy()
            if H.number_of_edges() == 0 or H.number_of_nodes() <= max_size:
                continue
            try:
                sub_comm = leiden_partition(
                    H, quality=quality,
                    resolution=base_resolution + resolution_step*(depth+1),
                    random_state=random_state,
                    weight=weight
                )
            except Exception:
                continue
            max_id = max(comm.values())
            sub_ids = sorted(set(sub_comm.values()))
            mapping = { sid: (max_id + 1 + i) for i, sid in enumerate(sub_ids) }
            for n in nodes:
                comm[n] = mapping[sub_comm[n]]
        depth += 1
    return relabel_by_size(comm)


# ---------------------------
# 3) 클러스터별 '모든' 키워드 CSV 저장
# ---------------------------
def save_clusters_full_keywords_csv(
    out_csv_path: str,
    G: nx.Graph,
    comm_map: Dict[str, int],
    group_id: int,
    years: List[int],
):
    rows = []
    for n in G.nodes():
        cid = comm_map[n]
        deg = int(G.degree(n))
        strength = float(sum(G[n][nbr].get("weight", 1.0) for nbr in G.neighbors(n)))
        freq = int(G.nodes[n].get("freq", 0))
        rows.append({"node": n, "community_id": cid, "degree": deg, "strength": strength, "node_freq": freq})
    members = pd.DataFrame(rows)

    out_rows = []
    years_range = f"{years[0]}–{years[-1]}" if len(years) > 1 else f"{years[0]}"
    for cid, sub in members.groupby("community_id"):
        sub_sorted = sub.sort_values(["strength", "degree", "node_freq"], ascending=False)
        keywords_all = sub_sorted["node"].tolist()
        H = G.subgraph(sub["node"].tolist())
        n_nodes = H.number_of_nodes()
        n_edges = H.number_of_edges()
        density = nx.density(H) if n_nodes > 1 else 0.0
        total_w = float(sum(H[u][v].get("weight", 1.0) for u, v in H.edges()))
        avg_deg = float(np.mean([H.degree(n) for n in H.nodes()])) if n_nodes > 0 else 0.0

        out_rows.append({
            "group_id": group_id,
            "years_range": years_range,
            "community_id": int(cid),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "total_edge_weight": total_w,
            "avg_degree": avg_deg,
            "keywords": ", ".join(keywords_all)
        })

    df_out = pd.DataFrame(out_rows).sort_values(["community_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_out.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved: {out_csv_path}  (clusters={df_out.shape[0]})")


# ---------------------------
# 4) 실행 함수 (PPMI 전용, 그룹별 파이프라인)
# ---------------------------
def export_group_cluster_files_ppmi(
    json_path: str,
    year_cutoffs: List[int],                  # 예) [2010, 2013, 2016, 2019, 2025]
    outdir: str = "./temporal_communities_ppmi",
    blocks_field: str = "keywords_filtered_blocks",  # 또는 "keywords_cleaned_blocks"
    min_token_len: int = 1,
    # PPMI 파라미터
    min_cooc: int = 1,
    min_edge_ppmi: float = 0.0,
    alpha: float = 0.0,
    lowercase: bool = False,
    # 희소화
    sparsify_mode: str = "none",      # "none" | "topk" | "disparity"
    topk_k: int = 8,
    disparity_alpha: float = 0.05,
    # 커뮤니티
    comm_method: str = "leiden",      # "louvain" | "leiden"
    quality: str = "cpm",             # leiden일 때만 사용: "cpm"|"rb"|"modularity"
    resolution: float = 1.0,
    random_state: int = 42,
    # 큰 커뮤니티 재귀 분할
    refine: bool = True,
    refine_max_size: int = 80,
    refine_depth: int = 3,
    resolution_step: float = 0.5,
):
    # 전체 연도 수집
    with open(json_path, "r", encoding="utf-8-sig") as f:
        records = json.load(f)
    all_years = sorted({int(r["year"]) for r in records if "year" in r})

    # 컷오프 → 그룹
    groups = split_years_into_groups(year_cutoffs, all_years)
    print("Groups:", groups)
    print("Weight mode: PPMI")

    for g, yrs in groups.items():
        print(f"\n[Group {g}] years={yrs}")
        # 1) PPMI 그래프
        G = build_group_graph_ppmi(
            json_path=json_path,
            years=yrs,
            blocks_field=blocks_field,
            min_token_len=min_token_len,
            lowercase=lowercase,
            min_cooc=min_cooc,
            min_edge_ppmi=min_edge_ppmi,
            alpha=alpha
        )
        print(f"  Graph(raw): |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")

        if G.number_of_nodes() == 0:
            print("  ⛔️ Empty graph — skipped")
            continue

        # 1.5) (옵션) 희소화
        if sparsify_mode == "topk":
            G = sparsify_pernode_topk(G, k=topk_k, weight="weight")
            print(f"  Sparsify[per-node top-{topk_k}]: |E|={G.number_of_edges():,}")
        elif sparsify_mode == "disparity":
            G = sparsify_disparity_filter(G, alpha=disparity_alpha, weight="weight")
            print(f"  Sparsify[disparity α={disparity_alpha}]: |E|={G.number_of_edges():,}")
        else:
            print("  Sparsify: none")

        # 2) 1차 커뮤니티
        if comm_method == "louvain":
            comm0 = louvain_partition(G, resolution=resolution, random_state=random_state, weight="weight")
        else:
            comm0 = leiden_partition(G, quality=quality, resolution=resolution, random_state=random_state, weight="weight")

        # 2.5) 재귀 분할(큰 커뮤니티만)
        if refine:
            comm = recursive_refine_communities(
                G, comm0,
                max_size=refine_max_size,
                quality=quality,                # Leiden 사용
                base_resolution=resolution,
                resolution_step=resolution_step,
                max_depth=refine_depth,
                random_state=random_state,
                weight="weight"
            )
        else:
            comm = comm0

        # 크기 기준 재라벨 → 정렬 안정화
        size_by_c = Counter(comm.values())
        old2new = {old: i+1 for i, (old, _) in enumerate(sorted(size_by_c.items(), key=lambda kv: (-kv[1], kv[0])))}
        comm = {n: old2new[c] for n, c in comm.items()}

        # 3) CSV 저장
        gdir = os.path.join(outdir, f"group_{g}")
        out_csv = os.path.join(gdir, "clusters_full_keywords.csv")
        save_clusters_full_keywords_csv(out_csv, G, comm, group_id=g, years=yrs)


import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
import os

def plot_group_clusters_from_graph(
    G: nx.Graph,
    clusters_csv_path: str,
    group_id: int,
    min_keywords_per_cluster: int = 15,
    layout: str = "spring",      # 'spring' | 'kamada'
    use_core_neighbor_labels: bool = True,
    seed: int = 42,
    font_path: str | None = None,    # "./fonts/NanumGothic.otf" 이런 식
    font_name: str | None = None,    # "NanumGothic"
    out_png: str | None = None,
):
    """
    이미 만들어진 PPMI 그래프 G 위에,
    - group_id에 해당하는 클러스터에 속한 노드만 모은 subgraph H를 그리고
    - '핵심 키워드 + 그 이웃'만 라벨로 표시.

    반환:
      - H: 선택된 클러스터 노드들로 만든 서브그래프
    """

    # 🔹 폰트 설정(있으면)
    fp = FontProperties(fname=font_path) if font_path is not None else None

    # 1) 클러스터 정보 읽기
    cid2kws, cid2core = load_cluster_keywords_with_core(
        clusters_csv_path,
        group_id=group_id,
        min_keywords=min_keywords_per_cluster,
        core_col="핵심 키워드",
    )

    # 2) 선택된 클러스터에 들어있는 노드들만 모으기
    keep_nodes = set().union(*[set(kws) for kws in cid2kws.values()])
    keep_nodes = [n for n in keep_nodes if n in G]

    H = G.subgraph(keep_nodes).copy()
    if H.number_of_nodes() == 0:
        raise ValueError("선택된 클러스터 키워드가 G에 없습니다. (G와 CSV 세팅 일치 확인)")

    # 3) 레이아웃
    if layout == "kamada":
        pos = nx.kamada_kawai_layout(H, weight="weight")
    else:
        pos = nx.spring_layout(H, seed=seed, k=None, weight="weight")

    # 4) 클러스터별 색상
    cids = sorted(cid2kws.keys())
    cmap = cm.get_cmap("tab20", len(cids))
    cid2color = {cid: cmap(i) for i, cid in enumerate(cids)}

    # 5) 그리기 시작
    plt.figure(figsize=(20, 10))

    # 엣지 먼저
    nx.draw_networkx_edges(
        H, pos,
        width=[0.3 + 0.3 * float(H[u][v].get("weight", 1.0)) for u, v in H.edges()],
        alpha=0.15,
        edge_color="#888888",
    )

    # 노드: 클러스터별 색
    for cid, kws in cid2kws.items():
        nodes_here = [n for n in kws if n in H]
        if not nodes_here:
            continue

        nx.draw_networkx_nodes(
            H, pos,
            nodelist=nodes_here,
            node_color=[cid2color[cid]],
            node_size=[80 + 20 * H.degree(n) for n in nodes_here],
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
        )

    # 6) 라벨: '핵심 키워드 + 그 이웃'
    if use_core_neighbor_labels:
        label_nodes_global = set()

        for cid, core_list in cid2core.items():
            if not core_list:
                continue

            core_here = [n for n in core_list if n in H]
            if not core_here:
                continue

            neighbors_here = set()
            for c in core_here:
                neighbors_here.update(H.neighbors(c))

            label_nodes_global.update(core_here)
            label_nodes_global.update(neighbors_here)

        labels = {n: n for n in label_nodes_global if n in H}

        label_kwargs = {
            "font_size": 9,
            "font_color": "black",
        }
        if font_name is not None:
            label_kwargs["font_family"] = font_name

        texts = nx.draw_networkx_labels(
            H, pos,
            labels=labels,
            **label_kwargs,
        )

        # font_path가 있으면 Text 객체에 직접 FontProperties 주입
        if fp is not None:
            for t in texts.values():
                t.set_fontproperties(fp)

    # 7) 범례
    handles = [
        Patch(color=cid2color[cid], label=f"C{cid} (|kw|≥{min_keywords_per_cluster})")
        for cid in cids
    ]
    legend_kwargs = {
        "handles": handles,
        "loc": "upper right",
        "fontsize": 9,
        "frameon": False,
    }
    if fp is not None:
        legend_kwargs["prop"] = fp

    plt.legend(**legend_kwargs)

    plt.axis("off")
    plt.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300)
        print(f"💾 Saved: {out_png}")

    plt.show()

    return H


# ============================================================
# 실행 래퍼
# ============================================================

def run_table3():
    """Table 3 원본 커뮤니티 생성 (전체 기간을 단일 그룹으로 처리)."""
    config.COMMUNITY_OUTDIR.mkdir(exist_ok=True)

    export_group_cluster_files_ppmi(
        json_path=str(config.FILTERED_BLOCKS_JSON),
        year_cutoffs=[2025],  # 하나의 컷오프 = 전체 기간을 단일 그룹(group_id=1)으로 처리
        outdir=str(config.COMMUNITY_OUTDIR),
        blocks_field="keywords_cleaned_blocks",
        min_token_len=1,
        # --- PPMI ---
        min_cooc=1,
        min_edge_ppmi=0.0,
        alpha=0.0,
        lowercase=False,
        # --- 희소화 ---
        sparsify_mode="none",
        topk_k=8,
        disparity_alpha=0.05,
        # --- 커뮤니티 ---
        comm_method="louvain",
        quality="cpm",
        resolution=1.0,
        random_state=1,
        # --- 재귀 분할 ---
        refine=True,
        refine_max_size=80,
        refine_depth=3,
        resolution_step=0.5,
    )
    print(f"[Table 3] 저장 완료: {config.COMMUNITY_CSV}")


def run_figure6(group_id: int = 1, min_keywords_per_cluster: int = 15):
    """Figure 6: Table 3와 동일한 PPMI 그래프에 핵심 키워드+이웃만 라벨링해 시각화."""
    years = list(range(2005, 2026))
    G = build_group_graph_ppmi(
        json_path=str(config.FILTERED_BLOCKS_JSON),
        years=years,
        blocks_field="keywords_cleaned_blocks",
        min_token_len=1,
        min_cooc=1,
        min_edge_ppmi=0.0,
        alpha=0.0,
        lowercase=False,
    )

    plot_group_clusters_from_graph(
        G,
        clusters_csv_path=str(config.COMMUNITY_CSV),
        group_id=group_id,
        min_keywords_per_cluster=min_keywords_per_cluster,
        layout="spring",
        use_core_neighbor_labels=True,
        font_path=str(config.FONT_PATH) if config.FONT_PATH.exists() else None,
        font_name="NanumGothic",
        out_png=str(config.FIGURE6_PNG),
    )
    print(f"[Figure 6] 저장 완료: {config.FIGURE6_PNG}")


def run():
    run_table3()
    run_figure6()


if __name__ == "__main__":
    run()
