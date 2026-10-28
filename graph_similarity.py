# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import config


# -*- coding: utf-8 -*-
import json, os, warnings
from collections import defaultdict, Counter
from itertools import combinations

import igraph as ig
import leidenalg as la
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine
from scipy.linalg import eigh

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# (선택) netlsd가 설치되어 있으면 사용, 아니면 None
try:
    import netlsd
except Exception:
    netlsd = None


class FourMethodSimilarityAnalyzer:
    """
    - Method 1: Node Jaccard (유사도)
    - Method 2: Edge Jaccard (유사도)
    - Method 3: Structural Features (유사도)  ← 필요시 사용
    - Method 4: NetLSD (거리): 우선 netlsd.heat 사용, 에러나면 로컬 구현으로 동일 서명 계산
      * 플롯은 NetLSD는 '거리'를 그대로 쓰고, 색만 반전(Reds_r)
    """
    def __init__(self, json_path, outdir="./outputs"):
        with open(json_path, 'r', encoding='utf-8-sig') as f:
            self.data = json.load(f)
        self.yearly_graphs = {}         # year -> ig.Graph
        self.yearly_communities = {}    # year -> leiden partition
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    # -------------------------------------------------------
    # 0) 연도별 co-occurrence 그래프 생성
    # -------------------------------------------------------
    def build_yearly_graphs(self, weight_min=1, blocks_field="keywords_cleaned_blocks"):
        """
        연도별 키워드 co-occurrence 그래프 생성
        - weight_min: 엣지 최소 가중치 필터
        - blocks_field: JSON에서 키워드 블록이 들어있는 필드명
        """
        yearly_edges = defaultdict(Counter)
        for doc in self.data:
            year = int(doc['year'])
            for block in doc.get(blocks_field, []):
                if not isinstance(block, list) or len(block) < 2:
                    continue
                for kw1, kw2 in combinations(sorted(map(str, block)), 2):
                    yearly_edges[year][(kw1, kw2)] += 1

        for year, edge_counter in yearly_edges.items():
            items = [((a, b), w) for (a, b), w in edge_counter.items() if w >= weight_min]
            if not items:
                continue

            nodes = set()
            edges, weights = [], []
            for (a, b), w in items:
                nodes.update([a, b])
                edges.append((a, b))
                weights.append(float(w))

            node_list = sorted(nodes)                 # 순서 고정
            idx = {n: i for i, n in enumerate(node_list)}

            g = ig.Graph()
            g.add_vertices(node_list)                 # name 속성 자동 채워짐
            g.add_edges([(idx[a], idx[b]) for (a, b) in edges])
            g.es['weight'] = weights
            g.vs['name'] = node_list
            self.yearly_graphs[year] = g

        print(f"연도별 그래프 생성 완료: {len(self.yearly_graphs)}개 연도")

    # -------------------------------------------------------
    # 1) Leiden 커뮤니티 탐지 (옵션)
    # -------------------------------------------------------
    def detect_communities(self, seed=42, method="modularity", resolution=1.0):
        for year, g in self.yearly_graphs.items():
            if method == "rb":
                part = la.find_partition(
                    g, la.RBConfigurationVertexPartition,
                    weights='weight', n_iterations=-1, seed=seed,
                    resolution_parameter=resolution
                )
            elif method == "cpm":
                part = la.find_partition(
                    g, la.CPMVertexPartition,
                    weights='weight', n_iterations=-1, seed=seed,
                    resolution_parameter=resolution
                )
            else:  # modularity
                part = la.find_partition(
                    g, la.ModularityVertexPartition,
                    weights='weight', n_iterations=-1, seed=seed
                )
            self.yearly_communities[year] = part
        print("커뮤니티 탐지 완료")

    # -------------------------------------------------------
    # Method 1: Node Jaccard (유사도)
    # -------------------------------------------------------
    def similarity_node_jaccard(self):
        years = sorted(self.yearly_graphs.keys())
        n = len(years)
        S = np.zeros((n, n))
        for i, y1 in enumerate(years):
            n1 = set(self.yearly_graphs[y1].vs['name'])
            for j, y2 in enumerate(years):
                if i == j:
                    S[i, j] = 1.0
                else:
                    n2 = set(self.yearly_graphs[y2].vs['name'])
                    inter = len(n1 & n2); union = len(n1 | n2)
                    S[i, j] = inter / union if union > 0 else 0.0
        return S, years, "Method 1: Node Jaccard (Similarity)"

    # -------------------------------------------------------
    # Method 2: Edge Jaccard (유사도)
    # -------------------------------------------------------
    def similarity_edge_jaccard(self, weighted=False):
        years = sorted(self.yearly_graphs.keys())
        n = len(years)
        S = np.zeros((n, n))

        def weighted_jaccard(d1, d2):
            keys = set(d1.keys()) | set(d2.keys())
            inter = sum(min(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
            union = sum(max(d1.get(k, 0.0), d2.get(k, 0.0)) for k in keys)
            return inter / union if union > 0 else 0.0

        for i, y1 in enumerate(years):
            g1 = self.yearly_graphs[y1]
            nodes1 = set(g1.vs['name'])
            for j, y2 in enumerate(years):
                if i == j:
                    S[i, j] = 1.0; continue
                g2 = self.yearly_graphs[y2]
                common = nodes1 & set(g2.vs['name'])
                if not common:
                    S[i, j] = 0.0; continue

                sub1 = g1.subgraph([g1.vs.find(name=n).index for n in common])
                sub2 = g2.subgraph([g2.vs.find(name=n).index for n in common])

                if weighted:
                    e1, e2 = {}, {}
                    for e in sub1.es:
                        a, b = sorted([sub1.vs[e.source]['name'], sub1.vs[e.target]['name']])
                        w = float(e['weight']) if 'weight' in e.attributes() else 1.0
                        e1[(a, b)] = e1.get((a, b), 0.0) + w
                    for e in sub2.es:
                        a, b = sorted([sub2.vs[e.source]['name'], sub2.vs[e.target]['name']])
                        w = float(e['weight']) if 'weight' in e.attributes() else 1.0
                        e2[(a, b)] = e2.get((a, b), 0.0) + w
                    S[i, j] = weighted_jaccard(e1, e2)
                else:
                    E1, E2 = set(), set()
                    for e in sub1.es:
                        a, b = sorted([sub1.vs[e.source]['name'], sub1.vs[e.target]['name']])
                        E1.add((a, b))
                    for e in sub2.es:
                        a, b = sorted([sub2.vs[e.source]['name'], sub2.vs[e.target]['name']])
                        E2.add((a, b))
                    inter, union = len(E1 & E2), len(E1 | E2)
                    S[i, j] = inter / union if union > 0 else 0.0

        return S, years, f"Method 2: Edge Jaccard ({'Weighted' if weighted else 'Unweighted'})"

    # -------------------------------------------------------
    # Method 3: Structural Features (유사도)  ← 필요시 사용
    # -------------------------------------------------------
    def similarity_structural_features(self):
        years = sorted(self.yearly_graphs.keys())
        feats = []
        for y in years:
            g = self.yearly_graphs[y]
            vcount, ecount = g.vcount(), g.ecount()
            density = g.density() if vcount > 1 else 0.0
            deg = np.array(g.degree(), dtype=float)
            deg_mean = float(np.mean(deg)) if vcount > 0 else 0.0
            deg_std  = float(np.std(deg))  if vcount > 0 else 0.0
            clustering = g.transitivity_undirected() or 0.0
            assort = g.assortativity_degree()
            assort = float(assort) if np.isfinite(assort) else 0.0

            if hasattr(g, "components"):
                comps = g.components()
            else:
                comps = g.connected_components()
            n_comps = len(comps) if comps is not None else 0
            giant_ratio = max(len(c) for c in comps) / vcount if (vcount > 0 and n_comps > 0) else 0.0

            if vcount > 0 and deg.size > 0:
                hist = np.bincount(deg.astype(int), minlength=int(deg.max())+1)
                p = hist / hist.sum() if hist.sum() > 0 else hist
                deg_entropy = float(-(p[p>0] * np.log(p[p>0])).sum()) if p.sum() > 0 else 0.0
            else:
                deg_entropy = 0.0

            if (hasattr(g, "is_connected") and g.is_connected()) and vcount > 1:
                avg_path = g.average_path_length()
                diam = g.diameter()
            else:
                avg_path = 0.0; diam = 0.0

            feats.append([
                vcount, ecount, density, deg_mean, deg_std,
                clustering, assort, n_comps, giant_ratio,
                deg_entropy, avg_path, diam,
            ])

        X = np.array(feats, dtype=float)
        Xn = StandardScaler().fit_transform(X)
        n = len(years)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                S[i, j] = 1.0 if i == j else 1 - cosine(Xn[i], Xn[j])
        return S, years, "Method 3: Structural Features (Extended)"

    # -------------------------------------------------------
    # NetLSD Heat signature (로컬 우회 구현)
    # -------------------------------------------------------
    @staticmethod
    def _heat_signature_local(A, timescales, binarize=False, k_eigs=None):
        """
        정규화 라플라시안 L = I - D^{-1/2} A D^{-1/2} 의 고유값으로 heat signature 계산.
        - signature(t) = mean(exp(-t * lambda_i))  (size-normalized)
        - k_eigs: 사용할 최소 고유값 개수 (정수). None 이면 전체 사용.
                  SciPy eigh(subset_by_index) 사용, 실패 시 전체로 대체.
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        if n == 0:
            return np.zeros_like(timescales, dtype=float)
        if binarize:
            A = (A > 0).astype(float)
            np.fill_diagonal(A, 0.0)

        deg = np.asarray(A.sum(axis=1), dtype=float)
        with np.errstate(divide='ignore'):
            d_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 0, None))
        d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        # 고유값 구하기
        try:
            if k_eigs is not None and 1 <= k_eigs <= n:
                evals = eigh(L, eigvals_only=True, subset_by_index=(0, k_eigs-1))
            else:
                evals = eigh(L, eigvals_only=True)
        except Exception:
            # subset_by_index 미지원 환경 대비: 전체
            evals = eigh(L, eigvals_only=True)

        evals = np.sort(np.real(evals))
        # Heat trace의 평균(크기 정규화)
        sig = np.array([float(np.mean(np.exp(-t * evals))) for t in timescales], dtype=float)
        return sig

    # -------------------------------------------------------
    # Method 4: NetLSD — 거리 행렬 그대로 반환 (에러 시 로컬 우회)
    # -------------------------------------------------------
    def netlsd_distance(self, timescales=None, binarize=False, k_eigs=None, force_local=False):
        """
        NetLSD L2 거리(D) 행렬을 그대로 반환 (유사도 변환 X).
        - 기본: netlsd.heat 사용
        - TypeError/AttributeError 등 발생 시: _heat_signature_local 로 우회
        - timescales: 기본 logspace(-2, 2, 256)
        - binarize=True면 가중치 무시(구조만 비교)
        - k_eigs: 로컬 우회 시 사용할 최소 고유값 개수(정수)
        - force_local=True면 무조건 로컬 우회 사용
        """
        years = sorted(self.yearly_graphs.keys())
        n = len(years)
        if timescales is None:
            timescales = np.logspace(-2, 2, 256)

        desc = {}
        use_local = force_local or (netlsd is None)

        for y in years:
            A = np.array(self.yearly_graphs[y].get_adjacency().data, dtype=float)
            if A.size == 0:
                desc[y] = np.zeros_like(timescales, dtype=float)
                continue

            if use_local:
                desc[y] = self._heat_signature_local(A, timescales, binarize=binarize, k_eigs=k_eigs)
            else:
                try:
                    if binarize:
                        A_bin = (A > 0).astype(float)
                        np.fill_diagonal(A_bin, 0.0)
                        desc[y] = netlsd.heat(A_bin, timescales=timescales)
                    else:
                        desc[y] = netlsd.heat(A, timescales=timescales)
                except (TypeError, AttributeError, ValueError):
                    # 호환성 에러 → 로컬 우회
                    desc[y] = self._heat_signature_local(A, timescales, binarize=binarize, k_eigs=k_eigs)

        # L2 거리
        Y = list(years)
        D = np.zeros((n, n))
        for i, y1 in enumerate(Y):
            for j, y2 in enumerate(Y):
                if i == j:
                    D[i, j] = 0.0
                else:
                    v1, v2 = desc[y1], desc[y2]
                    D[i, j] = float(np.linalg.norm(v1 - v2))
        return D, years, "Method 4: NetLSD (L2 distance)"

    # -------------------------------------------------------
    # 플롯: (좌) Node 유사도 / (중) Edge 유사도 / (우) NetLSD 거리(색 반전)
    # -------------------------------------------------------
    def plot_node_edge_similarity_and_netlsd_distance(
        self,
        filename='year_NODE_EDGE_similarity__NETLSD_distance.png',
        weighted_edge=False,
        binarize_netlsd=False,
        timescales=None,
        k_eigs=None,
        force_local=False
    ):
        # 계산
        S_node,  years, _ = self.similarity_node_jaccard()
        S_edge,  _,     _ = self.similarity_edge_jaccard(weighted=weighted_edge)
        D_netlsd, _,   _ = self.netlsd_distance(
            timescales=timescales,
            binarize=binarize_netlsd,
            k_eigs=k_eigs,
            force_local=force_local
        )

        # 보기 좋게 각 행렬별로 vmin/vmax를 동적 설정
        def nice_range(M):
            m, Mx = float(np.min(M)), float(np.max(M))
            if m == Mx:
                return m, Mx + 1e-6
            pad = 0.02 * (Mx - m)
            return m - pad, Mx + pad

        vmin_Sn, vmax_Sn = nice_range(S_node)
        vmin_Se, vmax_Se = nice_range(S_edge)
        vmin_Dn, vmax_Dn = nice_range(D_netlsd)

        fig = plt.figure(figsize=(24, 7))

        # (좌) Node Jaccard (Similarity)
        ax1 = fig.add_subplot(1, 3, 1)
        sns.heatmap(S_node, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='Reds', vmin=vmin_Sn, vmax=vmax_Sn,
                    cbar_kws={'label': 'Similarity'},
                    ax=ax1, square=True)
        ax1.set_title("Method 1: Node Jaccard (Similarity)", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Year'); ax1.set_ylabel('Year')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # (중) Edge Jaccard (Similarity)
        ax2 = fig.add_subplot(1, 3, 2)
        sns.heatmap(S_edge, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='Reds', vmin=vmin_Se, vmax=vmax_Se,
                    cbar_kws={'label': 'Similarity'},
                    ax=ax2, square=True)
        ax2.set_title(f"Method 2: Edge Jaccard (Similarity, {'Weighted' if weighted_edge else 'Unweighted'})",
                      fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Year'); ax2.set_ylabel('Year')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        # (우) NetLSD (Distance) — 색 반전
        ax3 = fig.add_subplot(1, 3, 3)
        sns.heatmap(D_netlsd, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='Reds_r', vmin=vmin_Dn, vmax=vmax_Dn,
                    cbar_kws={'label': 'Distance (lower = more similar)'},
                    ax=ax3, square=True)
        ttl = "Method 4: NetLSD (L2 Distance)"
        if binarize_netlsd:
            ttl += " [Binarized A]"
        if force_local:
            ttl += " [Local]"
        ax3.set_title(ttl, fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Year'); ax3.set_ylabel('Year')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

        plt.tight_layout()
        out = os.path.join(self.outdir, filename)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n그림 저장됨 → {out}")

    # -------------------------------------------------------
    # (옵션) 기존 3-패널(Edge/Structural/NetLSD=유사도 변환) 그림도 만들고 싶다면
    # -------------------------------------------------------
    def similarity_netlsd_similarity(self, timescales=None, **kwargs):
        # 거리→유사도(1/(1+d))로 변환한 버전
        D, years, _ = self.netlsd_distance(timescales=timescales, **kwargs)
        S = 1.0 / (1.0 + D)
        return S, years, "Method 4: NetLSD (L2→Similarity)"

    def plot_all_similarities(self, filename='year_similarity_EDGE_STRUCT_NETLSD_S.png'):
        mats = [
            lambda: self.similarity_edge_jaccard(weighted=False),
            self.similarity_structural_features,
            self.similarity_netlsd_similarity,   # ← 유사도 변환 버전
        ]
        fig = plt.figure(figsize=(24, 7))
        for idx, method in enumerate(mats):
            sim_matrix, years, title = method()
            ax = fig.add_subplot(1, 3, idx + 1)
            sns.heatmap(sim_matrix,
                        annot=True, fmt='.2f',
                        xticklabels=years, yticklabels=years,
                        cmap='Reds',
                        cbar_kws={'label': 'Similarity'},
                        ax=ax, square=True)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Year'); ax.set_ylabel('Year')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout()
        out = os.path.join(self.outdir, filename)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n히트맵 저장됨 → {out}")

    # -------------------------------------------------------
    # (옵션) 콘솔 요약
    # -------------------------------------------------------
    def analyze_similarity_trends(self):
        print("\n" + "="*70)
        print("연도 간 유사도/거리 경향 분석")
        print("="*70)
        methods = [
            ("Node Jaccard (Similarity)", self.similarity_node_jaccard),
            ("Edge Jaccard (Similarity, Unweighted)", lambda: self.similarity_edge_jaccard(weighted=False)),
            ("NetLSD (Distance, Local Fallback)", lambda: self.netlsd_distance()),
        ]
        for name, func in methods:
            M, years, _ = func()
            tri = M[np.triu_indices_from(M, k=1)]
            adj = [M[i, i+1] for i in range(len(years)-1)] if len(years) > 1 else []
            print(f"\n{name}:")
            if adj:
                mean_adj, std_adj = float(np.mean(adj)), float(np.std(adj))
                print(f"  - 인접 연도 평균: {mean_adj:.3f} (±{std_adj:.3f})")
            print(f"  - 전체 평균: {float(np.mean(tri)):.3f}")
            print(f"  - 중앙값: {float(np.median(tri)):.3f}")
            r, c = np.triu_indices_from(M, k=1)
            mx, mn = np.argmax(tri), np.argmin(tri)
            print(f"  - 최댓값 쌍: {years[r[mx]]}-{years[c[mx]]} ({M[r[mx], c[mx]]:.3f})")
            print(f"  - 최솟값 쌍: {years[r[mn]]}-{years[c[mn]]} ({M[r[mn], c[mn]]:.3f})")

    # -------------------------------------------------------
    # (옵션) JSON 저장
    # -------------------------------------------------------
    def save_similarity_matrices(self, filename='similarity_matrices.json'):
        results = {}
        for key, func in [
            ("node_jaccard", self.similarity_node_jaccard),
            ("edge_jaccard_unweighted", lambda: self.similarity_edge_jaccard(weighted=False)),
            ("netlsd_distance", self.netlsd_distance),
        ]:
            M, years, _ = func()
            results[key] = {'years': years, 'matrix': M.tolist()}
        out = os.path.join(self.outdir, filename)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n유사도/거리 매트릭스 JSON 저장됨 → {out}")

    # --- 클래스 내부에 추가 ---
    def netlsd_similarity(self, timescales=None, binarize=False, k_eigs=None,
                        force_local=False, kernel="exp", tau=None):
        """
        NetLSD 거리(D)를 유사도(S)로 변환해서 반환.
        - kernel:
            * "exp" : S = exp(-D / tau)     ← 권장 (감쇠형)
            * "inv" : S = 1 / (1 + D / tau) ← 완만, 0~1 보장
        - tau: 스케일 파라미터. None이면 오프대각 D의 중앙값 사용(robust).
        """
        D, years, _ = self.netlsd_distance(timescales=timescales,
                                        binarize=binarize,
                                        k_eigs=k_eigs,
                                        force_local=force_local)

        # 오프대각선만 모아서 스케일 추정
        tri = D[np.triu_indices_from(D, k=1)]
        if tau is None:
            # 너무 0에 붙으면 약간의 최소값 부여
            tau = float(np.median(tri)) if tri.size else 1.0
            tau = max(tau, 1e-8)

        if kernel == "exp":
            S = np.exp(-D / tau)
        elif kernel == "inv":
            S = 1.0 / (1.0 + D / tau)
        else:
            raise ValueError("kernel must be 'exp' or 'inv'")

        # 대각은 정확히 1로 맞춤 (보기 좋게)
        np.fill_diagonal(S, 1.0)
        return S, years, f"Method 4: NetLSD → Similarity ({kernel}, tau={tau:.3g})"


    # --- 클래스 내부에 추가 ---
    def plot_node_edge_and_netlsd_similarity(
        self,
        filename='year_NODE_EDGE_NETLSD_similarity.png',
        weighted_edge=False,
        binarize_netlsd=False,
        timescales=None,
        k_eigs=None,
        force_local=False,
        kernel="exp",
        tau=None
    ):
        S_node,  years, _ = self.similarity_node_jaccard()
        S_edge,  _,     _ = self.similarity_edge_jaccard(weighted=weighted_edge)
        S_netlsd, _,  ttl = self.netlsd_similarity(timescales=timescales,
                                                binarize=binarize_netlsd,
                                                k_eigs=k_eigs,
                                                force_local=force_local,
                                                kernel=kernel,
                                                tau=tau)

        def nice_range(M):
            m, Mx = float(np.min(M)), float(np.max(M))
            if m == Mx:
                return m, Mx + 1e-6
            pad = 0.02 * (Mx - m)
            return m - pad, Mx + pad

        vmin_Sn, vmax_Sn = nice_range(S_node)
        vmin_Se, vmax_Se = nice_range(S_edge)
        vmin_Sl, vmax_Sl = nice_range(S_netlsd)

        fig = plt.figure(figsize=(24, 7))

        ax1 = fig.add_subplot(1, 3, 1)
        sns.heatmap(S_node, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='rocket_r', vmin=vmin_Sn, vmax=vmax_Sn,
                    cbar_kws={'label': 'Similarity'},
                    ax=ax1, square=True)
        ax1.set_title("Method 1: Node Jaccard (Similarity)", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Year'); ax1.set_ylabel('Year')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        ax2 = fig.add_subplot(1, 3, 2)
        sns.heatmap(S_edge, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='rocket_r', vmin=vmin_Se, vmax=vmax_Se,
                    cbar_kws={'label': 'Similarity'},
                    ax=ax2, square=True)
        ax2.set_title(f"Method 2: Edge Jaccard (Similarity, {'Weighted' if weighted_edge else 'Unweighted'})",
                    fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Year'); ax2.set_ylabel('Year')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        ax3 = fig.add_subplot(1, 3, 3)
        sns.heatmap(S_netlsd, annot=True, fmt='.2f',
                    xticklabels=years, yticklabels=years,
                    cmap='rocket_r', vmin=vmin_Sl, vmax=vmax_Sl,
                    cbar_kws={'label': 'Similarity'},
                    ax=ax3, square=True)
        ax3.set_title(ttl, fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Year'); ax3.set_ylabel('Year')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)

        plt.tight_layout()
        out = os.path.join(self.outdir, filename)
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"\n그림 저장됨 → {out}")

    def save_node_edge_and_netlsd_similarity_matrices(
        self,
        prefix='year_NODE_EDGE_NETLSD',
        weighted_edge=False,
        binarize_netlsd=False,
        timescales=None,
        k_eigs=None,
        force_local=False,
        kernel="exp",
        tau=None,
        save_csv=True,
        save_npy=False,
        save_distance_also=True  # NetLSD 거리행렬도 같이 저장할지
    ):
        """
        Node Jaccard / Edge Jaccard(유사도) / NetLSD(유사도, 선택적으로 거리) 행렬 저장
        - JSON: 세 행렬을 한 파일로 묶어 저장
        - CSV: 각 행렬을 연도 라벨 포함 테이블로 개별 저장
        - NPY: 넘파이 원시 배열로 저장(옵션)
        """
        # 1) 계산
        S_node,  years, _  = self.similarity_node_jaccard()
        S_edge,  _,     _  = self.similarity_edge_jaccard(weighted=weighted_edge)
        S_netlsd, _, ttl   = self.netlsd_similarity(timescales=timescales,
                                                binarize=binarize_netlsd,
                                                k_eigs=k_eigs,
                                                force_local=force_local,
                                                kernel=kernel,
                                                tau=tau)

        results = {
            "years": years,
            "node_jaccard_similarity": S_node.tolist(),
            "edge_jaccard_similarity": S_edge.tolist(),
            "netlsd_similarity": S_netlsd.tolist(),
            "netlsd_similarity_meta": {
                "title": ttl,
                "kernel": kernel,
                "tau": None if tau is None else float(tau),
                "binarize": bool(binarize_netlsd)
            }
        }

        # (옵션) NetLSD 거리도 함께 저장
        if save_distance_also:
            D_netlsd, _, _ = self.netlsd_distance(timescales=timescales,
                                                binarize=binarize_netlsd,
                                                k_eigs=k_eigs,
                                                force_local=force_local)
            results["netlsd_distance"] = D_netlsd.tolist()

        # 2) JSON 한 방에
        json_path = os.path.join(self.outdir, f"{prefix}__matrices.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 3) CSV (연도 라벨 포함)
        if save_csv:
            def to_df(M): return pd.DataFrame(M, index=years, columns=years)
            to_df(S_node).to_csv(os.path.join(self.outdir, f"{prefix}__node_jaccard_similarity.csv"), encoding="utf-8-sig")
            to_df(S_edge).to_csv(os.path.join(self.outdir, f"{prefix}__edge_jaccard_similarity.csv"), encoding="utf-8-sig")
            to_df(S_netlsd).to_csv(os.path.join(self.outdir, f"{prefix}__netlsd_similarity_{kernel}.csv"), encoding="utf-8-sig")
            if save_distance_also:
                to_df(D_netlsd).to_csv(os.path.join(self.outdir, f"{prefix}__netlsd_distance.csv"), encoding="utf-8-sig")

        # 4) NPY (원시 배열)
        if save_npy:
            np.save(os.path.join(self.outdir, f"{prefix}__node_jaccard_similarity.npy"), S_node)
            np.save(os.path.join(self.outdir, f"{prefix}__edge_jaccard_similarity.npy"), S_edge)
            np.save(os.path.join(self.outdir, f"{prefix}__netlsd_similarity_{kernel}.npy"), S_netlsd)
            if save_distance_also:
                np.save(os.path.join(self.outdir, f"{prefix}__netlsd_distance.npy"), D_netlsd)

        print("✅ 저장 완료:")
        print(" - JSON :", json_path)
        if save_csv:
            print(" - CSV  :", "(node/edge/netlsd 각각 개별 파일)")
        if save_npy:
            print(" - NPY  :", "(node/edge/netlsd 각각 개별 파일)")

        return results




# -*- coding: utf-8 -*-
"""
Year similarity grouping via Ward clustering + Consensus
(자동 가중치 지원 + 가중치 상세 출력)

입력:
  - Node:   ./output_1111/year_NODE_EDGE_NETLSD__node_jaccard_similarity.csv
  - Edge:   ./output_1111/year_NODE_EDGE_NETLSD__edge_jaccard_similarity.csv
  - NetLSD: ./output_1111/year_NODE_EDGE_NETLSD__netlsd_similarity_exp.csv
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================================
# 유틸 함수들
# ============================================

def _coerce_square(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    df = df.sort_index().sort_index(axis=1)
    common = sorted(set(df.index) & set(df.columns))
    df = df.loc[common, common]
    df = (df + df.T) / 2.0
    np.fill_diagonal(df.values, 1.0)
    return df


def _ward_clusters_from_similarity(sim_df: pd.DataFrame, k: int) -> Dict[int, List[int]]:
    sim = np.clip(sim_df.values, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    groups = {}
    for yr, lab in zip(sim_df.index.tolist(), labels):
        groups.setdefault(lab, []).append(int(yr))
    groups = {lab: sorted(v) for lab, v in groups.items()}
    return dict(sorted(groups.items(), key=lambda kv: (kv[1][0], len(kv[1]))))


def _threshold_components(sim_df: pd.DataFrame, percentile: float = 75.0) -> List[List[int]]:
    vals = sim_df.values
    offdiag = vals[np.triu_indices_from(vals, k=1)]
    thr = float(np.percentile(offdiag, percentile))
    years = sim_df.index.tolist()
    G = nx.Graph()
    G.add_nodes_from(years)
    for i, yi in enumerate(years):
        for j in range(i + 1, len(years)):
            if sim_df.iat[i, j] >= thr:
                G.add_edge(yi, years[j])
    comps = [sorted(list(c)) for c in nx.connected_components(G)]
    return sorted(comps, key=lambda x: (x[0], len(x)))


def _coclustering_labels(sim_df: pd.DataFrame, k: int) -> np.ndarray:
    sim = np.clip(sim_df.values, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    return fcluster(Z, t=k, criterion="maxclust")


def _consensus_groups(mats: Dict[str, pd.DataFrame], k: int = 3, agree_ratio: float = 2/3) -> List[List[int]]:
    keys = list(mats.keys())
    years = mats[keys[0]].index.tolist()
    n = len(years)
    co = np.zeros((n, n), dtype=float)
    for name in keys:
        labels = _coclustering_labels(mats[name], k=k)
        co += (labels[:, None] == labels[None, :]).astype(float)
    co /= len(keys)
    G = nx.Graph()
    G.add_nodes_from(years)
    for i in range(n):
        for j in range(i + 1, n):
            if co[i, j] >= agree_ratio:
                G.add_edge(years[i], years[j], w=float(co[i, j]))
    comps = [sorted(list(c)) for c in nx.connected_components(G)]
    return sorted(comps, key=lambda x: (x[0], len(x)))


def _rows_from_clusters(method_name: str, clusters) -> List[dict]:
    rows = []
    if isinstance(clusters, dict):
        for cid, yrs in sorted(clusters.items()):
            rows.append({"method": method_name, "cluster_id": int(cid),
                         "years": ", ".join(map(str, yrs)), "size": len(yrs)})
    else:
        for i, yrs in enumerate(clusters, 1):
            rows.append({"method": method_name, "cluster_id": i,
                         "years": ", ".join(map(str, yrs)), "size": len(yrs)})
    return rows

# ============================================
# 자동 가중치 계산
# ============================================

def _auto_weights(mats: Dict[str, pd.DataFrame], mode: str) -> Dict[str, float]:
    keys = list(mats.keys())
    explanation = {}

    if mode == "variance":
        vars_ = {k: np.var(mats[k].values) for k in keys}
        inv = {k: (1.0 / v if v > 0 else 0.0) for k, v in vars_.items()}
        total = sum(inv.values()) or 1.0
        weights = {k: inv[k] / total for k in keys}
        explanation = {k: f"1/Var={1/v:.4f}" if v > 0 else "Var=0" for k, v in vars_.items()}

    elif mode == "correlation":
        flat = {k: mats[k].values.flatten() for k in keys}
        cor = {}
        for k in keys:
            others = [kk for kk in keys if kk != k]
            vals = []
            for kk in others:
                c = np.corrcoef(flat[k], flat[kk])[0, 1]
                if np.isnan(c): c = 0.0
                vals.append(max(c, 0.0))
            cor[k] = np.mean(vals) if vals else 0.0
        total = sum(cor.values()) or 1.0
        weights = {k: cor[k] / total for k in keys}
        explanation = {k: f"mean corr={cor[k]:.4f}" for k in keys}

    elif mode == "pca":
        X = np.vstack([mats[k].values.flatten() for k in keys]).T
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=1).fit(X)
        comp = np.abs(pca.components_[0])
        total = np.sum(comp) or 1.0
        weights = {k: comp[i] / total for i, k in enumerate(keys)}
        explanation = {k: f"|PC1 loading|={comp[i]:.4f}" for i, k in enumerate(keys)}

    else:
        raise ValueError("mode must be 'variance', 'correlation', or 'pca'")

    print("\n🔹 자동 가중치 계산 결과 (" + mode + " 기반)")
    for k in keys:
        print(f"  - {k:<8}: weight={weights[k]:.4f}  ({explanation[k]})")
    print()
    return weights

# ============================================
# 메인 함수
# ============================================

def group_year_trends(
    files: Dict[str, str],
    out_dir: str = "./output_1111",
    weights: Dict[str, float] = None,
    weight_mode: str = None,
    ward_ks: Tuple[int, ...] = (3, 4),
    threshold_percentile: float = 75.0,
    consensus_k: int = 3,
    consensus_agree_ratio: float = 2/3,
    save_detail_csv: bool = False,
) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    # 1️⃣ 행렬 로드
    mats = {}
    for name, path in files.items():
        df = pd.read_csv(path, index_col=0)
        mats[name] = _coerce_square(df)

    # 2️⃣ 공통 연도
    common_years = sorted(set.intersection(*(set(df.index) for df in mats.values())))
    mats = {k: v.loc[common_years, common_years] for k, v in mats.items()}

    # 3️⃣ 가중치 결정
    if weights is None:
        if weight_mode:
            weights = _auto_weights(mats, weight_mode)
        else:
            weights = {k: 1.0 for k in mats.keys()}
            print("⚖️ 균등 가중치 적용:", weights)

    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}
    print("📊 최종 정규화 가중치:", weights)

    # 4️⃣ 결합 유사도
    combined = sum(weights[k] * mats[k] for k in mats.keys())
    combined = combined.clip(lower=0, upper=1)

    # 5️⃣ Ward 군집
    rows = []
    for k in ward_ks:
        groups = _ward_clusters_from_similarity(combined, k)
        rows += _rows_from_clusters(f"Ward-k={k} (combined)", groups)

    # 6️⃣ Threshold 군집
    thr = _threshold_components(combined, threshold_percentile)
    rows += _rows_from_clusters(f"Threshold@P{int(threshold_percentile)} (combined)", thr)

    # 7️⃣ Consensus 군집
    cons = _consensus_groups(mats, k=consensus_k, agree_ratio=consensus_agree_ratio)
    rows += _rows_from_clusters(f"Consensus (k={consensus_k}, ≥{consensus_agree_ratio:.2f})", cons)

    # 8️⃣ 저장
    summary = pd.DataFrame(rows, columns=["method", "cluster_id", "years", "size"])
    out_summary = os.path.join(out_dir, "year_trend_groups_summary.csv")
    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")

    if save_detail_csv:
        combined.to_csv(os.path.join(out_dir, "combined_similarity.csv"), encoding="utf-8-sig")

    print(f"✅ 저장 완료 → {out_summary}")
    return summary





# ============================================================
# 실행 래퍼
# ============================================================

def run_similarity():
    """Figure 5: Node/Edge/NetLSD 유사도 계산 + 히트맵 + 행렬 저장."""
    config.SIMILARITY_OUTDIR.mkdir(exist_ok=True)

    analyzer = FourMethodSimilarityAnalyzer(str(config.FILTERED_BLOCKS_JSON), outdir=str(config.SIMILARITY_OUTDIR))
    analyzer.build_yearly_graphs(weight_min=1, blocks_field="keywords_cleaned_blocks")

    analyzer.plot_node_edge_and_netlsd_similarity(
        filename="year_NODE_EDGE_NETLSD_similarity.png",
        weighted_edge=False,
        binarize_netlsd=False,
        timescales=np.logspace(-2, 2, 256),
        k_eigs=None,
        force_local=False,
        kernel="exp",
        tau=None,
    )

    analyzer.save_node_edge_and_netlsd_similarity_matrices(
        prefix="year_NODE_EDGE_NETLSD",
        weighted_edge=False,
        binarize_netlsd=False,
        timescales=np.logspace(-2, 2, 256),
        k_eigs=None,
        force_local=False,
        kernel="exp",
        tau=None,
        save_csv=True,
        save_npy=False,
        save_distance_also=True,
    )
    print(f"[Figure 5] 저장 완료: {config.SIMILARITY_OUTDIR}")


def run_ward_clustering() -> pd.DataFrame:
    """Table 2: Node/Edge/NetLSD를 PCA 가중치로 결합해 Ward 계층적 군집화."""
    prefix = config.SIMILARITY_OUTDIR / "year_NODE_EDGE_NETLSD"
    files = {
        "Node": str(prefix) + "__node_jaccard_similarity.csv",
        "Edge": str(prefix) + "__edge_jaccard_similarity.csv",
        "NetLSD": str(prefix) + "__netlsd_similarity_exp.csv",
    }

    summary = group_year_trends(
        files=files,
        out_dir=str(config.SIMILARITY_OUTDIR),
        weight_mode="pca",
        ward_ks=(3, 4, 5, 6),
        threshold_percentile=75,
        consensus_k=4,
        consensus_agree_ratio=2 / 3,
        save_detail_csv=False,
    )
    print(f"[Table 2] 저장 완료: {config.WARD_SUMMARY_CSV}")

    table2 = summary[summary["method"] == "Ward-k=5 (combined)"].reset_index(drop=True)
    return table2


def run():
    run_similarity()
    table2 = run_ward_clustering()
    print(table2)
    return table2


if __name__ == "__main__":
    run()
