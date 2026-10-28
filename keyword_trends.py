# -*- coding: utf-8 -*-
"""
§3.3 연도별 키워드 변화 분석 — Figure 3·4, Table 1.

입력: config.KIWI_CLEANED_JSON (preprocessing.py의 산출물)
출력:
  - config.KEYWORD_SIMPLE_SUM_CSV   (Figure 3: 연도별 고유/신규 키워드·문서 수)
  - config.TERM_FREQ_SCORE_CSV      (Figure 4 원본: 연도별 ΔTF)
  - config.RESULT_4_1_CSV           (Table 1 집계: 연도별 top10 신규/증가/감소)
  - config.TABLE1_FINAL_CSV         (Table 1 최종 포맷, 증감키워드.csv)
"""
import json
import re

import numpy as np
import pandas as pd

import config


# ============================================================
# 공통 리소스 로드 (정규화 규칙 / 분리 규칙 / 제외 키워드)
# ============================================================

def load_normalize_map() -> dict:
    return json.loads(config.NORMALIZE_MAP_PATH.read_text(encoding="utf-8"))


def load_split_patterns():
    raw = json.loads(config.SPLIT_RULES_PATH.read_text(encoding="utf-8"))
    return [(re.compile(pat), parts) for pat, parts in raw.items()]


def load_stopwords() -> set[str]:
    return set(config.STOPWORDS_PATH.read_text(encoding="utf-8").splitlines())


# ============================================================
# Figure 3 — 연도별 고유/신규 키워드 · 문서 수
# ============================================================

def flatten_keywords(blocks):
    all_kw = []
    for kw_list in blocks:
        all_kw.extend(kw_list)
    return set(all_kw)


def summarize_year(df: pd.DataFrame, target_year: int, idf_threshold: float = 0.5) -> dict:
    df_year = df[df["year"] == target_year]
    num_docs = len(df_year)

    df_subset = df[df["year"] <= target_year]
    N = len(df_subset)

    df_counts = {}
    for kws in df_subset["doc_keywords"]:
        for kw in set(kws):
            df_counts[kw] = df_counts.get(kw, 0) + 1

    idf = {kw: __import__("math").log(N / df_counts[kw]) for kw in df_counts}

    all_keywords = []
    for kws in df_year["doc_keywords"]:
        for kw in kws:
            if kw in idf and idf[kw] >= idf_threshold:
                all_keywords.append(kw)
    num_total_keywords = len(all_keywords)

    filtered_keywords = set(all_keywords)
    num_unique_keywords_after_idf_filter = len(filtered_keywords)

    past_docs = df[df["year"] < target_year]["doc_keywords"]
    past_keywords = set()
    for kws in past_docs:
        past_keywords.update(kws)
    new_keywords = filtered_keywords - past_keywords

    new_keywords_filtered = {kw for kw in new_keywords if kw in idf and idf[kw] >= idf_threshold}
    num_new_keywords = len(new_keywords_filtered)
    new_keywords_list = sorted(list(new_keywords_filtered))

    return {
        "year": target_year,
        "num_docs": num_docs,
        "num_total_keywords": num_total_keywords,
        "num_unique_keywords_after_idf_filter": num_unique_keywords_after_idf_filter,
        "num_new_keywords": num_new_keywords,
        "new_keywords_list": new_keywords_list,
    }


def summarize_all_years(df: pd.DataFrame, idf_threshold: float = 0.5) -> pd.DataFrame:
    years_sorted = sorted(df["year"].unique())
    rows = [summarize_year(df, y, idf_threshold=idf_threshold) for y in years_sorted]
    return pd.DataFrame(rows)


def run_figure3() -> pd.DataFrame:
    with open(config.KIWI_CLEANED_JSON, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df["doc_keywords"] = df["keywords_cleaned_blocks"].apply(flatten_keywords)
    df_sum = summarize_all_years(df, idf_threshold=config.IDF_THRESHOLD)

    df_sum.to_csv(config.KEYWORD_SIMPLE_SUM_CSV, encoding="utf-8-sig", index=None)
    print(f"[Figure 3] 저장 완료: {config.KEYWORD_SIMPLE_SUM_CSV}")
    return df_sum


# ============================================================
# Figure 4 / Table 1 원본 계산 — 연도별 ΔTF
# ============================================================

def _normalize_fixpoint(term: str, norm_map: dict, max_steps: int = 3) -> str:
    cur = term
    for _ in range(max_steps):
        nxt = norm_map.get(cur, cur)
        if nxt == cur:
            break
        cur = nxt
    return cur


def _split_if_needed(term: str, split_patterns, keep_original: bool = False) -> list[str]:
    for pat, parts in split_patterns:
        if pat.fullmatch(term):
            return ([term] + parts) if keep_original else list(parts)
    return [term]


def apply_norm_and_split_terms(
    df: pd.DataFrame,
    norm_map: dict,
    split_patterns,
    col: str = "keywords_cleaned_blocks",
    keep_original_in_split: bool = False,
    min_len: int = 1,
) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"DataFrame에 '{col}' 컬럼이 없습니다.")

    def _process_block(block):
        seen = set()
        out = []
        for t in (block or []):
            if not isinstance(t, str):
                t = str(t) if t is not None else ""
            t = t.strip()
            if not t:
                continue
            t_norm = _normalize_fixpoint(t, norm_map)
            parts = _split_if_needed(t_norm, split_patterns, keep_original=keep_original_in_split)
            for p in parts:
                p = p.strip()
                if len(p) < min_len:
                    continue
                if p not in seen:
                    seen.add(p)
                    out.append(p)
        return out

    new_col = []
    for blocks in df[col]:
        if isinstance(blocks, list):
            new_blocks = []
            for block in blocks:
                if isinstance(block, list):
                    cleaned = _process_block(block)
                    if cleaned:
                        new_blocks.append(cleaned)
            new_col.append(new_blocks)
        else:
            new_col.append([])
    df = df.copy()
    df[col] = new_col
    return df


def filter_functional_nouns(df: pd.DataFrame, stopwords: set[str], term_col: str = "term") -> pd.DataFrame:
    """의미 없는 기능형 명사/행정명사/수량형 단어를 제거."""
    before = len(df)
    df = df[~df[term_col].isin(stopwords)].reset_index(drop=True)
    after = len(df)
    print(f"[Figure 4] 제외된 기능형 명사: {before - after}개")
    return df


def filter_low_idf_terms(
    flat_df: pd.DataFrame,
    df: pd.DataFrame,
    threshold: float = 2.0,
    min_total_freq: int = 2,
    min_perdoc_freq: int = 2,
) -> pd.DataFrame:
    """IDF < threshold / total_freq < min / 문서당 1회만 등장 단어 제거."""
    N_docs = len(df)
    df = df.reset_index(drop=True)
    df["doc_id"] = df.index

    records = []
    for doc_id, row in df.iterrows():
        blocks = row.get("keywords_cleaned_blocks", [])
        if isinstance(blocks, list):
            for block in blocks or []:
                if isinstance(block, list):
                    for kw in block:
                        records.append({"doc_id": doc_id, "term": kw})
    doc_term_df = pd.DataFrame(records)
    if doc_term_df.empty:
        print("doc_term_df가 비었습니다. 입력 데이터 확인 필요.")
        return flat_df.copy()

    df_term = (
        doc_term_df.drop_duplicates(subset=["doc_id", "term"])
        .groupby("term")["doc_id"].nunique().reset_index(name="df_term")
    )
    total_freq = doc_term_df.groupby("term")["doc_id"].count().reset_index(name="total_freq")
    df_term = pd.merge(df_term, total_freq, on="term", how="left")
    df_term["idf"] = np.log(N_docs / df_term["df_term"])

    perdoc = doc_term_df.groupby(["doc_id", "term"]).size().reset_index(name="perdoc_freq")

    kept_terms = df_term.query("idf >= @threshold and total_freq >= @min_total_freq")["term"].unique()
    kept_pairs = perdoc[
        (perdoc["term"].isin(kept_terms)) & (perdoc["perdoc_freq"] >= min_perdoc_freq)
    ][["doc_id", "term"]]

    if "doc_id" in flat_df.columns:
        out = flat_df.merge(kept_pairs, on=["doc_id", "term"], how="inner")
    else:
        out = flat_df[flat_df["term"].isin(kept_pairs["term"].unique())].copy()

    print(f"[Figure 4] IDF>={threshold} 필터 후 고유 단어: {out['term'].nunique()}개")
    return out


def compute_tf_delta() -> pd.DataFrame:
    norm_map = load_normalize_map()
    split_patterns = load_split_patterns()
    stopwords = load_stopwords()

    with open(config.KIWI_CLEANED_JSON, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    df = apply_norm_and_split_terms(
        df, norm_map, split_patterns,
        col="keywords_cleaned_blocks", keep_original_in_split=False, min_len=1,
    )

    records = []
    for _, row in df.iterrows():
        year = int(row["year"])
        keyword_blocks = row["keywords_cleaned_blocks"]
        if isinstance(keyword_blocks, list):
            for block in keyword_blocks:
                if isinstance(block, list):
                    for kw in block:
                        records.append({"year": year, "term": kw})
    flat_df = pd.DataFrame(records)

    flat_df = filter_functional_nouns(flat_df, stopwords, term_col="term")
    flat_df = filter_low_idf_terms(flat_df, df, threshold=config.IDF_THRESHOLD)

    docs_per_year = df.groupby("year").size().to_dict()

    tf = flat_df.groupby(["year", "term"]).size().reset_index(name="count")
    tf["num_docs"] = tf["year"].map(docs_per_year)
    tf["TF_norm"] = tf["count"] / tf["num_docs"]  # 논문 식 (1) TF_doc

    tf = tf.sort_values(["term", "year"])
    tf["TF_cummean_prev"] = tf.groupby("term")["TF_norm"].transform(lambda x: x.shift().expanding().mean())
    tf["Delta_abs"] = tf.groupby("term")["TF_norm"].transform(lambda x: x - x.shift())  # 논문 식 (2) ΔTF_doc
    tf["Delta_rel"] = tf["Delta_abs"] / tf["TF_cummean_prev"]
    tf["is_new"] = tf["TF_cummean_prev"].isna()
    tf.loc[tf["is_new"], "Delta_abs"] = tf.loc[tf["is_new"], "TF_norm"]

    tf["Delta_rel"] = tf["Delta_rel"].replace([np.inf, -np.inf], np.nan).fillna(0)
    tf["Delta_plot"] = np.sign(tf["Delta_rel"]) * np.log1p(np.abs(tf["Delta_rel"]))
    tf["trend"] = np.select(
        [tf["is_new"], tf["Delta_rel"] > 0, tf["Delta_rel"] < 0],
        ["New", "Positive", "Negative"],
        default="Neutral",
    )

    tf.to_csv(config.TERM_FREQ_SCORE_CSV, encoding="utf-8-sig", index=None)
    print(f"[Figure 4] 저장 완료: {config.TERM_FREQ_SCORE_CSV}")
    return tf


# ============================================================
# Table 1 집계 — 연도별 신규/증가/감소 top10
# ============================================================

def aggregate_top10(tf: pd.DataFrame) -> pd.DataFrame:
    result_rows = []
    for year, group in tf.groupby("year"):
        new_kw = group[group["is_new"] == True].nlargest(10, "Delta_abs")[["term", "Delta_abs"]]
        inc_kw = group[group["trend"] == "Positive"].nlargest(10, "Delta_abs")[["term", "Delta_abs"]]
        dec_kw = group[group["trend"] == "Negative"].nsmallest(10, "Delta_abs")[["term", "Delta_abs"]]

        for i in range(10):
            result_rows.append({
                "year": year,
                "new_keyword": new_kw.iloc[i]["term"] if i < len(new_kw) else "",
                "new_keyword_score": new_kw.iloc[i]["Delta_abs"] if i < len(new_kw) else 0.0,
                "increase_keyword": inc_kw.iloc[i]["term"] if i < len(inc_kw) else "",
                "increase_keyword_score": inc_kw.iloc[i]["Delta_abs"] if i < len(inc_kw) else 0.0,
                "decrease_keyword": dec_kw.iloc[i]["term"] if i < len(dec_kw) else "",
                "decrease_keyword_score": dec_kw.iloc[i]["Delta_abs"] if i < len(dec_kw) else 0.0,
            })

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(config.RESULT_4_1_CSV, index=False, encoding="utf-8-sig")
    print(f"[Table 1] 저장 완료: {config.RESULT_4_1_CSV}")
    return result_df


def build_table1_final(result_df: pd.DataFrame) -> pd.DataFrame:
    def group_and_join(series):
        return ", ".join(series.dropna().astype(str).str.strip().loc[lambda x: x != ""])

    table1 = result_df.groupby("year").agg(
        new_keywords=("new_keyword", group_and_join),
        increase_keywords=("increase_keyword", group_and_join),
        decrease_keywords=("decrease_keyword", group_and_join),
    ).reset_index()

    table1.to_csv(config.TABLE1_FINAL_CSV, encoding="utf-8-sig", index=None)
    print(f"[Table 1] 저장 완료: {config.TABLE1_FINAL_CSV}")
    return table1


def run():
    run_figure3()
    tf = compute_tf_delta()
    result_df = aggregate_top10(tf)
    build_table1_final(result_df)


if __name__ == "__main__":
    run()
