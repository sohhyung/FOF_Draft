# -*- coding: utf-8 -*-
"""파이프라인 전역 경로/파라미터 설정."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
FONTS_DIR = BASE_DIR / "fonts"
OUTPUT_DIR = BASE_DIR / "outputs"
RESOURCES_DIR = BASE_DIR / "resources"

for _d in (DATA_DIR, FONTS_DIR, OUTPUT_DIR, RESOURCES_DIR):
    _d.mkdir(exist_ok=True)

# --- 입력 데이터 ---
# soynlp WordExtractor로 후보 키워드까지 뽑아낸 원본(§3.2 전처리의 입력).
# 이 파일은 파이프라인이 생성하지 않으므로 직접 이 경로에 복사해 둬야 함.
RAW_KEYWORDS_JSON = DATA_DIR / "final_table.with_keywords.json"

# preprocessing.py의 출력 (Kiwi 정제 결과)
KIWI_CLEANED_JSON = DATA_DIR / "keywords_cleaned_blocks_kiwi3.json"

# Figure5/Table2/Table3가 쓰는, IDF로 한 번 더 거른 co-occurrence 블록 JSON.
# (table1.py가 만드는 산출물과는 별개 트랙 — 직접 준비해 두거나 동일 기준으로 생성해야 함)
FILTERED_BLOCKS_JSON = DATA_DIR / "keywords_cleaned_blocks_kiwi.filtered_blocks_idf_2.0.json"

FONT_PATH = FONTS_DIR / "NanumGothic.otf"

NORMALIZE_MAP_PATH = RESOURCES_DIR / "keyword_normalize_map.json"
SPLIT_RULES_PATH = RESOURCES_DIR / "keyword_split_rules.json"
STOPWORDS_PATH = RESOURCES_DIR / "keyword_stopwords.txt"

# 논문 §3.2 "IDF >= 2.0 기준으로 제외" 서술에 맞춘 임계값.
# Figure3 파이프라인(요약 통계)과 Figure4/Table1 파이프라인(개별 키워드 필터) 모두 이 값을 공유.
IDF_THRESHOLD = 2.0

# table1.py 산출물
KEYWORD_SIMPLE_SUM_CSV = OUTPUT_DIR / "keyword_simple_sum.csv"
TERM_FREQ_SCORE_CSV = OUTPUT_DIR / "term_freq_score_full_final222.csv"
RESULT_4_1_CSV = OUTPUT_DIR / "result_4.1.csv"
TABLE1_FINAL_CSV = OUTPUT_DIR / "증감키워드.csv"

# table2.py 산출물
SIMILARITY_OUTDIR = OUTPUT_DIR / "similarity"
WARD_SUMMARY_CSV = SIMILARITY_OUTDIR / "year_trend_groups_summary.csv"

# table3.py 산출물
COMMUNITY_OUTDIR = OUTPUT_DIR / "table3_communities"
COMMUNITY_CSV = COMMUNITY_OUTDIR / "group_1" / "clusters_full_keywords.csv"
FIGURE6_PNG = OUTPUT_DIR / "figure6_network.png"
