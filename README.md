# kfof_pipeline

JKIIE 논문 "모태펀드 키워드 네트워크 기반 한국 벤처투자 환경의 구조적 패턴 분석"의
전체 분석 파이프라인을 모듈화한 코드입니다. 논문의 §3.2~§3.5 구성을 그대로 따릅니다.

## 폴더 구조

```
kfof_pipeline/
├── main.py                   전체 파이프라인 실행 진입점
├── config.py                 경로·파라미터 중앙 관리 (IDF 임계값 등)
├── preprocessing.py          §3.2 데이터 수집 및 키워드 추출 (Kiwi 정제)
├── keyword_trends.py         §3.3 연도별 키워드 변화 분석 (Figure 3·4, Table 1)
├── graph_similarity.py       §3.4 연도별 그래프 구조 유사도 분석 (Figure 5, Table 2)
├── community_detection.py    §3.5 커뮤니티 탐지 기반 주제 구조 분석 (Table 3, Figure 6)
│
├── resources/                 하드코딩 대신 분리해 둔 데이터
│   ├── keyword_normalize_map.json   용어 정규화 규칙 (콘텐→콘텐츠 등)
│   ├── keyword_split_rules.json     합성어 분리 규칙
│   └── keyword_stopwords.txt        분석에서 제외할 기능형 명사
│
├── data/      입력 JSON을 직접 복사해 둘 폴더
├── fonts/     NanumGothic.otf 등 시각화용 폰트
└── outputs/   각 단계 산출 CSV/PNG
```

## 실행 전 준비

1. `data/final_table.with_keywords.json` — soynlp WordExtractor로 후보 키워드까지 뽑아낸 원본 (이 파이프라인은 생성하지 않음, 직접 복사)
2. `data/keywords_cleaned_blocks_kiwi.filtered_blocks_idf_2.0.json` — Figure5/Table2/Table3가 쓰는 IDF 필터링된 co-occurrence 블록 JSON (`preprocessing.py` 산출물과는 별개 트랙)
3. `fonts/NanumGothic.otf` — Figure 5/6 시각화용 폰트
4. 필요 패키지: `kiwipiepy`, `networkx`, `python-louvain`, `igraph`, `leidenalg`, `scipy`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

## 실행

```bash
# 전체 파이프라인
python main.py

# 특정 단계만
python main.py --stage preprocessing
python main.py --stage keyword_trends
python main.py --stage graph_similarity
python main.py --stage community_detection
```
