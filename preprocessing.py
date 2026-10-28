# -*- coding: utf-8 -*-
"""
§3.2 데이터 수집 및 키워드 추출 — Kiwi 형태소 분석기 기반 정제 단계.

입력: RAW_KEYWORDS_JSON
  - soynlp WordExtractor로 이미 후보 키워드(block_keywords)까지 뽑아낸 원본.
  - 이 파일 자체는 이 모듈이 만들지 않으므로, config.RAW_KEYWORDS_JSON 위치에 직접 준비해 둬야 함.

출력: config.KIWI_CLEANED_JSON (및 동일 이름의 .csv)
  - 조사/어미를 제거하고 명사 중심 복합어로 정리한 keywords_cleaned_blocks.
  - 이 산출물을 keyword_trends.py가 그대로 불러와 Figure3/Figure4/Table1을 만든다.
"""
import json
import re

import pandas as pd
from kiwipiepy import Kiwi

import config

kiwi = Kiwi()  # 필요시 Kiwi(num_workers=2) 등으로 병렬화 가능

DIGIT_RE = re.compile(r"\d")

# 남길 품사(내용어 중심): 일반/고유/의존명사, 수사, 어근, 외래어/한자, 관형사/접두사
KEEP_POS = {"NNG", "NNP", "NNB", "NR", "XR", "SL", "SH", "MM", "XPN"}
# 버릴 품사(조사/어미/기호/숫자 등)
DROP_PREFIXES = ("J", "E")  # J*: 조사, E*: 어미
DROP_EXACT = {
    "SN",                        # 숫자
    "SP", "SF", "SS", "SE", "SO",  # 구두점/기호
    "VCP", "VCN",                # 긍정/부정 서술격 조사(형태)
}
# 접사류: 명사 파생 접미사(XSN)·동사/형용사 파생 접미사(XSV/XSA)·보조 용언(VX)은 기본적으로 제외
DROP_DERIV = {"XSN", "XSV", "XSA", "VX"}


def _is_drop(pos: str) -> bool:
    if pos in DROP_EXACT or pos in DROP_DERIV:
        return True
    return pos.startswith(DROP_PREFIXES)


def kiwi_clean_eojeol(eojeol: str, min_len: int = 2) -> list[str]:
    """
    조사(J*)/어미(E*) 이전까지의 핵심 덩어리를 한 토큰으로 보존.
    - 접두적 관형사/접두사(MM/XPN) + 명사군(N*) + XR/SL/SH 까지는 붙여서 유지
    - J*/E*을 만나면 거기서 컷(조사/어미 제거)
    - 동사/형용사 등(다른 품사)을 만나면 과감히 중단(명사 복합어만 유지)
    - 숫자 섞인 것은 전체 무시
    """
    if not isinstance(eojeol, str) or DIGIT_RE.search(eojeol):
        return []

    eojeol = eojeol.replace("·", "").replace("-", "")

    morphs = kiwi.tokenize(eojeol)

    keep_first_chunk = []
    for tk in morphs:
        mor, pos = tk.form, tk.tag

        if pos.startswith("J") or pos.startswith("E"):
            break

        if pos in {"MM", "XPN", "XR", "SL", "SH"} or pos.startswith("N"):
            keep_first_chunk.append(mor)
            continue

        if pos in {"SP", "SF", "SS", "SE", "SO"}:
            break

        break

    token = "".join(keep_first_chunk)
    return [token] if len(token) >= min_len else []


def clean_block_keywords_by_block(block_keywords):
    cleaned_blocks = []
    for kw_list in block_keywords or []:
        block_clean = []
        seen = set()
        for tok in (kw_list or []):
            tokens = kiwi_clean_eojeol(str(tok))
            for ct in tokens:
                if ct not in seen:
                    seen.add(ct)
                    block_clean.append(ct)
        if block_clean:
            cleaned_blocks.append(block_clean)
    return cleaned_blocks


def records_to_dataframe(doc_obj) -> pd.DataFrame:
    records = doc_obj if isinstance(doc_obj, list) else [doc_obj]
    rows = []
    for r in records:
        rows.append({
            "year": r.get("year"),
            "month": r.get("month"),
            "type": r.get("type"),
            "rest": r.get("rest"),
            "keywords_cleaned_blocks": clean_block_keywords_by_block(r.get("block_keywords") or []),
        })
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    with open(config.RAW_KEYWORDS_JSON, "r", encoding="utf-8-sig") as f:
        doc = json.load(f)

    df = records_to_dataframe(doc)
    print(df.head(1).to_dict(orient="records")[0])

    df.to_json(config.KIWI_CLEANED_JSON, indent=2, force_ascii=False)
    df.to_csv(config.KIWI_CLEANED_JSON.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"저장 완료: {config.KIWI_CLEANED_JSON}")
    return df


if __name__ == "__main__":
    run()
