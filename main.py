# -*- coding: utf-8 -*-
"""
모태펀드 키워드 네트워크 분석 파이프라인 — 전체 실행 진입점.

단계:
  1. preprocessing      §3.2 Kiwi 정제
  2. keyword_trends     §3.3 Figure 3·4, Table 1
  3. graph_similarity   §3.4 Figure 5, Table 2
  4. community_detection §3.5 Table 3, Figure 6

사용법:
  python main.py                 # 전체 단계 순서대로 실행
  python main.py --stage table1  # 특정 단계만 실행
"""
import argparse

import preprocessing
import keyword_trends
import graph_similarity
import community_detection

STAGES = {
    "preprocessing": preprocessing.run,
    "keyword_trends": keyword_trends.run,
    "graph_similarity": graph_similarity.run,
    "community_detection": community_detection.run,
}


def main():
    parser = argparse.ArgumentParser(description="모태펀드 키워드 네트워크 분석 파이프라인")
    parser.add_argument(
        "--stage",
        choices=list(STAGES) + ["all"],
        default="all",
        help="실행할 단계 (기본값: all)",
    )
    args = parser.parse_args()

    stages = STAGES if args.stage == "all" else {args.stage: STAGES[args.stage]}
    for name, fn in stages.items():
        print(f"\n=== [{name}] 시작 ===")
        fn()
        print(f"=== [{name}] 완료 ===")


if __name__ == "__main__":
    main()
