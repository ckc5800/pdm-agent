"""LLM 진단의 접지(grounding) 실측.

이 프로젝트는 "탐지는 결정적 알고리즘이, LLM은 해석만"을 설계 원칙으로
내세우고, 그 장치로 "탐지 근거에 있는 수치만 사용하라"는 프롬프트 제약을
쓴다. 그 제약이 실제로 지켜지는지를 여기서 잰다.

프로토콜
- seed를 바꾼 시나리오마다 탐지를 돌리고, LLM에 넘긴 근거 텍스트를 그대로
  보관한다(diagnose.build_evidence — 프롬프트 조립과 같은 함수).
- 진단문에서 숫자를 뽑아 근거 수치와 대조한다. 근거에 없는 수치가
  미접지(ungrounded)다.
- 검사기 자체는 tests/test_grounding.py로 먼저 고정했다. 한국어는 숫자에
  단위·조사가 붙어 영어식 단어 경계 가정이 깨지므로 이 순서가 중요하다.

같은 프롬프트라도 LLM 출력은 실행마다 달라진다. --repeat으로 반복해
편차와 함께 보고한다.

실행: python eval_llm.py [--seeds 1,7,42] [--repeat 3]
      (Ollama + PDM_LLM_MODEL, 기본 qwen2.5:3b 필요)
"""
import argparse
import json
import statistics
from pathlib import Path

import httpx

from pdm.detectors import run_all
from pdm.diagnose import MODEL, build_evidence, diagnose
from pdm.grounding import check_grounding, evidence_coverage
from pdm.simulator import simulate

OUT = Path("results-llm")


def run_once(seed: int) -> dict:
    data = simulate(seed=seed)
    events = run_all(data)
    evidence = build_evidence(data, events)
    text = diagnose(data, events)
    result = check_grounding(text, evidence)
    cover = evidence_coverage(text, [e.sensor for e in events])
    return {"seed": seed, "diagnosis": text, "evidence": evidence,
            **result, **cover}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1,7,42,123,2026",
                    help="시나리오 seed 목록 (쉼표 구분)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="seed마다 반복 실행 횟수 (출력 편차 확인용)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = []
    for rep in range(args.repeat):
        for seed in seeds:
            try:
                r = run_once(seed)
            except httpx.HTTPError as ex:
                raise SystemExit(f"LLM 호출 실패: {ex} — Ollama 서버를 확인하세요")
            runs.append({"repeat": rep, **r})
            flag = "OK " if r["ungrounded_count"] == 0 else "미접지"
            print(f"[{flag}] seed={seed:5} rep={rep}  숫자 {r['numbers_in_diagnosis']:2}개 중 "
                  f"미접지 {r['ungrounded_count']}개 | 탐지 센서 언급 "
                  f"{len(r['sensors_mentioned'])}/{len(r['sensors_flagged'])}"
                  + (f" → {r['ungrounded']}" if r["ungrounded"] else ""))

    total_nums = sum(r["numbers_in_diagnosis"] for r in runs)
    total_bad = sum(r["ungrounded_count"] for r in runs)
    clean_runs = sum(1 for r in runs if r["ungrounded_count"] == 0)
    per_run_bad = [r["ungrounded_count"] for r in runs]

    summary = {
        "model": MODEL,
        "runs": len(runs),
        "seeds": seeds,
        "repeat": args.repeat,
        "numbers_total": total_nums,
        "ungrounded_total": total_bad,
        "ungrounded_pct": round(100 * total_bad / total_nums, 1) if total_nums else None,
        "clean_runs": clean_runs,
        "clean_run_pct": round(100 * clean_runs / len(runs), 1),
        "ungrounded_per_run_max": max(per_run_bad),
        "ungrounded_per_run_stdev": (round(statistics.pstdev(per_run_bad), 2)
                                     if len(per_run_bad) > 1 else 0.0),
        # 숫자를 아예 안 쓰면 접지 검사를 자동 통과하므로 함께 본다
        "runs_without_numbers": sum(1 for r in runs if r["numbers_in_diagnosis"] == 0),
        "sensor_coverage_pct": round(statistics.fmean(
            r["coverage_pct"] for r in runs if r["coverage_pct"] is not None), 1),
        "details": [{k: v for k, v in r.items() if k != "evidence"} for r in runs],
    }

    OUT.mkdir(exist_ok=True)
    path = OUT / f"grounding-{MODEL.replace(':', '-')}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"\n모델 {MODEL} — 실행 {len(runs)}회")
    print(f"수치 접지: {total_nums - total_bad}/{total_nums} "
          f"(미접지 {summary['ungrounded_pct']}%)")
    print(f"미접지 0건인 실행: {clean_runs}/{len(runs)} ({summary['clean_run_pct']}%), "
          f"실행당 미접지 최대 {summary['ungrounded_per_run_max']}개 "
          f"(표준편차 {summary['ungrounded_per_run_stdev']})")
    print(f"단, 숫자를 하나도 쓰지 않은 실행이 {summary['runs_without_numbers']}/{len(runs)} — "
          f"수치 없는 진단은 접지 검사를 자동 통과한다")
    print(f"탐지 센서 언급률: {summary['sensor_coverage_pct']}% "
          f"(근거를 실제로 활용했는지의 지표)")
    print(f"\n저장: {path}")


if __name__ == "__main__":
    main()
