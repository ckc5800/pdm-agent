"""FD002 (운전 조건 6개) 검증 — 조건별 z-정규화 후 같은 탐지기 적용.

FD001과 달리 FD002는 운전 조건이 6개라 센서 절대값이 열화와 무관하게
널뛴다. 조건별 정규화 통계는 보정 엔진(전반 130대)에서만 계산하고,
경보/RUL 지표는 평가 엔진(후반 130대)에서만 잰다 — 정규화 통계와
한계치 보정 양쪽 모두 라벨 누수를 차단한다.
"""
import argparse
import json
import statistics
from pathlib import Path

from pdm.cmapss import SENSORS, load_rows, normalize, regime_stats
from eval_cmapss import (
    BASELINE, CHECKPOINTS, TREND_WIN, engine_alarm, estimate_rul,
    estimate_rul_exp,
)

DATA = Path("data/raw/cmapss/train_FD002.txt")
OUT = Path("results-cmapss")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=3.0, help="EWMA 관리한계 kσ")
    ap.add_argument("--model", choices=["linear", "exp"], default="exp")
    args = ap.parse_args()
    rul_fn = estimate_rul if args.model == "linear" else estimate_rul_exp

    rows = load_rows(DATA)
    units = sorted(rows)
    calib, evalu = units[:len(units) // 2], units[len(units) // 2:]

    stats = regime_stats(rows, calib)
    print(f"운전 조건 {len(stats)}개 식별, 엔진 {len(units)}대 "
          f"(보정 {len(calib)} / 평가 {len(evalu)})")

    engines = normalize(rows, stats)
    lives = {u: len(engines[u]["s2"]) for u in units}

    # ── 경보 (평가 엔진만) ──
    detected, lead_times, premature = 0, [], 0
    for u in evalu:
        alarm = engine_alarm(engines[u], args.k)
        if alarm is None:
            continue
        detected += 1
        lead_times.append(lives[u] - alarm)
        if alarm < lives[u] * 0.3:
            premature += 1

    # ── RUL: 한계치는 보정 엔진의 고장 시점 값 ──
    limits = {s: statistics.median(engines[u][s][-1] for u in calib)
              for s in SENSORS}
    rul_errors = {cp: [] for cp in CHECKPOINTS}
    rul_valid = {cp: 0 for cp in CHECKPOINTS}
    for u in evalu:
        for cp in CHECKPOINTS:
            upto = lives[u] - cp
            if upto < TREND_WIN + BASELINE:
                continue
            est = rul_fn(engines[u], upto, limits)
            if est is not None:
                rul_valid[cp] += 1
                rul_errors[cp].append(abs(est - cp))

    summary = {
        "dataset": "FD002",
        "k": args.k,
        "rul_model": args.model,
        "regimes": len(stats),
        "engines_eval": len(evalu),
        "life_range": [min(lives[u] for u in evalu), max(lives[u] for u in evalu)],
        "detected_before_failure": detected,
        "lead_time_median": statistics.median(lead_times) if lead_times else None,
        "premature_alarms_first30pct": premature,
        "rul": {str(cp): {
            "evaluated": rul_valid[cp],
            "mae_cycles": round(statistics.fmean(rul_errors[cp]), 1) if rul_errors[cp] else None,
            "median_abs_err": round(statistics.median(rul_errors[cp]), 1) if rul_errors[cp] else None,
        } for cp in CHECKPOINTS},
    }

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"metrics-fd002-k{args.k}-{args.model}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"고장 전 경보: {detected}/{len(evalu)}대 (평가 엔진 기준)")
    if lead_times:
        print(f"경보 리드타임: 중앙값 {summary['lead_time_median']:.0f} 사이클")
    print(f"수명 초반 30% 내 경보(조기 경보): {premature}대")
    for cp in CHECKPOINTS:
        r = summary["rul"][str(cp)]
        print(f"RUL@고장 {cp}사이클 전 ({args.model}): 평가 {r['evaluated']}대, "
              f"MAE {r['mae_cycles']} 사이클, 중앙값 오차 {r['median_abs_err']} 사이클")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
