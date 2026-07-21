"""NASA C-MAPSS FD001 실데이터로 탐지기 검증.

시뮬레이터가 아니라 진짜 벤치마크 데이터에 기존 탐지기를 그대로 적용한다.
엔진 100대가 전부 고장까지 운전된 데이터라, "고장 전에 경보가 울렸는가"와
"경보가 얼마나 일찍(또는 너무 일찍) 울렸는가"를 라벨 없이도 측정할 수 있다.

프로토콜
- 경보: 센서별 EWMA 관리한계 이탈(detect_level_shift 그대로, baseline 30 사이클).
  서로 다른 센서 2개가 이탈해야 엔진 경보로 인정 (단일 센서 노이즈 방지).
- RUL: 엔진 1~50의 고장 시점 센서값 중앙값으로 한계치를 보정하고,
  엔진 51~100에서 고장 30/50 사이클 전 시점에 선형 외삽 RUL을 평가.
  (보정과 평가를 분리해 라벨 누수를 막는다)
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from pdm.cmapss import SENSORS, load
from pdm.detectors import detect_level_shift, linreg_slope

DATA = Path("data/raw/cmapss/train_FD001.txt")
OUT = Path("results-cmapss")
BASELINE = 30      # 경보 기준선 사이클 수
VOTES = 2          # 엔진 경보로 인정할 최소 센서 수
TREND_WIN = 30     # RUL 외삽에 쓰는 최근 구간
RUL_CAP = 400
CHECKPOINTS = [30, 50]   # 고장 N 사이클 전에 RUL을 물어본다


def first_alarm_cycle(series: list[float], k: float) -> int | None:
    events = detect_level_shift(series, "cmapss", k=k, baseline_n=BASELINE)
    return events[0].start_idx if events else None


def engine_alarm(eng: dict[str, list[float]], k: float) -> int | None:
    firsts = sorted(
        c for c in (first_alarm_cycle(v, k) for v in eng.values()) if c is not None)
    return firsts[VOTES - 1] if len(firsts) >= VOTES else None


def estimate_rul(eng: dict[str, list[float]], upto: int,
                 limits: dict[str, float]) -> float | None:
    """upto 사이클까지만 보고 선형 외삽으로 남은 사이클 추정."""
    estimates = []
    for s, series in eng.items():
        y = series[max(0, upto - TREND_WIN):upto]
        if len(y) < TREND_WIN:
            continue
        slope = linreg_slope(y)
        if slope <= 0:
            continue
        remain = (limits[s] - series[upto - 1]) / slope
        if 0 < remain <= RUL_CAP:
            estimates.append(remain)
    return statistics.median(estimates) if estimates else None


def estimate_rul_exp(eng: dict[str, list[float]], upto: int,
                     limits: dict[str, float]) -> float | None:
    """지수 열화 모델: 기준선 대비 편차 d(t)가 지수적으로 커진다고 가정.

    C-MAPSS의 열화는 뒤로 갈수록 가속하는 곡선이라 선형 외삽이 남은 수명을
    과대평가한다. ln(d)에 선형회귀를 하면 d(t) = A·exp(b·t) 피팅이 되고,
    한계 편차에 도달하는 시점을 닫힌식으로 구할 수 있다.
    """
    estimates = []
    for s, series in eng.items():
        base = series[:BASELINE]
        mu_b = statistics.fmean(base)
        y = series[max(0, upto - TREND_WIN):upto]
        if len(y) < TREND_WIN:
            continue
        d = [v - mu_b for v in y]
        # 아직 기준선 근처면(음수 섞임) 지수 모델을 적용할 수 없다
        if min(d) <= 0:
            continue
        ln_d = [math.log(v) for v in d]
        b = linreg_slope(ln_d)
        if b <= 0:
            continue
        limit_d = limits[s] - mu_b
        if limit_d <= d[-1]:
            estimates.append(0.0)
            continue
        remain = (math.log(limit_d) - math.log(d[-1])) / b
        if 0 < remain <= RUL_CAP:
            estimates.append(remain)
    return statistics.median(estimates) if estimates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=float, default=4.0, help="EWMA 관리한계 kσ")
    ap.add_argument("--model", choices=["linear", "exp"], default="linear",
                    help="RUL 외삽 모델")
    args = ap.parse_args()
    rul_fn = estimate_rul if args.model == "linear" else estimate_rul_exp

    engines = load(DATA)
    units = sorted(engines)
    lives = {u: len(engines[u]["s2"]) for u in units}

    # ── 1. 경보: 고장 전에 울렸는가 ──
    detected, lead_times, premature = 0, [], 0
    for u in units:
        alarm = engine_alarm(engines[u], args.k)
        if alarm is None:
            continue
        detected += 1
        lead_times.append(lives[u] - alarm)
        if alarm < lives[u] * 0.3:
            premature += 1

    # ── 2. RUL: 보정(1~50) / 평가(51~100) 분리 ──
    calib, evalu = units[:50], units[50:]
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
        "k": args.k,
        "rul_model": args.model,
        "engines": len(units),
        "life_range": [min(lives.values()), max(lives.values())],
        "detected_before_failure": detected,
        "lead_time_median": statistics.median(lead_times),
        "lead_time_min": min(lead_times),
        "premature_alarms_first30pct": premature,
        "rul": {str(cp): {
            "evaluated": rul_valid[cp],
            "mae_cycles": round(statistics.fmean(rul_errors[cp]), 1) if rul_errors[cp] else None,
            "median_abs_err": round(statistics.median(rul_errors[cp]), 1) if rul_errors[cp] else None,
        } for cp in CHECKPOINTS},
    }

    OUT.mkdir(exist_ok=True)
    suffix = f"-{args.model}" if args.model != "linear" else ""
    out_path = OUT / f"metrics-k{args.k}{suffix}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"엔진 {summary['engines']}대 (수명 {summary['life_range'][0]}~{summary['life_range'][1]} 사이클)")
    print(f"고장 전 경보: {detected}/{len(units)}대")
    print(f"경보 리드타임: 중앙값 {summary['lead_time_median']:.0f} 사이클, 최소 {summary['lead_time_min']} 사이클")
    print(f"수명 초반 30% 내 경보(조기 경보): {premature}대")
    for cp in CHECKPOINTS:
        r = summary["rul"][str(cp)]
        print(f"RUL@고장 {cp}사이클 전: 평가 {r['evaluated']}대, "
              f"MAE {r['mae_cycles']} 사이클, 중앙값 오차 {r['median_abs_err']} 사이클")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
