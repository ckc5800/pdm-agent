# -*- coding: utf-8 -*-
"""보류가 실력인지 회피인지 — 위험-커버리지 곡선.

이 RUL 추정기는 열화가 안 보이면 답을 보류한다(공식 test 셋 응답률 75~86%).
그래서 전수 응답하는 학습 모델과 MAE 를 직접 비교할 수 없다는 한계를 README
에 적어 두었다. 그런데 보류 자체가 좋은 것인지 나쁜 것인지도 아직 재지
않았다 — 어려운 엔진을 골라 피하는 것이라면 실력이고, 아무 데서나 답을
못 내는 것이라면 회피다.

선택적 예측(selective prediction)의 표준 도구가 위험-커버리지 곡선이다.
예측마다 신뢰도를 매기고, 낮은 쪽부터 버리면서 커버리지를 낮췄을 때 오차가
얼마나 내려가는지 본다. 내려가면 신뢰도가 오차를 예측하는 것이고(=보류가
실력), 평평하면 보류 기준이 정보가 없다는 뜻이다.

신뢰도는 새로 만들지 않고 이미 계산하는 값을 쓴다.
  support  뒷받침 센서 수. 조건을 통과한 센서만 살아남으므로 근거의 두께다
  spread   센서별 추정치의 사분위 범위 ÷ 중앙값. 센서끼리 얼마나 합의하는가

대조군도 같은 커버리지에서 잰다. 상수 예측기(train 수명 중앙값 기반)를 같은
부분집합에 적용해야, 커버리지를 낮춰서 얻은 이득인지 신뢰도가 만든 이득인지
갈린다. 부분집합 지표를 상수 예측기와 대조하는 것은 이 저장소에서 이미
여러 번 겪은 함정이다.

    python eval_coverage.py --fd FD001 --smooth 5
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from pdm.cmapss import (
    SENSORS, UNSIGNED_ALL, apply_signs, load, load_rows, normalize,
    regime_stats, select_sensors,
)
from pdm.evaluate import (
    BASELINE, MIN_SENSORS, SENSOR_MODELS, calibrate_limits, estimate_rul,
    health_index, normalize_per_engine, smooth_engines, survival_baseline_rul,
)

RAW = Path("data/raw/cmapss")
OUT = Path("results-cmapss")
MULTI_REGIME = {"FD002", "FD004"}
COVERAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20]


def _series_len(eng: dict[str, list[float]]) -> int:
    return len(next(iter(eng.values())))


def _spread(estimates: list[float]) -> float:
    """센서 간 불일치 = 사분위 범위 ÷ 중앙값. 작을수록 합의가 강하다."""
    if len(estimates) < 2:
        return float("inf")
    med = statistics.median(estimates)
    if med <= 0:
        return float("inf")
    q = statistics.quantiles(estimates, n=4)
    return (q[2] - q[0]) / med


def collect(fd: str, model: str, smooth: int, method: str, select: bool,
            norm_engine: bool, fuse: bool, min_sensors: int):
    """공식 test 셋에서 (예측, 라벨, 신뢰도)를 모은다. eval_test 와 같은 경로."""
    labels = [int(x) for x in (RAW / f"RUL_{fd}.txt").read_text().split()]
    picker = UNSIGNED_ALL if select else None
    if fd in MULTI_REGIME:
        rows_tr = load_rows(RAW / f"train_{fd}.txt", picker)
        stats = regime_stats(rows_tr, sorted(rows_tr))
        train = normalize(rows_tr, stats, picker)
        test = normalize(load_rows(RAW / f"test_{fd}.txt", picker), stats, picker)
    else:
        train = load(RAW / f"train_{fd}.txt", picker)
        test = load(RAW / f"test_{fd}.txt", picker)
    if select:
        sensors = select_sensors(train, sorted(train))
        train, test = apply_signs(train, sensors), apply_signs(test, sensors)
    if norm_engine:
        train = normalize_per_engine(train, BASELINE)
        test = normalize_per_engine(test, BASELINE)
    train = smooth_engines(train, smooth, method, False)
    test = smooth_engines(test, smooth, method, False)
    if fuse:
        train = {u: health_index(e) for u, e in train.items()}
        test = {u: health_index(e) for u, e in test.items()}
    limits = calibrate_limits(train, sorted(train))
    typical = statistics.median(_series_len(e) for e in train.values())

    units = sorted(test)
    if len(units) != len(labels):
        raise ValueError(f"{fd}: test {len(units)}대 vs 라벨 {len(labels)}개")

    sensor_fn = SENSOR_MODELS[model]
    out = []
    for i, u in enumerate(units):
        upto = _series_len(test[u])
        est = estimate_rul(test[u], upto, limits, model, min_sensors)
        if est is None:
            continue
        per_sensor = sensor_fn(test[u], upto, limits)
        out.append({
            "unit": u,
            "pred": min(est, 125.0),          # 문헌 관례와 같은 125 캡
            "label": labels[i],
            "support": len(per_sensor),
            "spread": _spread(per_sensor),
            "survival": survival_baseline_rul(upto, typical),
        })
    return out, labels


def _mae(rows, key="pred"):
    return statistics.fmean(abs(r[key] - r["label"]) for r in rows)


def curve(rows: list[dict], by: str) -> list[dict]:
    """신뢰도 순으로 정렬해 커버리지를 낮추며 MAE 를 잰다."""
    if by == "support":                        # 클수록 신뢰 → 내림차순
        ordered = sorted(rows, key=lambda r: (-r["support"], r["spread"]))
    else:                                      # spread 는 작을수록 신뢰
        ordered = sorted(rows, key=lambda r: (r["spread"], -r["support"]))
    total = len(ordered)
    out = []
    for cov in COVERAGES:
        n = max(2, round(total * cov / 100))
        sub = ordered[:n]
        out.append({
            "coverage_pct": cov,
            "n": n,
            "mae": round(_mae(sub), 1),
            # 같은 부분집합에서 상수 예측기(생존시간 기반)도 잰다.
            # 커버리지를 낮춰서 쉬워진 것인지 신뢰도가 고른 것인지 가른다
            "survival_mae": round(
                statistics.fmean(abs(r["survival"] - r["label"]) for r in sub), 1),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--model", default="exp", choices=sorted(SENSOR_MODELS))
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--smooth-method", dest="method", default="ma")
    ap.add_argument("--select-sensors", dest="select", action="store_true")
    ap.add_argument("--normalize-engine", dest="norm_engine", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--min-sensors", type=int, default=MIN_SENSORS)
    args = ap.parse_args()

    rows, labels = collect(args.fd, args.model, args.smooth, args.method,
                           args.select, args.norm_engine, args.fuse,
                           args.min_sensors)
    base_cov = 100 * len(rows) / len(labels)
    print(f"{args.fd} · 응답 {len(rows)}/{len(labels)}대 "
          f"(커버리지 {base_cov:.1f}%) · smooth={args.smooth}")
    print("  이 커버리지를 100%로 두고, 신뢰도 낮은 쪽부터 버리며 잰다.\n")

    result = {}
    for by in ("support", "spread"):
        c = curve(rows, by)
        result[by] = c
        print(f"  [{by} 기준]")
        print("   커버리지  n   MAE   상수 대조군   마진")
        for row in c:
            margin = row["survival_mae"] - row["mae"]
            print("   %6d%% %4d %6.1f %10.1f %8.1f"
                  % (row["coverage_pct"], row["n"], row["mae"],
                     row["survival_mae"], margin))
        first, last = c[0]["mae"], c[-1]["mae"]
        verdict = ("신뢰도가 오차를 예측함 — 보류는 실력"
                   if last < first * 0.85 else
                   "커버리지를 낮춰도 오차가 안 내려감 — 이 신뢰도는 정보가 없음")
        print(f"   판정: 100% {first} → 20% {last} · {verdict}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"coverage-{args.fd.lower()}-{args.model}-s{args.smooth}.json"
    path.write_text(json.dumps(
        {"dataset": args.fd, "model": args.model, "smooth": args.smooth,
         "answered": len(rows), "engines": len(labels),
         "base_coverage_pct": round(base_cov, 1), "curves": result},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
