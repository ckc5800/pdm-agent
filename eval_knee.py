# -*- coding: utf-8 -*-
"""오탐율이 knee 가정에 얼마나 기대고 있는지 잰다.

오탐율은 "잔여 수명 > knee 면 아직 건강하다"는 문헌 관례(piecewise-linear
RUL 타깃의 knee, 관례값 125) 위에 서 있다. C-MAPSS 에는 건강한 채 끝나는
엔진이 없어 음성 클래스를 이렇게 만들 수밖에 없는데, 그러면 숫자가 가정과
함께 움직인다 — knee 를 낮추면 "건강하다"고 보는 구간이 길어져 오탐율이
올라간다.

README 에 "절대적 오탐율이 아니라 동일 가정 아래 구성 간 비교용"이라고
적어 두기만 했지 얼마나 움직이는지는 재지 않았다. 여기서 잰다.

두 가지를 본다.
  1. knee 를 훑을 때 오탐율이 어떻게 움직이는가 (민감도)
  2. 그때 구성 간 순위가 뒤집히는가 (결론의 강건성)

2번이 핵심이다. 절대값이 흔들려도 "k=2.0 이 낫다" 같은 비교 결론이 유지되면
그 결론은 가정에 기대지 않는 것이고, 뒤집히면 결론에 가정을 명시해야 한다.

    python eval_knee.py                      # FD001, k 2.0~4.0, knee 75~175
    python eval_knee.py --fd FD002 --smooth 3
"""
import argparse
import json
import statistics
from pathlib import Path

from pdm.cmapss import (
    SENSORS, UNSIGNED_ALL, apply_signs, load, load_rows, normalize,
    regime_stats, select_sensors,
)
from pdm.evaluate import (
    BASELINE, MIN_HEALTHY_LEN, VOTES, false_alarm_rate, health_index,
    healthy_prefix_end, normalize_per_engine, smooth_engines,
)

RAW = Path("data/raw/cmapss")
OUT = Path("results-cmapss")
MULTI_REGIME = {"FD002", "FD004"}

KNEES = [75, 100, 125, 150, 175]
KS = [2.0, 2.5, 3.0, 3.5, 4.0]


def _load_engines(fd: str, select: bool, smooth: int, method: str,
                  despike: bool, norm_engine: bool, fuse: bool):
    """eval_cmapss.eval_train 과 같은 전처리 경로. 경보 대상 엔진만 돌려준다."""
    picker = UNSIGNED_ALL if select else None
    if fd in MULTI_REGIME:
        rows = load_rows(RAW / f"train_{fd}.txt", picker)
        units = sorted(rows)
        calib, evalu = units[:len(units) // 2], units[len(units) // 2:]
        engines = normalize(rows, regime_stats(rows, calib), picker)
        alarm_units = evalu
    else:
        engines = load(RAW / f"train_{fd}.txt", picker)
        units = sorted(engines)
        calib = units[:len(units) // 2]
        alarm_units = units
    if select:
        engines = apply_signs(engines, select_sensors(engines, calib))
    if norm_engine:
        engines = normalize_per_engine(engines, BASELINE)
    engines = smooth_engines(engines, smooth, method, despike)
    votes = VOTES
    if fuse:
        engines, votes = {u: health_index(e) for u, e in engines.items()}, 1
    return {u: engines[u] for u in alarm_units}, votes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--smooth-method", dest="method", default="ma")
    ap.add_argument("--despike", action="store_true")
    ap.add_argument("--normalize-engine", dest="norm_engine", action="store_true")
    ap.add_argument("--select-sensors", dest="select", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    args = ap.parse_args()

    engines, votes = _load_engines(args.fd, args.select, args.smooth,
                                   args.method, args.despike,
                                   args.norm_engine, args.fuse)
    ruls = {u: 0 for u in engines}          # train 은 run-to-failure

    rows, table = [], {}
    for knee in KNEES:
        # 건강 구간이 탐지기 최소 요건보다 짧으면 평가에서 빠진다.
        # knee 를 올릴수록 그런 엔진이 늘어 분모가 줄어든다 — 함께 보고한다.
        for k in KS:
            fa = false_alarm_rate(engines, ruls, k, votes=votes, knee=knee)
            rows.append({"knee": knee, "k": k, **fa})
            table[(knee, k)] = fa

    print(f"{args.fd} · 오탐율(%) — 행 knee, 열 k "
          f"(smooth={args.smooth}/{args.method}"
          f"{', select' if args.select else ''}{', fuse' if args.fuse else ''})")
    print("      " + "".join("%8.1f" % k for k in KS) + "   평가 엔진")
    for knee in KNEES:
        cells = []
        for k in KS:
            pct = table[(knee, k)]["false_alarm_pct"]
            cells.append("%8s" % ("-" if pct is None else pct))
        n = table[(knee, KS[0])]["evaluated"]
        print("%5d " % knee + "".join(cells) + "%9d" % n)

    # 결론의 강건성 — knee 마다 오탐 0 을 유지하는 가장 낮은 k
    print("\nknee 별 '오탐 0' 최소 k")
    lowest = {}
    for knee in KNEES:
        zero = [k for k in KS if (table[(knee, k)]["false_alarm_pct"] or 0) == 0]
        lowest[knee] = min(zero) if zero else None
        print("  knee %3d → %s" % (knee, lowest[knee] if zero else "없음(전 구간 오탐)"))
    picks = [v for v in lowest.values() if v is not None]
    if picks and len(set(picks)) == 1:
        print(f"\n판정: knee 를 {KNEES[0]}~{KNEES[-1]} 로 흔들어도 최소 k 가 "
              f"{picks[0]} 로 같음 — 운전점 선택은 이 가정에 기대지 않음.")
    else:
        print(f"\n판정: knee 에 따라 최소 k 가 달라짐{sorted(set(picks))} — "
              "운전점을 말할 때 knee 가정을 함께 명시해야 함.")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"knee-{args.fd.lower()}-s{args.smooth}{args.method}"
    path = OUT / f"{stem}.json"
    path.write_text(json.dumps(
        {"dataset": args.fd, "smooth": args.smooth, "method": args.method,
         "select": args.select, "fuse": args.fuse,
         "knees": KNEES, "ks": KS, "rows": rows,
         "lowest_k_with_zero_fa": {str(a): b for a, b in lowest.items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
