# -*- coding: utf-8 -*-
"""지표가 아니라 비용으로 운전점을 고른다.

지금까지 운전점(k·평활창)은 유효 경보율로 골랐다. 그런데 정비 현장에서
묻는 것은 "유효율이 몇 퍼센트냐"가 아니라 "어느 쪽이 싸냐"다. 그리고 그
답은 비용비에 따라 달라진다 — 계획 외 정지가 비싼 설비와, 수명을 버리는
쪽이 비싼 설비는 최적 k 가 같을 수 없다.

경보 하나를 이미 4분류(actionable/early/late/missed)로 판정하고 있으므로,
각 분류에 비용을 매기면 그대로 기대 비용이 된다. 여기에 오탐(건강 구간에서
울린 경보)의 출동 비용을 더한다.

비용 정의 (한 대당, 상대값)
  missed     C_fail          계획 외 정지. 가장 비싸다
  late       C_fail * 0.7    울렸지만 대응 시간이 없어 대부분 정지로 이어진다
  actionable C_pm            계획 정비
  early      C_pm + C_waste  계획 정비 + 남은 수명 폐기
  오탐       C_visit         나가 봤더니 정상

기본값은 C_fail=100, C_pm=10, C_waste=15, C_visit=3 이다. 근거가 있는 값이
아니라 비율을 보기 위한 눈금이며, --fail/--pm/--waste/--visit 으로 바꾼다.
결론은 "이 비용에서 k=2.0" 이 아니라 "비용비가 이 구간이면 k 가 이렇게
움직인다"여야 한다. 그래서 기본 출력은 한 점이 아니라 스윕 표다.

    python eval_cost.py --fd FD001
    python eval_cost.py --fd FD001 --fail 30      # 정지가 덜 비싼 설비
"""
import argparse
import json
from pathlib import Path

from pdm.evaluate import (
    BASELINE, VOTES, alarm_quality, engine_alarm, false_alarm_rate,
    health_index, normalize_per_engine, smooth_engines, trivial_alarms,
)
from pdm.cmapss import (
    UNSIGNED_ALL, apply_signs, load, load_rows, normalize, regime_stats,
    select_sensors,
)

RAW = Path("data/raw/cmapss")
OUT = Path("results-cmapss")
MULTI_REGIME = {"FD002", "FD004"}

KS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
SMOOTHS = [1, 2, 3, 5]


def _series_len(eng: dict[str, list[float]]) -> int:
    return len(next(iter(eng.values())))


def _prepare(fd: str, smooth: int, method: str, select: bool,
             norm_engine: bool, fuse: bool):
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
        calib, alarm_units = units[:len(units) // 2], units
    if select:
        engines = apply_signs(engines, select_sensors(engines, calib))
    if norm_engine:
        engines = normalize_per_engine(engines, BASELINE)
    engines = smooth_engines(engines, smooth, method, False)
    votes = VOTES
    if fuse:
        engines, votes = {u: health_index(e) for u, e in engines.items()}, 1
    return {u: engines[u] for u in alarm_units}, votes


def expected_cost(quality: dict, fa: dict, cost: dict) -> dict:
    """4분류 + 오탐을 한 대당 기대 비용으로 환산."""
    n = sum(quality[c] for c in ("actionable", "early", "late", "missed")) or 1
    per_engine = (
        quality["missed"] * cost["fail"]
        + quality["late"] * cost["fail"] * 0.7
        + quality["actionable"] * cost["pm"]
        + quality["early"] * (cost["pm"] + cost["waste"])
    ) / n
    # 오탐은 건강 구간 평가에서 나온 비율이라 별도 분모를 쓴다
    fa_pct = (fa.get("false_alarm_pct") or 0.0) / 100
    visit = fa_pct * cost["visit"]
    return {"maintenance": round(per_engine, 2),
            "false_alarm": round(visit, 2),
            "total": round(per_engine + visit, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--smooth-method", dest="method", default="ma")
    ap.add_argument("--select-sensors", dest="select", action="store_true")
    ap.add_argument("--normalize-engine", dest="norm_engine", action="store_true")
    ap.add_argument("--fuse", action="store_true")
    ap.add_argument("--fail", type=float, default=100.0, help="계획 외 정지 비용")
    ap.add_argument("--pm", type=float, default=10.0, help="계획 정비 비용")
    ap.add_argument("--waste", type=float, default=15.0, help="조기 교체의 수명 폐기 비용")
    ap.add_argument("--visit", type=float, default=3.0, help="오탐 출동 비용")
    args = ap.parse_args()
    cost = {"fail": args.fail, "pm": args.pm, "waste": args.waste,
            "visit": args.visit}

    print(f"{args.fd} · 한 대당 기대 비용 (fail={args.fail} pm={args.pm} "
          f"waste={args.waste} visit={args.visit})")
    print("        " + "".join("%9s" % f"k={k}" for k in KS))
    rows, best = [], None
    for smooth in SMOOTHS:
        engines, votes = _prepare(args.fd, smooth, args.method, args.select,
                                  args.norm_engine, args.fuse)
        lives = {u: _series_len(e) for u, e in engines.items()}
        ruls = {u: 0 for u in engines}
        cells = []
        for k in KS:
            alarms = {u: engine_alarm(e, k, votes=votes)
                      for u, e in engines.items()}
            quality = alarm_quality(alarms, lives)
            fa = false_alarm_rate(engines, ruls, k, votes=votes)
            c = expected_cost(quality, fa, cost)
            rows.append({"smooth": smooth, "k": k, **quality,
                         "false_alarm_pct": fa["false_alarm_pct"], **c})
            cells.append("%9.1f" % c["total"])
            if best is None or c["total"] < best["total"]:
                best = {"smooth": smooth, "k": k, **c}
        print("  w=%-3d" % smooth + "".join(cells))

    # 자명한 대조군 — 기준선 직후 전 엔진 경보. 비용 축에서도 이겨야 한다
    engines, votes = _prepare(args.fd, 1, args.method, args.select,
                              args.norm_engine, args.fuse)
    lives = {u: _series_len(e) for u, e in engines.items()}
    trivial = alarm_quality(trivial_alarms(sorted(engines)), lives)
    trivial_cost = expected_cost(trivial, {"false_alarm_pct": 0.0}, cost)
    never = expected_cost(
        {"actionable": 0, "early": 0, "late": 0, "missed": len(engines)},
        {"false_alarm_pct": 0.0}, cost)

    print(f"\n최소 비용: w={best['smooth']} k={best['k']} → {best['total']}"
          f" (정비 {best['maintenance']} + 오탐 {best['false_alarm']})")
    print(f"대조군 — 무조건 경보 {trivial_cost['total']} · "
          f"경보 없음 {never['total']}")
    saved = never["total"] - best["total"]
    print(f"경보를 아예 안 하는 것 대비 절감 {saved:.1f} "
          f"({100 * saved / never['total']:.0f}%)")

    OUT.mkdir(parents=True, exist_ok=True)
    # 비용 조합이 파일명에 다 들어가야 한다 — fail 만 넣었다가 waste 만
    # 바꾼 실행이 앞 결과를 덮어썼다
    stem = (f"cost-{args.fd.lower()}-f{int(args.fail)}-p{int(args.pm)}"
            f"-w{int(args.waste)}-v{int(args.visit)}")
    path = OUT / f"{stem}.json"
    path.write_text(json.dumps(
        {"dataset": args.fd, "cost": cost, "rows": rows, "best": best,
         "trivial": trivial_cost, "never_alarm": never},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
