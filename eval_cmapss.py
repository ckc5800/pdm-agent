"""NASA C-MAPSS 검증 CLI — FD001~FD004 통합.

시뮬레이터가 아니라 진짜 벤치마크 데이터에 기존 탐지기를 그대로 적용한다.

두 가지 프로토콜:

--split train (기본): run-to-failure인 train 셋을 반으로 갈라
  보정(한계치·정규화 통계) / 평가 엔진을 분리한다. "고장 전에 경보가
  울렸는가"와 "고장 30/50 사이클 전 시점의 RUL 오차"를 잰다.
  - 경보: 센서별 EWMA 관리한계 이탈, 서로 다른 센서 VOTES개 동시 이탈 시 인정.
    단일 조건(FD001/FD003)은 경보에 교차 엔진 보정이 없으므로 전체 엔진에서,
    다중 조건(FD002/FD004)은 정규화 통계가 보정 엔진에 의존하므로
    평가 엔진에서만 잰다.

--split test: 공식 test 셋 + RUL_FDxxx 정답 라벨 평가.
  train 전체로 보정하고, 각 test 엔진의 마지막 사이클에서 RUL을 추정해
  라벨과 비교한다 (MAE / RMSE / PHM08 score).
  주의 — 이 추정기는 열화가 안 보이면 보류하므로 응답 엔진이 부분집합이고,
  그 부분집합은 고장이 임박한 쉬운 쪽으로 치우친다. 따라서 이 MAE를
  전수 응답하는 문헌 수치와 직접 비교하면 안 된다. 대신 같은 응답
  집합에 상수(라벨 중앙값) 예측기를 돌린 constant_baseline_mae를 함께
  내보내, 점수가 추정 능력에서 온 것인지 집합 선택에서 온 것인지 가른다.
  PHM08 합계도 엔진 수에 비례하므로 phm08_per_engine으로 비교할 것.

다중 조건 셋(FD002/FD004)은 조건별 z-정규화 후 같은 파이프라인을 쓴다.
--fuse는 센서 10개를 health index 하나로 융합해 경보(1표)·RUL을 잰다.
"""
import argparse
import json
import math
import statistics
from pathlib import Path

from pdm.cmapss import load, load_rows, normalize, regime_stats
from pdm.evaluate import (
    CHECKPOINTS, MIN_SENSORS, SENSOR_MODELS, TREND_WIN, BASELINE, VOTES,
    calibrate_limits, constant_baseline_mae, engine_alarm, estimate_rul,
    health_index, nasa_score,
)

RAW = Path("data/raw/cmapss")
OUT = Path("results-cmapss")
MULTI_REGIME = {"FD002", "FD004"}


def series_len(eng: dict[str, list[float]]) -> int:
    return len(next(iter(eng.values())))


def fuse_all(engines: dict[int, dict]) -> dict[int, dict]:
    return {u: health_index(eng) for u, eng in engines.items()}


def eval_train(fd: str, k: float, model: str, fuse: bool,
               min_sensors: int = MIN_SENSORS) -> dict:
    if fd in MULTI_REGIME:
        rows = load_rows(RAW / f"train_{fd}.txt")
        units = sorted(rows)
        calib, evalu = units[:len(units) // 2], units[len(units) // 2:]
        stats = regime_stats(rows, calib)
        engines = normalize(rows, stats)
        alarm_units, regimes = evalu, len(stats)
    else:
        engines = load(RAW / f"train_{fd}.txt")
        units = sorted(engines)
        calib, evalu = units[:len(units) // 2], units[len(units) // 2:]
        alarm_units, regimes = units, None

    votes = VOTES
    if fuse:
        engines, votes = fuse_all(engines), 1
    lives = {u: series_len(engines[u]) for u in units}

    # ── 경보: 고장 전에 울렸는가 ──
    detected, lead_times, premature = 0, [], 0
    for u in alarm_units:
        alarm = engine_alarm(engines[u], k, votes=votes)
        if alarm is None:
            continue
        detected += 1
        lead_times.append(lives[u] - alarm)
        if alarm < lives[u] * 0.3:
            premature += 1

    # ── RUL: 보정/평가 엔진 분리 (라벨 누수 차단) ──
    limits = calibrate_limits(engines, calib)
    rul_errors = {cp: [] for cp in CHECKPOINTS}
    for u in evalu:
        for cp in CHECKPOINTS:
            upto = lives[u] - cp
            if upto < TREND_WIN + BASELINE:
                continue
            est = estimate_rul(engines[u], upto, limits, model, min_sensors)
            if est is not None:
                rul_errors[cp].append(abs(est - cp))

    return {
        "dataset": fd, "split": "train", "k": k, "rul_model": model,
        "fused": fuse, "regimes": regimes,
        "engines_alarm": len(alarm_units), "engines_rul": len(evalu),
        "life_range": [min(lives.values()), max(lives.values())],
        "detected_before_failure": detected,
        "lead_time_median": statistics.median(lead_times) if lead_times else None,
        "premature_alarms_first30pct": premature,
        "rul": {str(cp): {
            "evaluated": len(errs),
            "mae_cycles": round(statistics.fmean(errs), 1) if errs else None,
            "median_abs_err": round(statistics.median(errs), 1) if errs else None,
        } for cp, errs in rul_errors.items()},
    }


def eval_test(fd: str, model: str, fuse: bool,
              min_sensors: int = MIN_SENSORS) -> dict:
    """공식 test 셋 RUL 평가. 경보를 내지 않으므로 k(관리한계)는 쓰이지 않는다."""
    labels = [int(x) for x in (RAW / f"RUL_{fd}.txt").read_text().split()]
    if fd in MULTI_REGIME:
        rows_tr = load_rows(RAW / f"train_{fd}.txt")
        stats = regime_stats(rows_tr, sorted(rows_tr))
        train = normalize(rows_tr, stats)
        test = normalize(load_rows(RAW / f"test_{fd}.txt"), stats)
    else:
        train = load(RAW / f"train_{fd}.txt")
        test = load(RAW / f"test_{fd}.txt")

    if fuse:
        train, test = fuse_all(train), fuse_all(test)
    limits = calibrate_limits(train, sorted(train))

    # 라벨 파일은 유닛 번호 오름차순 한 줄에 하나. 개수가 어긋나면 이후 인덱싱이
    # 조용히 밀려 전 지표가 틀어지므로 여기서 끊는다.
    units = sorted(test)
    if len(units) != len(labels):
        raise ValueError(
            f"{fd}: test 엔진 {len(units)}대 vs RUL 라벨 {len(labels)}개 — 불일치")

    sensor_fn = SENSOR_MODELS[model]
    errors, errors_cap, answered_true, support = [], [], [], []
    for i, u in enumerate(units):
        upto = series_len(test[u])
        est = estimate_rul(test[u], upto, limits, model, min_sensors)
        if est is None:
            continue
        errors.append(est - labels[i])
        # 문헌 관례: 열화가 보이기 전 구간은 구분 불가하므로 추정치를
        # 125 사이클에 캡 (piecewise-linear RUL 타깃과 같은 발상)
        errors_cap.append(min(est, 125) - labels[i])
        answered_true.append(labels[i])
        support.append(len(sensor_fn(test[u], upto, limits)))

    n, ans = len(labels), len(errors)

    def stats(errs: list[float]) -> dict:
        if not errs:
            return {"mae_cycles": None, "rmse_cycles": None,
                    "phm08_score": None, "phm08_per_engine": None}
        return {
            "mae_cycles": round(statistics.fmean(abs(e) for e in errs), 1),
            "rmse_cycles": round(math.sqrt(statistics.fmean(e * e for e in errs)), 1),
            "phm08_score": round(nasa_score(errs), 1),
            # 합은 엔진 수에 비례하므로 집합 크기가 다른 결과와는 엔진당 값으로 비교
            "phm08_per_engine": round(nasa_score(errs) / len(errs), 2),
        }

    return {
        "dataset": fd, "split": "test", "rul_model": model,
        "fused": fuse, "engines": n,
        "true_rul_range": [min(labels), max(labels)],
        "answered": ans, "coverage_pct": round(100 * ans / n, 1),
        "answered_true_rul_median": statistics.median(answered_true) if answered_true else None,
        # 응답 집합이 쉬운 쪽으로 치우쳤는지 판별하는 대조군.
        # 같은 응답 집합에서 상수 예측기를 못 이기면 그 MAE는 추정 능력이 아니다.
        "constant_baseline_mae": (round(constant_baseline_mae(answered_true), 1)
                                  if answered_true else None),
        "constant_baseline_mae_all_engines": round(constant_baseline_mae(labels), 1),
        "sensors_backing_median": statistics.median(support) if support else None,
        "answered_on_one_sensor": sum(1 for c in support if c == 1),
        **stats(errors),
        "worst_error_cycles": round(max(errors, key=abs), 1) if errors else None,
        "cap125": stats(errors_cap),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fd", default="FD001",
                    choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--k", type=float, default=3.0,
                    help="EWMA 관리한계 kσ (경보를 내는 --split train 에서만 쓰임)")
    ap.add_argument("--model", choices=sorted(SENSOR_MODELS), default="exp",
                    help="RUL 외삽 모델")
    ap.add_argument("--fuse", action="store_true",
                    help="센서 10개를 health index 하나로 융합")
    ap.add_argument("--min-sensors", type=int, default=MIN_SENSORS,
                    dest="min_sensors",
                    help="RUL 추정에 요구할 최소 근거 센서 수 (융합 시 무의미)")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    args = ap.parse_args()

    if args.split == "train":
        s = eval_train(args.fd, args.k, args.model, args.fuse, args.min_sensors)
        print(f"{args.fd} train 프로토콜 (k={args.k}, {args.model}"
              f"{', fused' if args.fuse else ''})")
        lead = s["lead_time_median"]
        print(f"고장 전 경보: {s['detected_before_failure']}/{s['engines_alarm']}대"
              f" | 리드타임 중앙값 {f'{lead:.0f} 사이클' if lead is not None else '—(경보 없음)'}"
              f" | 수명 초반 30% 내 경보 {s['premature_alarms_first30pct']}대")
        for cp in CHECKPOINTS:
            r = s["rul"][str(cp)]
            print(f"RUL@고장 {cp}사이클 전: 평가 {r['evaluated']}/{s['engines_rul']}대, "
                  f"MAE {r['mae_cycles']} 사이클, 중앙값 오차 {r['median_abs_err']}")
    else:
        s = eval_test(args.fd, args.model, args.fuse, args.min_sensors)
        print(f"{args.fd} 공식 test 셋 ({args.model}"
              f"{', fused' if args.fuse else ''})")
        print(f"엔진 {s['engines']}대 (실제 RUL {s['true_rul_range'][0]}~"
              f"{s['true_rul_range'][1]} 사이클)")
        # 융합(--fuse)은 지표가 health index 하나뿐이라 근거 센서 수가 항상 1이다.
        support = ("융합 지표 1개" if s["fused"] else
                   f"근거 센서 중앙값 {s['sensors_backing_median']}개"
                   f" (1개뿐: {s['answered_on_one_sensor']}대)")
        print(f"응답: {s['answered']}/{s['engines']}대 ({s['coverage_pct']}%)"
              f" — 응답 엔진의 실제 RUL 중앙값 {s['answered_true_rul_median']}"
              f" | {support}")
        print(f"MAE {s['mae_cycles']} | RMSE {s['rmse_cycles']} | "
              f"최대 오차 {s['worst_error_cycles']:+} | PHM08 {s['phm08_score']} "
              f"(엔진당 {s['phm08_per_engine']})")
        c = s["cap125"]
        print(f"[추정 125캡] MAE {c['mae_cycles']} | RMSE {c['rmse_cycles']} | "
              f"PHM08 {c['phm08_score']} (엔진당 {c['phm08_per_engine']})")
        base, base_all = s["constant_baseline_mae"], s["constant_baseline_mae_all_engines"]
        verdict = "이김" if c["mae_cycles"] < base else "못 이김 ⚠"
        print(f"[대조군] 같은 응답 집합에 상수(중앙값) 예측 시 MAE {base} "
              f"→ 추정기 {c['mae_cycles']}: {verdict}"
              f"  (전체 {s['engines']}대에 상수 적용 시 MAE {base_all})")

    OUT.mkdir(exist_ok=True)
    # k는 경보 지표에만 영향을 주므로 test 결과 파일명에는 넣지 않는다
    stem = f"metrics-{s['dataset'].lower()}-{args.split}"
    if args.split == "train":
        stem += f"-k{args.k}"
    name = f"{stem}-{args.model}{'-fuse' if args.fuse else ''}.json"
    out_path = OUT / name
    out_path.write_text(json.dumps(s, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
