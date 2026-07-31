"""C-MAPSS 평가 공용 로직 — 경보, RUL 추정, health index, 지표.

eval_cmapss.py(CLI)가 쓰는 결정적 알고리즘 모음. FD001~FD004 공통이며
LLM과 무관하게 동작한다. 단위 테스트는 tests/test_evaluate.py.
"""
import math
import statistics

from .detectors import detect_level_shift, linreg_slope

BASELINE = 30      # 경보/정규화 기준선 사이클 수
VOTES = 2          # 엔진 경보로 인정할 최소 센서 수 (health index는 1)
TREND_WIN = 30     # RUL 외삽에 쓰는 최근 구간
RUL_CAP = 400
CHECKPOINTS = [30, 50]   # 고장 N 사이클 전에 RUL을 물어본다


# ── 경보 ──────────────────────────────────────────────────────────

def first_alarm_cycle(series: list[float], k: float,
                      baseline_n: int = BASELINE) -> int | None:
    events = detect_level_shift(series, "cmapss", k=k, baseline_n=baseline_n)
    return events[0].start_idx if events else None


def engine_alarm(eng: dict[str, list[float]], k: float,
                 votes: int = VOTES) -> int | None:
    """서로 다른 센서 votes개가 이탈한 시점 (단일 센서 노이즈 방지)."""
    firsts = sorted(
        c for c in (first_alarm_cycle(v, k) for v in eng.values()) if c is not None)
    return firsts[votes - 1] if len(firsts) >= votes else None


# ── RUL 추정 ──────────────────────────────────────────────────────

def estimate_rul_linear(eng: dict[str, list[float]], upto: int,
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
    한계 편차에 도달하는 시점이 닫힌식으로 나온다.
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


RUL_MODELS = {"linear": estimate_rul_linear, "exp": estimate_rul_exp}


# ── Health index (센서 융합) ──────────────────────────────────────

def health_index(eng: dict[str, list[float]],
                 baseline_n: int = BASELINE) -> dict[str, list[float]]:
    """센서별 초기 구간 z-정규화 후 평균 → 단일 열화 지표.

    센서 방향은 로더에서 이미 '올라가면 나쁨'으로 통일돼 있으므로
    평균만으로 융합이 된다. 개별 센서 노이즈가 1/√N로 줄어
    지수 모델이 더 일찍(편차가 작을 때부터) 답할 수 있다.
    반환 형식은 센서 dict와 동일해 경보/RUL 코드를 그대로 쓴다.
    """
    zs = []
    for series in eng.values():
        base = series[:baseline_n]
        mu = statistics.fmean(base)
        sd = statistics.pstdev(base) or 1e-9
        zs.append([(v - mu) / sd for v in series])
    return {"hi": [statistics.fmean(col) for col in zip(*zs)]}


# ── 보정/지표 ─────────────────────────────────────────────────────

def calibrate_limits(engines: dict[int, dict[str, list[float]]],
                     units: list[int]) -> dict[str, float]:
    """보정 엔진들의 고장 시점 센서값 중앙값 → 센서별 한계치."""
    sensors = next(iter(engines.values())).keys()
    return {s: statistics.median(engines[u][s][-1] for u in units)
            for s in sensors}


def nasa_score(errors: list[float]) -> float:
    """PHM08 비대칭 점수 (낮을수록 좋음). d = 추정 - 실제.

    늦은 예측(d>0, 고장을 놓침)을 이른 예측보다 무겁게 벌점한다.
    """
    return sum(math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1
               for d in errors)
