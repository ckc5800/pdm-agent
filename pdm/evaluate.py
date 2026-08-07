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
MIN_SENSORS = 1    # RUL 추정에 필요한 최소 뒷받침 센서 수
CHECKPOINTS = [30, 50]   # 고장 N 사이클 전에 RUL을 물어본다


# ── 경보 ──────────────────────────────────────────────────────────

def first_alarm_cycle(series: list[float], k: float,
                      baseline_n: int = BASELINE) -> int | None:
    events = detect_level_shift(series, "cmapss", k=k, baseline_n=baseline_n,
                                unit="사이클")
    return events[0].start_idx if events else None


def engine_alarm(eng: dict[str, list[float]], k: float,
                 votes: int = VOTES) -> int | None:
    """서로 다른 센서 votes개가 이탈한 시점 (단일 센서 노이즈 방지)."""
    firsts = sorted(
        c for c in (first_alarm_cycle(v, k) for v in eng.values()) if c is not None)
    return firsts[votes - 1] if len(firsts) >= votes else None


# ── RUL 추정 ──────────────────────────────────────────────────────

def _window_ok(upto: int) -> bool:
    """기준선 구간과 회귀 구간이 겹치지 않아야 추정을 시도한다.

    upto < BASELINE + TREND_WIN이면 series[:BASELINE]과 series[upto-TREND_WIN:upto]가
    겹쳐 기준선 평균이 회귀 구간 자신에 끌려간다. 그러면 편차 d가 인위적으로
    0 근처로 눌려 아무리 깨끗한 열화도 min(d) <= 0에 걸려 보류된다 —
    근거가 없어서가 아니라 창 산술 때문에 답을 못 하는 상태가 된다.
    이 경우 아예 시도하지 않아 '보류' 사유를 증거 기반으로만 남긴다.
    """
    return upto >= BASELINE + TREND_WIN


def _sensor_rul_linear(eng: dict[str, list[float]], upto: int,
                       limits: dict[str, float]) -> list[float]:
    """센서별 선형 외삽 추정치 목록 (추세가 상승인 센서만)."""
    if not _window_ok(upto):
        return []
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
    return estimates


def _sensor_rul_exp(eng: dict[str, list[float]], upto: int,
                    limits: dict[str, float]) -> list[float]:
    """센서별 지수 열화 모델 추정치 목록.

    C-MAPSS의 열화는 뒤로 갈수록 가속하는 곡선이라 선형 외삽이 남은 수명을
    과대평가한다. ln(d)에 선형회귀를 하면 d(t) = A·exp(b·t) 피팅이 되고,
    한계 편차에 도달하는 시점이 닫힌식으로 나온다.
    """
    if not _window_ok(upto):
        return []
    estimates = []
    for s, series in eng.items():
        mu_b = statistics.fmean(series[:BASELINE])
        y = series[max(0, upto - TREND_WIN):upto]
        if len(y) < TREND_WIN:
            continue
        d = [v - mu_b for v in y]
        # 아직 기준선 근처면(음수 섞임) 지수 모델을 적용할 수 없다
        if min(d) <= 0:
            continue
        b = linreg_slope([math.log(v) for v in d])
        if b <= 0:
            continue
        limit_d = limits[s] - mu_b
        if limit_d <= d[-1]:
            estimates.append(0.0)
            continue
        remain = (math.log(limit_d) - math.log(d[-1])) / b
        if 0 < remain <= RUL_CAP:
            estimates.append(remain)
    return estimates


SENSOR_MODELS = {"linear": _sensor_rul_linear, "exp": _sensor_rul_exp}


def estimate_rul(eng: dict[str, list[float]], upto: int,
                 limits: dict[str, float], model: str = "exp",
                 min_sensors: int = MIN_SENSORS) -> float | None:
    """센서별 추정치의 중앙값. 뒷받침 센서가 min_sensors 미만이면 보류.

    센서는 "상승 추세이고 한계치까지 남았다"는 조건을 통과한 것만 살아남으므로,
    살아남은 센서 수 자체가 근거의 두께다. 경보는 VOTES=2로 단일 센서를
    배제하는데 RUL에 같은 요건이 없으면 한 센서만 보고 답하는 일이 생긴다.
    """
    estimates = SENSOR_MODELS[model](eng, upto, limits)
    if len(estimates) < min_sensors:
        return None
    return statistics.median(estimates)


def estimate_rul_linear(eng: dict[str, list[float]], upto: int,
                        limits: dict[str, float]) -> float | None:
    return estimate_rul(eng, upto, limits, model="linear")


def estimate_rul_exp(eng: dict[str, list[float]], upto: int,
                     limits: dict[str, float]) -> float | None:
    return estimate_rul(eng, upto, limits, model="exp")


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
    합이므로 엔진 수에 비례한다 — 집합 크기가 다르면 엔진당 값으로 비교할 것.
    """
    return sum(math.exp(-d / 13) - 1 if d < 0 else math.exp(d / 10) - 1
               for d in errors)


def constant_baseline_mae(labels: list[float]) -> float | None:
    """같은 엔진 집합에 '항상 그 집합의 라벨 중앙값'을 답하는 예측기의 MAE.

    추정기가 응답한 엔진만 모아 점수를 내면 그 부분집합이 쉬운 쪽으로
    치우칠 수 있다. 동일 부분집합에서 상수 예측기와 비교해야 그 점수가
    추정 능력에서 나온 것인지, 집합 선택에서 나온 것인지 구분된다.
    """
    if not labels:
        return None
    c = statistics.median(labels)
    return statistics.fmean(abs(v - c) for v in labels)
