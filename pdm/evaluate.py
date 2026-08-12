"""C-MAPSS 평가 공용 로직 — 경보, RUL 추정, health index, 지표.

eval_cmapss.py(CLI)가 쓰는 결정적 알고리즘 모음. FD001~FD004 공통이며
LLM과 무관하게 동작한다. 단위 테스트는 tests/test_evaluate.py.
"""
import math
import statistics

from .detectors import detect_level_shift, linreg_slope
from .filters import (  # noqa: F401  (하위 호환 재수출)
    moving_average, normalize_per_engine, smooth_engines,
)

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


# ── 경보 품질: 탐지율·리드타임만으로는 부족하다 ──────────────────
#
# "고장 전에 울렸나"와 "얼마나 일찍 울렸나"만 재면, 기준선 직후 무조건
# 경보하는 예측기가 탐지율 100%에 최대 리드타임을 받는다. 그건 경보가
# 아니라 상수 출력이다. 남은 수명을 얼마나 낭비하는지까지 봐야 한다.

ACTIONABLE_MIN = 20    # 정비 일정을 잡는 데 필요한 최소 예고 사이클
ACTIONABLE_MAX = 100   # 이보다 이르면 멀쩡한 잔여 수명을 버리게 된다


def classify_alarm(alarm: int | None, life: int,
                   lo: int = ACTIONABLE_MIN, hi: int = ACTIONABLE_MAX) -> str:
    """경보 하나를 운용 관점에서 분류한다.

    missed     — 고장까지 한 번도 울리지 않음
    late       — 울렸으나 예고가 lo 미만이라 정비 일정을 잡을 수 없음
    actionable — 예고가 [lo, hi] 안 (쓸모 있는 경보)
    early      — 예고가 hi 초과. 고장은 맞혔지만 남은 수명을 버린다
    """
    if alarm is None:
        return "missed"
    rul = life - alarm
    if rul < lo:
        return "late"
    if rul > hi:
        return "early"
    return "actionable"


def alarm_quality(alarms: dict[int, int | None], lives: dict[int, int],
                  lo: int = ACTIONABLE_MIN, hi: int = ACTIONABLE_MAX) -> dict:
    """경보 집합을 운용 관점 4분류로 집계."""
    counts = {"actionable": 0, "early": 0, "late": 0, "missed": 0}
    for u, alarm in alarms.items():
        counts[classify_alarm(alarm, lives[u], lo, hi)] += 1
    n = len(alarms) or 1
    return {**counts, "actionable_pct": round(100 * counts["actionable"] / n, 1)}


# ── 오탐율: 없는 음성 클래스를 만들어낸다 ────────────────────────
#
# C-MAPSS는 전 엔진이 고장까지 운전되어 "건강한 채 끝나는" 엔진이 없다.
# 그래서 지금까지 오탐율을 잴 수 없었고, "수명 초반 30% 내 경보"라는
# 대리 지표로 때웠다.
#
# 그러나 각 엔진의 **앞부분**은 실제로 건강하다. 문헌이 RUL 타깃을
# piecewise-linear로 두고 knee(관례상 125) 이전을 상수 구간으로 보는 것과
# 같은 근거다. 잔여 수명이 knee를 넘는 구간만 잘라 탐지기에 넣으면,
# 거기서 울리는 경보는 정의상 오탐이다. 이렇게 음성 클래스를 구성한다.

HEALTHY_KNEE = 125    # 잔여 수명이 이보다 많으면 아직 열화 전으로 본다
MIN_HEALTHY_LEN = 45  # 탐지기 최소 요건 (기준선 30 + 지속 필터 10 + 여유)


def healthy_prefix_end(series_len: int, rul_at_end: int,
                       knee: int = HEALTHY_KNEE) -> int:
    """잔여 수명이 knee를 넘는 구간의 끝 인덱스.

    시점 i의 실제 잔여 수명은 (series_len - i + rul_at_end)이므로,
    그것이 knee를 넘는 구간은 i < series_len + rul_at_end - knee이다.
    run-to-failure(train)는 rul_at_end=0, 절단된 test는 라벨을 넣는다 —
    같은 식이 두 프로토콜에 그대로 적용된다.
    """
    return series_len + rul_at_end - knee


def false_alarm_rate(engines: dict[int, dict[str, list[float]]],
                     ruls_at_end: dict[int, int], k: float,
                     votes: int = VOTES, knee: int = HEALTHY_KNEE,
                     min_len: int = MIN_HEALTHY_LEN) -> dict:
    """건강 구간에만 탐지기를 돌려 경보가 울리는 비율 = 오탐율.

    구간이 탐지기 최소 요건보다 짧은 엔진은 판정에서 제외하고 그 수를
    함께 보고한다(분모를 조용히 바꾸지 않기 위해).
    """
    evaluated, false_alarms = 0, 0
    for u, eng in engines.items():
        length = len(next(iter(eng.values())))
        end = healthy_prefix_end(length, ruls_at_end.get(u, 0), knee)
        if end < min_len:
            continue
        evaluated += 1
        prefix = {s: v[:end] for s, v in eng.items()}
        if engine_alarm(prefix, k, votes=votes) is not None:
            false_alarms += 1
    return {
        "evaluated": evaluated,
        "false_alarms": false_alarms,
        "false_alarm_pct": (round(100 * false_alarms / evaluated, 1)
                            if evaluated else None),
        "skipped_short": len(engines) - evaluated,
        "knee": knee,
    }


def trivial_alarms(units: list[int], baseline_n: int = BASELINE
                   ) -> dict[int, int]:
    """자명한 대조군 — 기준선이 끝나자마자 전 엔진에 경보.

    탐지율 100%에 리드타임 최대를 받으므로, 탐지율·리드타임만 보는 지표가
    실제 경보 능력을 재고 있는지 판별하는 기준선이 된다.
    """
    return {u: baseline_n for u in units}


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


def survival_baseline_rul(elapsed: int, typical_life: float,
                          cap: float = 125.0) -> float:
    """센서를 전혀 보지 않고 경과 사이클만으로 RUL을 예측한다.

    RUL = (train 엔진 수명 중앙값 − 경과 사이클). 상수 대조군보다 훨씬 강한
    기준선이다 — 엔진이 얼마나 오래 돌았는지만 알아도 이 정도는 맞힌다.
    센서 기반 추정이 이것을 못 이긴다면 센서를 읽는 의미가 없다.
    """
    return max(0.0, min(typical_life - elapsed, cap))


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
