"""신호처리/통계 기반 이상 탐지기.

LLM 없이 결정적으로 동작한다. 각 탐지기는 (지표, 시점, 근거)를 담은
AnomalyEvent를 반환하고, LLM Agent는 이 결과를 해석·진단하는 역할만 맡는다.

- rolling z-score  : 급격한 스파이크 (순간 충격)
- EWMA 관리한계    : 평균 수준의 지속적 이탈 (과열 등)
- 추세 기울기      : 점진적 열화 (베어링 마모) + 간이 RUL 추정
"""
import statistics
from dataclasses import dataclass

from .simulator import MachineData


@dataclass
class AnomalyEvent:
    detector: str      # zscore | ewma | trend
    sensor: str        # vibration | temperature | current
    start_idx: int
    end_idx: int
    severity: str      # info | warning | critical
    evidence: str      # 수치 근거 (사람/LLM이 읽는 문장)


def _merge_indices(indices: list[int], gap: int = 5) -> list[tuple[int, int]]:
    """연속(또는 gap 이내) 인덱스를 구간으로 병합."""
    if not indices:
        return []
    spans, start, prev = [], indices[0], indices[0]
    for i in indices[1:]:
        if i - prev > gap:
            spans.append((start, prev))
            start = i
        prev = i
    spans.append((start, prev))
    return spans


def detect_spikes(values: list[float], sensor: str,
                  window: int = 60, z_th: float = 5.0) -> list[AnomalyEvent]:
    """rolling z-score로 순간 스파이크 탐지."""
    hits = []
    for i in range(window, len(values)):
        ref = values[i - window:i]
        mu = statistics.fmean(ref)
        sd = statistics.pstdev(ref) or 1e-9
        z = (values[i] - mu) / sd
        if z > z_th:
            hits.append((i, z))
    events = []
    for s, e in _merge_indices([i for i, _ in hits]):
        zmax = max(z for i, z in hits if s <= i <= e)
        events.append(AnomalyEvent(
            "zscore", sensor, s, e, "warning",
            f"z-score 최대 {zmax:.1f} (임계 {z_th}) — 순간 스파이크"))
    return events


def detect_level_shift(values: list[float], sensor: str,
                       alpha: float = 0.05, k: float = 4.0,
                       baseline_n: int = 240,
                       unit: str = "분") -> list[AnomalyEvent]:
    """EWMA가 초기 구간 기준 관리한계를 벗어나는 지속적 수준 이탈 탐지.

    unit은 근거 문장의 시간 단위 표기. 시뮬레이터는 1분 간격이라 기본값이
    "분"이고, C-MAPSS처럼 샘플 단위가 다른 데이터는 호출부에서 넘긴다.
    """
    base = values[:baseline_n]
    mu = statistics.fmean(base)
    sd = statistics.pstdev(base) or 1e-9
    upper = mu + k * sd
    ewma = mu
    hits = []
    for i, v in enumerate(values):
        ewma = alpha * v + (1 - alpha) * ewma
        if i >= baseline_n and ewma > upper:
            hits.append(i)
    events = []
    for s, e in _merge_indices(hits, gap=30):
        if e - s < 10:      # 너무 짧은 이탈은 무시
            continue
        events.append(AnomalyEvent(
            "ewma", sensor, s, e,
            "critical" if e - s > 120 else "warning",
            f"EWMA가 기준선({mu:.1f}) + {k}σ 관리한계({upper:.1f})를 "
            f"{e - s}{unit}간 초과 — 지속적 수준 이탈"))
    return events


def linreg_slope(y: list[float]) -> float:
    """최소제곱 선형회귀 기울기 (샘플 1개당 증가량)."""
    n = len(y)
    xbar = (n - 1) / 2
    ybar = statistics.fmean(y)
    sxy = sum((i - xbar) * (v - ybar) for i, v in enumerate(y))
    sxx = sum((i - xbar) ** 2 for i in range(n))
    return sxy / sxx


def detect_trend(values: list[float], sensor: str,
                 window: int = 12 * 60, slope_th: float = 0.0008,
                 limit: float | None = None) -> list[AnomalyEvent]:
    """최근 구간 선형회귀 기울기로 점진적 열화 탐지 + 간이 RUL 추정."""
    if len(values) < window:
        return []
    y = values[-window:]
    slope = linreg_slope(y)  # 분당 증가량
    if slope < slope_th:
        return []
    evidence = f"최근 {window // 60}시간 기울기 +{slope * 60:.3f}/시간 — 점진적 상승 추세"
    if limit is not None and slope > 0:
        remain_min = (limit - values[-1]) / slope
        if remain_min > 0:
            evidence += f", 한계치({limit}) 도달까지 약 {remain_min / 60:.0f}시간 (간이 RUL)"
    return [AnomalyEvent(
        "trend", sensor, len(values) - window, len(values) - 1,
        "critical" if limit and values[-1] > limit * 0.8 else "warning",
        evidence)]


def run_all(data: MachineData) -> list[AnomalyEvent]:
    events = []
    events += detect_spikes(data.vibration, "vibration")
    events += detect_level_shift(data.temperature, "temperature")
    events += detect_level_shift(data.vibration, "vibration")
    events += detect_trend(data.vibration, "vibration", limit=7.1)  # ISO 10816 Zone D 근사
    events += detect_trend(data.temperature, "temperature")
    return sorted(events, key=lambda e: e.start_idx)

