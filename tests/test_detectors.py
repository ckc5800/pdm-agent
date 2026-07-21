"""결정적 로직 단위 테스트 — LLM 없이 CI에서 돈다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdm.detectors import (  # noqa: E402
    _merge_indices, detect_level_shift, detect_spikes, detect_trend, linreg_slope,
)


def test_linreg_slope_exact():
    assert abs(linreg_slope([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-9
    assert abs(linreg_slope([5.0, 5.0, 5.0, 5.0])) < 1e-9


def test_merge_indices_gaps():
    assert _merge_indices([1, 2, 3, 10, 11], gap=5) == [(1, 3), (10, 11)]
    assert _merge_indices([], gap=5) == []


def test_spike_detected():
    values = [1.0] * 100
    values[80] = 50.0
    events = detect_spikes(values, "vibration")
    assert len(events) == 1
    assert events[0].start_idx == 80


def test_no_spike_on_flat():
    assert detect_spikes([1.0] * 100, "vibration") == []


def test_level_shift_detected():
    values = [10.0] * 300 + [30.0] * 100
    events = detect_level_shift(values, "temperature", baseline_n=240)
    assert len(events) >= 1
    assert events[0].start_idx >= 300


def test_trend_with_rul():
    values = [1.0 + 0.01 * i for i in range(800)]
    events = detect_trend(values, "vibration", window=720, limit=20.0)
    assert len(events) == 1
    assert "RUL" in events[0].evidence


def test_synthetic_eval_regression():
    """합성 시나리오 회귀 테스트 — seed 1에서 결함 3건 전부 탐지, 오탐 0."""
    from pdm.detectors import run_all
    from pdm.simulator import simulate

    data = simulate(seed=1)
    events = run_all(data)

    def overlaps(a_s, a_e, b_s, b_e, margin=30):
        return a_s <= b_e + margin and b_s - margin <= a_e

    detected = sum(
        any(overlaps(e.start_idx, e.end_idx, f.start_idx, f.end_idx)
            for e in events)
        for f in data.faults)
    assert detected == len(data.faults) == 3
