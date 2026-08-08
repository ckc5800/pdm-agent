"""평활·이상치 제거 필터 단위 테스트 (pdm/filters.py).

가장 중요한 성질은 **인과성(causality)** 이다. t 시점 출력이 t 이후 관측에
의존하면 "t까지만 보고 추정한다"는 평가 전제가 깨져 모든 수치가 무의미해진다.
필터를 추가할 때마다 아래 파라미터 테스트가 자동으로 그 성질을 검사한다.
"""
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdm.filters import (  # noqa: E402
    SMOOTHERS, ewma, hampel, median_filter, moving_average,
    normalize_per_engine, savgol_endpoint, smooth, smooth_engines,
)

WINDOW = 5


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
def test_smoother_is_causal(name):
    """미래 값을 바꿔도 그 이전 시점의 출력은 변하지 않아야 한다."""
    base = [float(i) for i in range(20)]
    tail = base[:-1] + [999.0]
    a = SMOOTHERS[name](base, WINDOW)
    b = SMOOTHERS[name](tail, WINDOW)
    assert a[:-1] == pytest.approx(b[:-1]), f"{name}: 미래 값이 과거 출력에 영향"


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
def test_smoother_preserves_length_and_passthrough(name):
    values = [1.0, 5.0, 2.0, 8.0, 3.0]
    assert len(SMOOTHERS[name](values, WINDOW)) == len(values)
    assert SMOOTHERS[name](values, 1) == values      # window 1 = 무처리


@pytest.mark.parametrize("name", sorted(SMOOTHERS))
def test_smoother_reduces_noise(name):
    """평탄 신호 + 노이즈에서 출력 분산이 입력보다 작아야 한다."""
    noisy = [10.0 + (1.0 if i % 2 else -1.0) for i in range(60)]
    out = SMOOTHERS[name](noisy, WINDOW)[WINDOW:]
    assert statistics.pstdev(out) < statistics.pstdev(noisy)


def test_moving_average_values():
    assert moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]


def test_ewma_span_convention():
    """span=window → α=2/(w+1). 계단 입력이 그 비율로 수렴한다."""
    out = ewma([0.0] + [1.0] * 50, 4)     # α = 0.4
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.4)
    assert out[-1] == pytest.approx(1.0, abs=1e-6)


def test_median_filter_rejects_spike():
    """단발 스파이크는 중앙값 필터가 흡수하고, 이동평균은 끌려간다."""
    values = [1.0] * 10 + [100.0] + [1.0] * 10
    med = median_filter(values, 5)
    avg = moving_average(values, 5)
    assert med[10] == pytest.approx(1.0)
    assert avg[10] > 15.0


def test_savgol_endpoint_has_no_lag_on_ramp():
    """일정 기울기 램프에서 savgol은 실제값을 그대로, MA는 지연만큼 낮게 준다."""
    ramp = [2.0 * i for i in range(40)]
    sg = savgol_endpoint(ramp, WINDOW, degree=1)
    ma = moving_average(ramp, WINDOW)
    assert sg[-1] == pytest.approx(ramp[-1])              # 지연 없음
    # MA는 (w-1)/2 · 기울기 = 2*2 = 4 만큼 뒤처진다
    assert ma[-1] == pytest.approx(ramp[-1] - 4.0)


def test_savgol2_tracks_curvature():
    """2차 곡선에서는 degree=2가 degree=1보다 끝점을 정확히 잡는다."""
    quad = [float(i * i) for i in range(40)]
    sg1 = savgol_endpoint(quad, 7, degree=1)
    sg2 = savgol_endpoint(quad, 7, degree=2)
    assert abs(sg2[-1] - quad[-1]) < abs(sg1[-1] - quad[-1])
    assert sg2[-1] == pytest.approx(quad[-1], abs=1e-6)


def test_hampel_replaces_outlier_only():
    """이상치는 중앙값으로 대체하되, 깨끗한 구간은 건드리지 않는다."""
    clean = [1.0, 1.1, 0.9, 1.05, 0.95] * 4
    assert hampel(clean, WINDOW) == pytest.approx(clean)
    spiked = list(clean)
    spiked[12] = 50.0
    out = hampel(spiked, WINDOW)
    assert out[12] < 2.0
    assert out[:12] == pytest.approx(spiked[:12])


def test_smooth_despike_runs_hampel_first():
    """despike=True면 스파이크가 평활 전에 제거돼 결과가 덜 오염된다."""
    values = [1.0] * 10 + [100.0] + [1.0] * 10
    plain = smooth(values, WINDOW, "ma", despike=False)
    despiked = smooth(values, WINDOW, "ma", despike=True)
    assert despiked[12] < plain[12]


def test_normalize_per_engine_standardizes_baseline():
    """초기 구간이 평균 0, 표준편차 1이 되고 엔진 개체차가 사라진다."""
    a = [10.0, 12.0, 8.0, 10.0] * 10          # 평균 10
    b = [110.0, 112.0, 108.0, 110.0] * 10     # 같은 모양, 오프셋 +100
    out = normalize_per_engine({1: {"s": a}, 2: {"s": b}}, baseline_n=20)
    assert statistics.fmean(out[1]["s"][:20]) == pytest.approx(0.0, abs=1e-9)
    assert statistics.pstdev(out[1]["s"][:20]) == pytest.approx(1.0)
    assert out[1]["s"] == pytest.approx(out[2]["s"])   # 개체차 제거


def test_smooth_engines_applies_to_all_sensors():
    engines = {1: {"a": [1.0, 3.0, 5.0], "b": [2.0, 4.0, 6.0]}}
    out = smooth_engines(engines, 2, "ma")
    assert out[1]["a"] == [1.0, 2.0, 4.0]
    assert out[1]["b"] == [2.0, 3.0, 5.0]
    # window 1이면 원본 그대로 반환
    assert smooth_engines(engines, 1, "ma") is engines
