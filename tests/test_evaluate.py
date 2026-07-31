"""평가 공용 로직(pdm/evaluate.py)과 C-MAPSS 정규화 단위 테스트."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdm.cmapss import SENSORS, normalize, regime_stats  # noqa: E402
from pdm.evaluate import (  # noqa: E402
    calibrate_limits, engine_alarm, estimate_rul_exp, estimate_rul_linear,
    health_index, nasa_score,
)


def test_rul_linear_exact():
    """기울기 0.1로 상승하는 시계열 → 한계 도달까지 남은 사이클이 정확히 나온다."""
    series = [0.1 * i for i in range(100)]
    est = estimate_rul_linear({"s": series}, upto=100, limits={"s": 20.0})
    assert est is not None
    assert abs(est - (20.0 - series[-1]) / 0.1) < 1e-6


def test_rul_linear_abstains_on_flat():
    assert estimate_rul_linear({"s": [1.0] * 100}, 100, {"s": 20.0}) is None


def test_rul_exp_exact():
    """d(t) = exp(0.05·t)인 시계열 → 닫힌식 추정이 실제 남은 사이클과 일치."""
    b = 0.05
    series = [10.0 + (0.01 if i % 2 else -0.01) for i in range(30)]  # 기준선
    series += [10.0 + math.exp(b * (i - 30)) for i in range(30, 100)]
    fail_at = 110  # 한계 편차 = exp(b·(110-30)) → 마지막 관측(i=99)에서 11 사이클 남음
    limit = 10.0 + math.exp(b * (fail_at - 30))
    est = estimate_rul_exp({"s": series}, upto=100, limits={"s": limit})
    assert est is not None
    assert abs(est - (fail_at - 99)) < 0.1


def test_rul_exp_abstains_near_baseline():
    """편차가 기준선 위로 확실히 올라오지 않으면 추정을 보류한다."""
    series = [10.0 + (0.1 if i % 2 else -0.1) for i in range(100)]
    assert estimate_rul_exp({"s": series}, 100, {"s": 20.0}) is None


def test_engine_alarm_votes():
    """센서 1개만 이탈하면 2표 요건에서는 경보가 없고, 1표면 울린다."""
    shifted = [10.0 + (0.05 if i % 2 else -0.05) for i in range(40)] + [20.0] * 60
    flat = [10.0 + (0.05 if i % 2 else -0.05) for i in range(100)]
    eng = {"a": shifted, "b": flat}
    assert engine_alarm(eng, k=3.0, votes=2) is None
    assert engine_alarm(eng, k=3.0, votes=1) is not None


def test_health_index_shape_and_direction():
    """융합 지표는 기준선에서 ~0, 열화 구간에서 상승해야 한다."""
    base = [1.0 + (0.1 if i % 2 else -0.1) for i in range(30)]
    eng = {
        "a": base + [1.0 + 0.05 * i for i in range(70)],
        "b": [v * 10 for v in base] + [10.0 + 0.2 * i for i in range(70)],
    }
    hi = health_index(eng)
    assert set(hi) == {"hi"} and len(hi["hi"]) == 100
    assert abs(sum(hi["hi"][:30]) / 30) < 0.5      # 기준선 근처는 ~0
    assert hi["hi"][-1] > hi["hi"][30] + 3          # 열화로 뚜렷이 상승


def test_calibrate_limits_median():
    engines = {1: {"s": [0.0, 5.0]}, 2: {"s": [0.0, 7.0]}, 3: {"s": [0.0, 9.0]}}
    assert calibrate_limits(engines, [1, 2, 3]) == {"s": 7.0}


def test_nasa_score_asymmetry():
    """늦은 예측(+d)이 이른 예측(-d)보다 무겁게 벌점된다. 정확하면 0."""
    assert nasa_score([0.0]) == 0.0
    assert nasa_score([10.0]) > nasa_score([-10.0]) > 0


def _row(val: float) -> dict[str, float]:
    return {s: val for s in SENSORS}


def test_regime_normalize_removes_condition_jumps():
    """조건별로 상수인 신호는 정규화 후 전부 0 — 조건 전환 점프가 사라진다."""
    rows = {1: [((0,), _row(100.0)), ((1,), _row(200.0))] * 20}
    stats = regime_stats(rows, [1])
    engines = normalize(rows, stats)
    assert all(abs(v) < 1e-6 for v in engines[1]["s2"])


def test_regime_normalize_unseen_key_falls_back():
    """보정에 없던 조건 키는 가장 가까운 조건의 통계로 폴백한다 (KeyError 없음)."""
    rows = {1: [((0,), _row(100.0)), ((10,), _row(200.0))] * 20}
    stats = regime_stats(rows, [1])
    unseen = {2: [((9,), _row(200.0))] * 5}
    engines = normalize(unseen, stats)
    # (9,)는 (10,)에 가장 가까움 → 그 조건의 통계(상수 200)로 정규화 → 0
    assert all(abs(v) < 1e-6 for v in engines[2]["s2"])
