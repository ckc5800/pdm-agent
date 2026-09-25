"""위험-커버리지 곡선의 성질 테스트.

곡선 자체가 결론을 만드는 도구라, 정렬이나 부분집합 계산이 틀리면 "보류는
실력"이라는 판정이 통째로 거짓이 된다. 수치가 아니라 성질을 고정한다.

  1. 신뢰도 순서대로 잘라야 한다 — support 는 큰 쪽, spread 는 작은 쪽부터
  2. 커버리지 100%는 전체 집합과 같아야 한다 (조용히 일부를 빠뜨리지 않기)
  3. 대조군을 같은 부분집합에서 재야 한다 (커버리지 효과와 신뢰도 효과 분리)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_coverage  # noqa: E402
from eval_coverage import _spread, curve  # noqa: E402


def _row(unit, pred, label, support, spread, survival=60.0):
    return {"unit": unit, "pred": pred, "label": label, "support": support,
            "spread": spread, "survival": survival}


def test_spread_is_scale_free_and_infinite_for_single_sensor():
    """센서가 하나뿐이면 합의를 말할 수 없다 — 신뢰도 최하로 둔다."""
    assert _spread([50.0]) == float("inf")
    tight = _spread([48.0, 50.0, 52.0, 51.0])
    loose = _spread([10.0, 50.0, 90.0, 51.0])
    assert tight < loose


def test_support_curve_drops_low_support_first(monkeypatch):
    """support 가 낮은(=근거가 얇은) 예측부터 버려야 한다."""
    monkeypatch.setattr(eval_coverage, "COVERAGES", [100, 50])
    rows = [_row(1, 50, 50, support=6, spread=0.1),     # 정확·근거 두꺼움
            _row(2, 50, 50, support=6, spread=0.1),
            _row(3, 90, 50, support=2, spread=0.9),     # 부정확·근거 얇음
            _row(4, 90, 50, support=2, spread=0.9)]
    c = curve(rows, "support")
    assert c[0]["coverage_pct"] == 100 and c[0]["n"] == 4
    assert c[0]["mae"] == 20.0                          # (0+0+40+40)/4
    assert c[1]["n"] == 2 and c[1]["mae"] == 0.0        # 두꺼운 쪽만 남음


def test_spread_curve_keeps_agreeing_sensors(monkeypatch):
    """spread 기준은 불일치가 큰 예측부터 버린다."""
    monkeypatch.setattr(eval_coverage, "COVERAGES", [100, 50])
    rows = [_row(1, 50, 50, support=3, spread=0.05),
            _row(2, 50, 50, support=3, spread=0.06),
            _row(3, 10, 50, support=3, spread=0.80),
            _row(4, 10, 50, support=3, spread=0.90)]
    c = curve(rows, "spread")
    assert c[1]["n"] == 2 and c[1]["mae"] == 0.0


def test_baseline_is_measured_on_the_same_subset(monkeypatch):
    """대조군도 같은 부분집합에서 재야 커버리지 효과와 신뢰도 효과가 갈린다."""
    monkeypatch.setattr(eval_coverage, "COVERAGES", [100, 50])
    rows = [_row(1, 50, 50, 6, 0.1, survival=55.0),     # 대조군 오차 5
            _row(2, 50, 50, 6, 0.1, survival=55.0),
            _row(3, 90, 50, 2, 0.9, survival=90.0),     # 대조군 오차 40
            _row(4, 90, 50, 2, 0.9, survival=90.0)]
    c = curve(rows, "support")
    assert c[0]["survival_mae"] == 22.5                 # (5+5+40+40)/4
    assert c[1]["survival_mae"] == 5.0                  # 남은 두 대만
