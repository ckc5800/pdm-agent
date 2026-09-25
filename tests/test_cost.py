"""비용 모델 회귀 테스트.

비용 함수가 조용히 틀리면 운전점 선택이 통째로 틀린다. 여기서 보는 것은
절대값이 아니라 비용 모델이 지켜야 할 성질이다.

  1. 분류별 비용 순서 — missed > late > early > actionable
  2. 오탐 비용이 분리 집계되는가 (정비 비용과 섞이면 원인을 못 가린다)
  3. 전부 놓친 경우 = "경보 없음" 상한과 같은가
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_cost import expected_cost  # noqa: E402

COST = {"fail": 100.0, "pm": 10.0, "waste": 15.0, "visit": 3.0}
NO_FA = {"false_alarm_pct": 0.0}


def _only(kind: str, n: int = 10) -> dict:
    q = {"actionable": 0, "early": 0, "late": 0, "missed": 0}
    q[kind] = n
    return q


def test_cost_order_matches_operational_severity():
    """놓침이 가장 비싸고, 늦은 경보가 그다음, 조기 교체가 유효 경보보다 비싸다."""
    costs = {k: expected_cost(_only(k), NO_FA, COST)["total"]
             for k in ("missed", "late", "early", "actionable")}
    assert costs["missed"] > costs["late"] > costs["early"] > costs["actionable"]


def test_all_missed_equals_never_alarming():
    """전부 놓치면 경보를 아예 안 한 것과 같아야 한다 — 상한의 정합성."""
    assert expected_cost(_only("missed"), NO_FA, COST)["total"] == COST["fail"]


def test_false_alarm_cost_is_reported_separately():
    """오탐 비용을 정비 비용에 섞지 않는다. 섞이면 어느 쪽이 비싼지 못 가린다."""
    q = _only("actionable", 10)
    clean = expected_cost(q, NO_FA, COST)
    noisy = expected_cost(q, {"false_alarm_pct": 50.0}, COST)
    assert clean["maintenance"] == noisy["maintenance"]
    assert noisy["false_alarm"] == 1.5           # 0.5 × 3
    assert noisy["total"] == clean["total"] + 1.5


def test_early_alarm_costs_pm_plus_waste():
    """조기 경보는 계획 정비를 하고도 남은 수명을 버린다."""
    early = expected_cost(_only("early"), NO_FA, COST)["total"]
    assert early == COST["pm"] + COST["waste"]


def test_empty_quality_does_not_divide_by_zero():
    """엔진이 하나도 없는 입력에서도 죽지 않아야 한다."""
    empty = {"actionable": 0, "early": 0, "late": 0, "missed": 0}
    assert expected_cost(empty, NO_FA, COST)["total"] == 0.0
