"""knee 민감도 스윕 회귀 테스트 — 가정을 흔드는 코드가 조용히 깨지면 안 된다.

실데이터 없이 돌려야 하므로 test_cmapss_pipeline 과 같은 방식으로 26열
C-MAPSS 포맷을 합성한다. 여기서 확인하는 것은 수치의 크기가 아니라
스윕이 성립하는 성질 두 가지다.

  1. knee 를 올리면 건강 구간이 짧아져 평가 대상 엔진이 줄어든다
     (그래서 높은 knee 의 낮은 오탐율은 표본이 줄어든 결과일 수 있다)
  2. 같은 knee 에서 k 를 올리면 오탐율은 단조 감소한다
     (관리한계를 넓히는데 오탐이 늘면 계산이 잘못된 것)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_knee  # noqa: E402
from pdm.cmapss import SENSORS  # noqa: E402

N_RAW_SENSORS = 21


def _row(unit: int, cycle: int, progress: float) -> str:
    cols = [str(unit), str(cycle), "0.0", "0.0", "100.0"]
    values = [10.0] * N_RAW_SENSORS
    for name, sign in SENSORS.items():
        values[int(name[1:]) - 1] = 10.0 + sign * 5.0 * progress
    return " ".join(cols + [f"{v:.4f}" for v in values])


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """수명 220~239 인 엔진 12대. knee 175 에서도 건강 구간이 남는 길이."""
    rows = []
    for i in range(12):
        unit, life = i + 1, 220 + i
        rows += [_row(unit, c + 1, (c / life) ** 2) for c in range(life)]
    (tmp_path / "train_FD001.txt").write_text("\n".join(rows))
    monkeypatch.setattr(eval_knee, "RAW", tmp_path)
    return tmp_path


def _sweep(knees, ks):
    engines, votes = eval_knee._load_engines(
        "FD001", select=False, smooth=1, method="ma",
        despike=False, norm_engine=False, fuse=False)
    ruls = {u: 0 for u in engines}
    out = {}
    for knee in knees:
        for k in ks:
            out[(knee, k)] = eval_knee.false_alarm_rate(
                engines, ruls, k, votes=votes, knee=knee)
    return out


def test_higher_knee_evaluates_fewer_engines(synthetic):
    """knee 를 올리면 분모가 줄거나 같아야 한다 — 늘어나면 구간 계산이 틀린 것."""
    table = _sweep([75, 125, 175], [3.0])
    counts = [table[(knee, 3.0)]["evaluated"] for knee in (75, 125, 175)]
    assert counts == sorted(counts, reverse=True)
    assert counts[0] > 0


def test_false_alarm_rate_is_monotone_in_k(synthetic):
    """같은 knee 에서 k 를 넓히면 오탐율이 늘어날 수는 없다."""
    ks = [2.0, 3.0, 4.0]
    table = _sweep([100], ks)
    pcts = [table[(100, k)]["false_alarm_pct"] or 0.0 for k in ks]
    assert pcts == sorted(pcts, reverse=True)


def test_sweep_reports_knee_with_each_row(synthetic):
    """어떤 가정에서 나온 수치인지 행마다 남아야 한다."""
    table = _sweep([75, 125], [2.5])
    assert table[(75, 2.5)]["knee"] == 75
    assert table[(125, 2.5)]["knee"] == 125
