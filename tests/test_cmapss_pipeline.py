"""C-MAPSS 파이프라인 통합 테스트 — 합성 데이터로 로더·평가 전 구간을 검증한다.

실데이터(data/raw)는 저장소에 없으므로 CI에서 eval_train/eval_test가
전혀 검증되지 않았다. C-MAPSS와 동일한 26열 포맷을 합성해 로더의 열
인덱싱, 라벨 정렬, 보정/평가 분리까지 한 번에 확인한다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval_cmapss  # noqa: E402
from pdm.cmapss import SENSORS, load  # noqa: E402

N_SETTINGS = 3
N_RAW_SENSORS = 21


def _row(unit: int, cycle: int, progress: float, regime: int = 0) -> str:
    """C-MAPSS 한 행: unit cycle setting1-3 s1..s21.

    추적 대상 센서에만 열화를 싣는다. 부호가 -1인 센서는 원본에서
    하강해야 로더가 뒤집었을 때 상승이 되므로 방향을 반대로 준다.
    """
    cols = [str(unit), str(cycle), f"{regime}.0", "0.0", "100.0"]
    values = [10.0] * N_RAW_SENSORS
    for name, sign in SENSORS.items():
        values[int(name[1:]) - 1] = 10.0 + sign * 5.0 * progress
    return " ".join(cols + [f"{v:.4f}" for v in values])


def _engine_rows(unit: int, life: int, start: int = 0) -> list[str]:
    """수명 life까지 단조 열화하는 엔진. start부터 잘라 test 엔진으로도 쓴다."""
    return [_row(unit, c + 1, (c / life) ** 2) for c in range(start, life)]


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """train 20대(수명 100~119) + test 20대(임의 시점 절단) + RUL 라벨."""
    train, test, labels = [], [], []
    for i in range(20):
        unit, life = i + 1, 100 + i
        train += _engine_rows(unit, life)
        cut = life - (10 + i)          # 고장 10~29 사이클 전에서 절단
        test += [_row(unit, c + 1, (c / life) ** 2) for c in range(cut)]
        labels.append(life - cut)

    (tmp_path / "train_FD001.txt").write_text("\n".join(train))
    (tmp_path / "test_FD001.txt").write_text("\n".join(test))
    (tmp_path / "RUL_FD001.txt").write_text("\n".join(str(v) for v in labels))
    monkeypatch.setattr(eval_cmapss, "RAW", tmp_path)
    return tmp_path, labels


def test_loader_column_indexing_and_sign(synthetic):
    """s2(+1)는 상승, s7(-1)은 뒤집혀 상승으로 통일돼야 한다."""
    path, _ = synthetic
    engines = load(path / "train_FD001.txt")
    assert len(engines) == 20
    for s in ("s2", "s7"):
        series = engines[1][s]
        assert series[-1] > series[0], f"{s}: 방향 통일 실패"


def test_eval_train_detects_and_estimates(synthetic):
    s = eval_cmapss.eval_train("FD001", k=3.0, model="exp", fuse=False)
    assert s["engines_alarm"] == 20
    assert s["detected_before_failure"] > 0
    assert s["premature_alarms_first30pct"] == 0
    assert s["rul"]["30"]["evaluated"] > 0


def test_eval_train_fused_runs(synthetic):
    s = eval_cmapss.eval_train("FD001", k=3.0, model="exp", fuse=True)
    assert s["fused"] is True
    assert s["detected_before_failure"] > 0


def test_eval_test_aligns_labels(synthetic):
    """단조 열화 합성 데이터에서는 추정이 실제 RUL을 크게 벗어나면 안 된다.

    라벨이 한 칸이라도 밀리면 엔진마다 라벨이 10~29로 다르게 배정돼
    오차가 부풀므로, 정렬이 깨지면 이 임계를 넘는다.
    """
    s = eval_cmapss.eval_test("FD001", model="linear", fuse=False)
    assert s["engines"] == 20
    assert s["answered"] > 0
    assert s["mae_cycles"] < 20


def test_eval_test_reports_constant_baseline(synthetic):
    """응답 집합이 쉬운 쪽으로 치우쳤는지 볼 대조군이 항상 함께 나와야 한다."""
    s = eval_cmapss.eval_test("FD001", model="linear", fuse=False)
    assert s["constant_baseline_mae"] is not None
    assert s["constant_baseline_mae_all_engines"] is not None
    assert s["phm08_per_engine"] is not None


def test_main_survives_no_alarm(synthetic, monkeypatch, capsys):
    """경보가 0건이면 리드타임 중앙값이 None — 출력에서 죽지 않아야 한다."""
    monkeypatch.setattr(sys, "argv",
                        ["eval_cmapss.py", "--fd", "FD001", "--k", "50"])
    monkeypatch.setattr(eval_cmapss, "OUT", synthetic[0] / "out")
    eval_cmapss.main()
    out = capsys.readouterr().out
    assert "고장 전 경보: 0/20대" in out
    assert "경보 없음" in out


def test_eval_test_rejects_label_count_mismatch(synthetic):
    path, labels = synthetic
    (path / "RUL_FD001.txt").write_text("\n".join(str(v) for v in labels[:-1]))
    with pytest.raises(ValueError, match="불일치"):
        eval_cmapss.eval_test("FD001", model="linear", fuse=False)
