"""프롬프트·근거 조립과 리포트 생성 테스트 (LLM 호출 없음).

가장 중요한 것은 **프롬프트와 근거 텍스트의 정합성**이다. 접지 검사
(eval_llm.py)는 "모델이 실제로 본 수치"와 대조한다는 전제 위에 있는데,
두 문자열이 따로 조립되면 한쪽만 수정됐을 때 그 전제가 조용히 깨진다.
그러면 멀쩡한 인용이 hallucination으로, 혹은 그 반대로 집계된다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdm.detectors import run_all  # noqa: E402
from pdm.diagnose import build_evidence, build_prompt, sensor_summary  # noqa: E402
from pdm.grounding import extract_numbers  # noqa: E402
from pdm.report import chart_svg, save_report  # noqa: E402
from pdm.simulator import simulate  # noqa: E402


def _case():
    data = simulate(seed=42)
    return data, run_all(data)


def test_prompt_numbers_all_appear_in_evidence():
    """프롬프트에 등장하는 수치는 전부 근거 텍스트에도 있어야 한다.

    접지 검사의 전제 그 자체다. 근거를 프롬프트와 따로 조립하면 여기서
    깨지고, 모델이 프롬프트 문구를 인용한 것이 hallucination으로 집계된다
    (실제로 "최근 1시간 평균"의 1이 근거에서 빠져 있었다).
    """
    data, events = _case()
    prompt_nums = set(extract_numbers(build_prompt(data, events)))
    evidence_nums = set(extract_numbers(build_evidence(data, events)))
    missing = prompt_nums - evidence_nums
    assert not missing, f"프롬프트에만 있는 수치: {sorted(missing)}"


def test_evidence_contains_summary_and_events():
    data, events = _case()
    ev = build_evidence(data, events)
    assert ev == build_prompt(data, events)   # 근거 = 모델이 본 것 전부
    assert data.machine_id in ev
    assert "센서 요약" in ev and "이상 탐지 결과" in ev
    for e in events:
        assert e.sensor in ev


def test_evidence_handles_no_events():
    """이상이 없어도 근거 텍스트는 만들어져야 한다 (진단 호출이 죽지 않도록)."""
    data = simulate(seed=1, inject=False)
    ev = build_evidence(data, [])
    assert "이상 없음" in ev


def test_sensor_summary_keys_and_rounding():
    data, _ = _case()
    s = sensor_summary(data)
    assert set(s) == {"vibration_rms", "temperature", "current"}
    assert all(isinstance(v, float) for v in s.values())


def test_chart_svg_is_wellformed_and_marks_events():
    """SVG가 닫히고, 탐지 구간마다 음영 사각형이 그려진다."""
    data, events = _case()
    svg = chart_svg(data, events)
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<rect") >= 2 + sum(
        1 for e in events if e.sensor in ("vibration", "temperature"))


def test_save_report_writes_both_files(tmp_path):
    data, events = _case()
    path = save_report(data, events, "진단 본문", out_dir=str(tmp_path))
    assert path.exists() and (tmp_path / "chart.svg").exists()
    text = path.read_text(encoding="utf-8")
    assert "진단 본문" in text
    assert all(e.detector in text for e in events)
