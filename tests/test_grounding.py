"""접지 검사기 자체를 먼저 고정한다.

채점기에 결함이 있으면 그 결함이 모델의 hallucination으로 잘못 귀속된다.
특히 한국어는 숫자에 단위·조사가 곧바로 붙어(22분간, 3.5σ를, 45.2도) 영어식
단어 경계 가정이 통하지 않는다. 아래 케이스로 추출·판정 규칙을 못박는다.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdm.grounding import (  # noqa: E402
    check_grounding, extract_numbers, is_grounded, rounding_tolerance,
)


def test_extract_numbers_with_korean_units_attached():
    """숫자에 단위·조사가 붙어 있어도 숫자만 정확히 떼어낸다."""
    text = "EWMA가 기준선(45.2) + 3.0σ 관리한계(52.8)를 22분간 초과했습니다."
    assert extract_numbers(text) == ["45.2", "3.0", "52.8", "22"]


def test_extract_handles_decimals_and_integers():
    assert extract_numbers("z-score 최대 7.3 (임계 5)") == ["7.3", "5"]
    assert extract_numbers("이상 없음") == []


def test_extract_skips_numbers_inside_identifiers():
    """설비 ID·센서명의 숫자는 측정값이 아니다.

    실측에서 이 케이스를 빠뜨려 'PUMP-01'의 01이 지어낸 수치로 집계되는
    바람에 미접지율이 100%로 나왔다. 검사기 결함이 모델 탓으로 귀속된
    전형적인 사례라 회귀 테스트로 못박는다.
    """
    assert extract_numbers("설비 PUMP-01은 정상입니다.") == []
    assert extract_numbers("센서 s21이 이탈") == []
    # 식별자 옆에 진짜 측정값이 있으면 그건 잡아야 한다
    assert extract_numbers("PUMP-01의 진동은 4.21 mm/s") == ["4.21"]


def test_grounded_when_value_present_in_evidence():
    ev = [45.2, 52.8, 22.0]
    assert is_grounded("45.2", ev)
    assert is_grounded("22", ev)


def test_ungrounded_when_value_absent():
    """근거에 없는 수치는 잡아내야 한다 — 이게 검사기의 존재 이유다."""
    assert not is_grounded("87.5", [45.2, 52.8, 22.0])


def test_rounding_tolerance_follows_written_precision():
    """허용오차는 표기 자릿수가 함의하는 반올림 폭이어야 한다.

    상대 허용오차(5% 등)를 쓰면 값이 큰 시간 인덱스에서 폭이 100을 넘어,
    근거 2580을 2680으로 바꿔 쓴 것을 통과시킨다 — 실측에서 실제로 놓쳤다.
    """
    assert is_grounded("22.4", [22.35])        # 소수 1자리 → ±0.05
    assert not is_grounded("28.0", [22.35])
    assert rounding_tolerance("22.4") == 0.05
    assert rounding_tolerance("2580") == 0.5   # 정수는 사실상 정확 일치
    assert is_grounded("2580", [2580.0])
    assert not is_grounded("2680", [2580.0])   # 100 차이는 지어낸 것


def test_ordinal_and_percent_boilerplate_not_extracted():
    """목록 번호("1.", "2번째")나 관용적 0%/100%는 추출 단계에서 걸러진다."""
    assert extract_numbers("1. 설비 상태 요약") == []
    assert extract_numbers("이상 징후 3번째는 다음과 같습니다") == []
    assert extract_numbers("미접지 0%를 달성했습니다") == []
    assert extract_numbers("정확도 100%입니다") == []


def test_real_small_values_are_still_caught_as_ungrounded():
    """근거에 없는 실측값이 우연히 1~3/100 범위여도 잡아야 한다.

    문맥 없이 값만 보고 면제하던 이전 버전은 이걸 놓쳤다(발견 당시엔
    회귀 테스트가 오히려 그 맹점을 고정하고 있었다).
    """
    assert extract_numbers("온도가 3도 상승했습니다") == ["3"]
    assert not is_grounded("3", [])
    assert not is_grounded("100", [45.2])
    assert extract_numbers("변화량은 1.5입니다") == ["1.5"]  # 소수는 서수 오판 없음


def test_check_grounding_reports_only_real_violations():
    evidence = "vibration 4.21, temperature 67.3, EWMA 22분간 초과"
    clean = "진동 4.21 mm/s와 온도 67.3도가 22분간 상승했습니다."
    assert check_grounding(clean, evidence)["ungrounded"] == []

    dirty = "진동이 4.21에서 9.87로 올랐고 베어링 수명은 340시간 남았습니다."
    result = check_grounding(dirty, evidence)
    assert set(result["ungrounded"]) == {"9.87", "340"}
    assert result["ungrounded_count"] == 2


def test_grounded_pct_none_when_no_numbers():
    r = check_grounding("이상 징후가 없습니다.", "vibration 1.0")
    assert r["numbers_in_diagnosis"] == 0
    assert r["grounded_pct"] is None
