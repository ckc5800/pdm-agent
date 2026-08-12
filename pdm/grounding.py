"""LLM 진단의 접지(grounding) 검사 — 근거에 없는 수치를 지어냈는지 판정.

이 프로젝트의 설계 원칙은 "탐지는 알고리즘이, LLM은 해석만"이고, 그 핵심
장치가 "탐지 근거에 있는 수치만 사용하라"는 프롬프트 제약이다. 그런데 그
제약이 실제로 지켜지는지는 한 번도 측정된 적이 없었다. 여기서 잰다.

**검사기를 먼저 고정한다.** 채점기 자체가 틀리면 그 결함이 모델 탓으로
귀속된다. 한국어 텍스트는 숫자에 단위·조사가 바로 붙어("22분간", "3.5σ를")
영어식 단어 경계 가정이 깨지므로, 추출 규칙을 합성 케이스로 먼저 검증한다.
"""
import re

# 숫자: 부호 없는 정수/소수. 앞뒤에 단위·조사가 붙어도 숫자만 떼어낸다.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# 식별자 안의 숫자는 측정값이 아니다 — "PUMP-01"의 01, 센서명 "s21"의 21.
# 실측에서 이걸 빠뜨려 전 실행이 미접지로 잘못 집계된 적이 있다(설비 ID).
# 숫자 바로 앞이 영문자이거나 "영문자-"이면 식별자의 일부로 본다.
IDENT_PREFIX_RE = re.compile(r"[A-Za-z]-?$")

# 접지 판정에서 면제할 수치 —
# 서수/목록 번호("1.", "첫 번째")나 백분율 0/100 같은 관용 표현까지 잡으면
# 거짓 양성만 늘어난다. 근거의 실제 측정값과 무관한 상용구다.
EXEMPT = {"0", "1", "2", "3", "100"}


def rounding_tolerance(token: str) -> float:
    """표기된 자릿수가 함의하는 반올림 폭 (마지막 유효숫자의 1/2).

    "22.4"로 썼다면 근거값이 22.35~22.45였다는 뜻이므로 허용오차는 0.05다.
    정수 "2580"이면 0.5 — 즉 사실상 정확히 일치해야 한다.

    상대 허용오차(예: 5%)를 쓰면 안 된다. 시간 인덱스처럼 값이 큰 수치에서
    5%는 100 이상이 되어, 근거의 2580을 2680으로 바꿔 쓴 것을 통과시킨다
    (실측에서 실제로 놓쳤다).
    """
    if "." in token:
        return 0.5 * 10 ** -len(token.split(".")[1])
    return 0.5


def extract_numbers(text: str) -> list[str]:
    """텍스트에서 측정값으로 보이는 숫자 토큰을 원문 표기 그대로 뽑는다.

    식별자에 붙은 숫자(PUMP-01, s21)는 제외한다 — 설비 이름을 그대로 옮겨
    쓴 것을 '지어낸 수치'로 세면 미접지율이 100%로 나온다(실제로 그랬다).
    """
    return [m.group() for m in NUMBER_RE.finditer(text)
            if not IDENT_PREFIX_RE.search(text[:m.start()])]


def _as_float(tok: str) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return None


def is_grounded(token: str, evidence_numbers: list[float]) -> bool:
    """숫자 하나가 근거 수치 중 하나와 (반올림 폭 안에서) 일치하는가."""
    if token in EXEMPT:
        return True
    value = _as_float(token)
    if value is None:
        return True
    tol = rounding_tolerance(token)
    return any(abs(value - e) <= tol for e in evidence_numbers)


# 센서명 → 진단문에서 찾을 한국어 표현
SENSOR_TERMS = {
    "vibration": ("진동",),
    "temperature": ("온도", "발열", "과열"),
    "current": ("전류",),
}


def evidence_coverage(diagnosis: str, event_sensors: list[str]) -> dict:
    """탐지된 센서를 진단문이 실제로 언급했는가.

    접지율만 보면 **숫자를 하나도 쓰지 않은 진단이 만점**을 받는다. 실제로
    qwen2.5:3b는 10회 중 9회를 숫자 없이 서술해 미접지 0%가 나왔다.
    "지어내지 않았는가"와 "근거를 쓰기는 했는가"는 별개이므로 같이 잰다.
    """
    flagged = sorted(set(event_sensors))
    mentioned = [s for s in flagged
                 if any(t in diagnosis for t in SENSOR_TERMS.get(s, ()))]
    return {
        "sensors_flagged": flagged,
        "sensors_mentioned": mentioned,
        "coverage_pct": (round(100 * len(mentioned) / len(flagged), 1)
                         if flagged else None),
    }


def check_grounding(diagnosis: str, evidence_text: str) -> dict:
    """진단문의 수치가 근거에 존재하는지 검사.

    evidence_text에는 LLM에 전달한 모든 근거(센서 요약 + 탐지 이벤트)를
    그대로 넣는다. 반환값의 ungrounded가 비어 있어야 원칙이 지켜진 것이다.
    """
    evidence_numbers = [v for v in map(_as_float, extract_numbers(evidence_text))
                        if v is not None]
    tokens = extract_numbers(diagnosis)
    ungrounded = [t for t in tokens if not is_grounded(t, evidence_numbers)]
    return {
        "numbers_in_diagnosis": len(tokens),
        "ungrounded": ungrounded,
        "ungrounded_count": len(ungrounded),
        "grounded_pct": (round(100 * (len(tokens) - len(ungrounded)) / len(tokens), 1)
                         if tokens else None),
    }
