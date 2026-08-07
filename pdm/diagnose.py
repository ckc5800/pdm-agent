"""LLM 진단 단계 — 탐지 결과를 해석해 정비 권고 리포트를 생성한다.

이상 '탐지'는 detectors.py의 결정적 알고리즘이 담당하고,
LLM은 탐지 근거를 종합해 사람이 읽을 진단(원인 추정·심각도·권고 조치)을
작성하는 역할만 맡는다. LLM이 수치를 만들어내지 않도록 탐지 근거를
그대로 전달하고, 근거에 없는 내용은 쓰지 말도록 프롬프트를 제한한다.
"""
import json
import os
import statistics

import httpx

from .detectors import AnomalyEvent
from .simulator import MachineData

OLLAMA_URL = os.getenv("PDM_LLM_URL", "http://localhost:11434/v1/chat/completions")
MODEL = os.getenv("PDM_LLM_MODEL", "qwen2.5:3b")

PROMPT = """당신은 회전기계 예지보전 엔지니어입니다.
아래 설비의 센서 요약과 이상 탐지 결과를 근거로 진단 리포트를 작성하세요.

규칙:
- 탐지 근거에 있는 수치만 사용하고, 새로운 수치를 만들지 마세요.
- 형식: [설비 상태 요약] / [이상 징후별 원인 추정] / [권고 조치] 세 섹션.
- 진동의 점진적 상승 + 온도 동반 상승은 베어링 마모/윤활 불량을 의심하세요.
- 한국어로 간결하게.

[설비: {machine_id}]
센서 요약(최근 1시간 평균): {summary}

이상 탐지 결과:
{events}
"""


def sensor_summary(data: MachineData) -> dict:
    last = 60
    return {
        "vibration_rms": round(statistics.fmean(data.vibration[-last:]), 2),
        "temperature": round(statistics.fmean(data.temperature[-last:]), 1),
        "current": round(statistics.fmean(data.current[-last:]), 2),
    }


def diagnose(data: MachineData, events: list[AnomalyEvent],
             timeout: float = 300.0) -> str:
    ev_lines = "\n".join(
        f"- [{e.severity}] {e.sensor} ({e.start_idx}~{e.end_idx}분, {e.detector}): {e.evidence}"
        for e in events) or "- 이상 없음"
    prompt = PROMPT.format(
        machine_id=data.machine_id,
        summary=json.dumps(sensor_summary(data), ensure_ascii=False),
        events=ev_lines,
    )
    resp = httpx.post(OLLAMA_URL, timeout=timeout, json={
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
