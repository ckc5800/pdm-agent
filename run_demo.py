"""E2E 데모: 시뮬레이션 → 이상 탐지 → LLM 진단 → 리포트 저장."""
import sys

from pdm.detectors import run_all
from pdm.diagnose import diagnose
from pdm.report import save_report
from pdm.simulator import simulate


def main():
    print("1) 48시간 센서 데이터 시뮬레이션 (결함 3종 주입)...")
    data = simulate()

    print("2) 이상 탐지 (z-score / EWMA / 추세)...")
    events = run_all(data)
    for e in events:
        print(f"   [{e.severity}] {e.sensor} {e.start_idx}~{e.end_idx}분 — {e.evidence[:60]}")

    print("3) LLM 진단 리포트 생성 (Ollama)...")
    try:
        text = diagnose(data, events)
    except Exception as ex:
        sys.exit(f"   LLM 호출 실패: {ex} — Ollama 서버를 확인하세요")

    path = save_report(data, events, text)
    print(f"\n저장: {path} (+ chart.svg)")
    print("\n===== 진단 =====")
    print(text)


if __name__ == "__main__":
    main()
