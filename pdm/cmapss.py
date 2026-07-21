"""NASA C-MAPSS 터보팬 데이터 로더.

FD001: 엔진 100대의 run-to-failure 시계열 (단일 운전 조건).
각 행: unit, cycle, setting1-3, s1..s21 (공백 구분).

여기서는 열화 추세가 알려진 센서만 쓴다. 방향이 하강인 센서는
부호를 뒤집어 "올라가면 나쁨"으로 통일한다.
"""
from pathlib import Path

# 센서: +1 = 열화 시 상승, -1 = 열화 시 하강 (FD001에서 단조 추세를 보이는 센서들)
SENSORS = {
    "s2": +1,   # LPC 출구 온도
    "s3": +1,   # HPC 출구 온도
    "s4": +1,   # LPT 출구 온도
    "s7": -1,   # HPC 출구 압력
    "s11": +1,  # 정압 (Ps30)
    "s12": -1,  # 연료-공기비 관련
    "s15": +1,  # 바이패스 비
    "s17": +1,  # 블리드 인탈피
    "s20": -1,  # HPT 냉각재 유량
    "s21": -1,  # LPT 냉각재 유량
}


def load(path: str | Path) -> dict[int, dict[str, list[float]]]:
    """유닛별로 방향 통일된 센서 시계열을 반환. engines[unit][sensor] = [값...]"""
    engines: dict[int, dict[str, list[float]]] = {}
    for line in Path(path).read_text().splitlines():
        cols = line.split()
        if not cols:
            continue
        unit = int(cols[0])
        eng = engines.setdefault(unit, {s: [] for s in SENSORS})
        for s, sign in SENSORS.items():
            idx = 4 + int(s[1:])  # s1 → col 5
            eng[s].append(sign * float(cols[idx]))
    return engines
