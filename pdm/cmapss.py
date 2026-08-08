"""NASA C-MAPSS 터보팬 데이터 로더 + 전처리.

각 행: unit, cycle, setting1-3, s1..s21 (공백 구분).

전처리 단계는 셋이다.
1) 조건별 z-정규화 — 운전 조건이 여러 개인 FD002/FD004에서 센서 절대값이
   열화와 무관하게 널뛰는 것을 제거한다.
2) 센서 선택·방향 통일 — "올라가면 나쁨"으로 부호를 맞춘다. 기본값은
   문헌 기반 고정 목록(SENSORS)이고, select_sensors()로 보정 엔진에서
   데이터 기반 선택도 가능하다.
3) 평활(smooth) — 추세 적합 전 노이즈를 줄인다. evaluate.py에 있다.
"""
import statistics
from pathlib import Path

ALL_SENSORS = tuple(f"s{i}" for i in range(1, 22))

# 문헌 기반 기본 집합: FD001에서 단조 추세를 보이는 센서 10개.
# +1 = 열화 시 상승, -1 = 열화 시 하강.
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

# 부호 없이 21개 전부 읽을 때 쓰는 선택자 (데이터 기반 선택의 입력)
UNSIGNED_ALL = {s: +1 for s in ALL_SENSORS}


def _col(sensor: str) -> int:
    """s1 → col 5 (unit, cycle, setting1-3 다음)."""
    return 4 + int(sensor[1:])


def load(path: str | Path, sensors: dict[str, int] | None = None
         ) -> dict[int, dict[str, list[float]]]:
    """유닛별 센서 시계열. engines[unit][sensor] = [값...] (부호 적용됨)."""
    sensors = SENSORS if sensors is None else sensors
    engines: dict[int, dict[str, list[float]]] = {}
    for line in Path(path).read_text().splitlines():
        cols = line.split()
        if not cols:
            continue
        eng = engines.setdefault(int(cols[0]), {s: [] for s in sensors})
        for s, sign in sensors.items():
            eng[s].append(sign * float(cols[_col(s)]))
    return engines


# ── 운전 조건(regime)별 정규화 ─────────────────────────────────
#
# FD002/FD004는 고도/마하수 등 운전 조건 6개를 오가며 운전된다. 조건이
# 바뀌면 센서 절대값이 열화와 무관하게 크게 널뛰므로, 조건별로 z-정규화해야
# 열화 추세가 드러난다.

def regime_key(s1: float, s2: float, s3: float) -> tuple:
    """설정값을 반올림해 운전 조건 키로 사용 (FD002에서 6개로 떨어짐)."""
    return (round(s1), round(s2, 2), round(s3))


def load_rows(path: str | Path, sensors: dict[str, int] | None = None
              ) -> dict[int, list[tuple[tuple, dict[str, float]]]]:
    """유닛별 (regime_key, {sensor: 원시값}) 행 목록."""
    sensors = SENSORS if sensors is None else sensors
    rows: dict[int, list[tuple[tuple, dict[str, float]]]] = {}
    for line in Path(path).read_text().splitlines():
        cols = line.split()
        if not cols:
            continue
        key = regime_key(float(cols[2]), float(cols[3]), float(cols[4]))
        values = {s: float(cols[_col(s)]) for s in sensors}
        rows.setdefault(int(cols[0]), []).append((key, values))
    return rows


def regime_stats(rows: dict[int, list], units: list[int]) -> dict:
    """보정용 유닛들의 행으로 (regime, sensor)별 평균/표준편차 계산."""
    acc: dict[tuple, dict[str, list[float]]] = {}
    for u in units:
        for key, values in rows[u]:
            bucket = acc.setdefault(key, {s: [] for s in values})
            for s, v in values.items():
                bucket[s].append(v)
    return {key: {s: (statistics.fmean(vals), statistics.pstdev(vals) or 1e-9)
                  for s, vals in bucket.items()}
            for key, bucket in acc.items()}


def _nearest_key(key: tuple, stats: dict) -> tuple:
    """보정 데이터에 없던 조건 키는 설정값이 가장 가까운 조건으로 매핑."""
    return min(stats, key=lambda k: sum((a - b) ** 2 for a, b in zip(k, key)))


def normalize(rows: dict[int, list], stats: dict,
              sensors: dict[str, int] | None = None
              ) -> dict[int, dict[str, list[float]]]:
    """조건별 z-정규화 후 부호 적용. 반환 형식은 load()와 동일."""
    sensors = SENSORS if sensors is None else sensors
    engines: dict[int, dict[str, list[float]]] = {}
    for u, unit_rows in rows.items():
        eng = engines.setdefault(u, {s: [] for s in sensors})
        for key, values in unit_rows:
            st = stats.get(key) or stats[_nearest_key(key, stats)]
            for s, sign in sensors.items():
                mu, sd = st[s]
                eng[s].append(sign * (values[s] - mu) / sd)
    return engines


# ── 데이터 기반 센서 선택 ──────────────────────────────────────

def trend_corr(y: list[float]) -> float:
    """사이클 인덱스와의 피어슨 상관. 단조 열화일수록 |r|이 1에 가깝다."""
    n = len(y)
    if n < 2:
        return 0.0
    xbar, ybar = (n - 1) / 2, statistics.fmean(y)
    sxy = sum((i - xbar) * (v - ybar) for i, v in enumerate(y))
    sxx = sum((i - xbar) ** 2 for i in range(n))
    syy = sum((v - ybar) ** 2 for v in y)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def select_sensors(engines: dict[int, dict[str, list[float]]],
                   units: list[int], min_abs_corr: float = 0.4
                   ) -> dict[str, int]:
    """보정 엔진에서 열화 추세가 뚜렷한 센서와 그 방향을 고른다.

    고정 목록은 FD001 문헌에서 온 것이라 데이터셋마다 맞지 않는다 — 실제로
    FD004에서는 목록에 없는 s9/s14가 가장 강한 신호(|r|≈0.8)인 반면 목록에
    있는 s7/s20/s21은 |r|≈0.3에 그친다. 유닛별 상관의 중앙값으로 고르면
    데이터셋에 맞는 집합이 나온다.

    반드시 **보정 엔진만** 넘길 것 — 평가 엔진이 섞이면 센서 선택 자체가
    라벨 누수가 된다.
    """
    selected = {}
    for s in next(iter(engines.values())):
        r = statistics.median(trend_corr(engines[u][s]) for u in units)
        if abs(r) >= min_abs_corr:
            selected[s] = 1 if r > 0 else -1
    return selected


def apply_signs(engines: dict[int, dict[str, list[float]]],
                sensors: dict[str, int]) -> dict[int, dict[str, list[float]]]:
    """선택된 센서만 남기고 방향을 "올라가면 나쁨"으로 통일."""
    return {u: {s: [sign * v for v in eng[s]] for s, sign in sensors.items()}
            for u, eng in engines.items()}
