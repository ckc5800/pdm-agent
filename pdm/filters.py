"""시계열 평활·이상치 제거 필터 — 전부 후행(causal).

t 시점의 출력은 t까지의 관측만 사용한다. 중심 필터를 쓰면 미래를 참조하게
되어 "t 시점까지만 보고 추정한다"는 평가 전제가 깨진다. 이 모듈의 모든
함수는 그 성질을 지키며, 단위 테스트로 고정되어 있다.

필터마다 노이즈 억제와 지연(lag)의 균형이 다르다.

| 필터 | 노이즈 억제 | 추세 지연 | 스파이크 | 비고 |
|---|---|---|---|---|
| ma       | 1/√w    | (w-1)/2·기울기 | 약함 | 가장 단순 |
| ewma     | 중간     | 약 (1-α)/α     | 약함 | 오래된 값도 조금 반영 |
| median   | 중간     | (w-1)/2·기울기 | 강함 | 스파이크에 강건 |
| savgol   | 약함     | ~0             | 약함 | 국소 직선의 끝점 — 지연 보정 |
| savgol2  | 더 약함   | ~0             | 약함 | 국소 2차 — 곡률까지 추종 |

이동평균은 상승 추세에서 현재값을 (w-1)/2·기울기만큼 과소평가한다.
C-MAPSS처럼 단조 열화하는 신호에서는 이 편향이 편차 d를 눌러 RUL 추정에
직접 영향을 준다 — savgol은 그 편향을 제거하는 대신 분산을 키운다.
"""
import statistics

DEFAULT_HAMPEL_SIGMA = 3.0
MAD_TO_SIGMA = 1.4826   # 정규분포에서 MAD → 표준편차 환산 상수


def moving_average(values: list[float], window: int) -> list[float]:
    """후행 이동평균. i번째 값은 i까지의 최근 window개 평균."""
    if window <= 1:
        return list(values)
    out, run = [], 0.0
    for i, v in enumerate(values):
        run += v
        if i >= window:
            run -= values[i - window]
        out.append(run / min(i + 1, window))
    return out


def ewma(values: list[float], window: int) -> list[float]:
    """지수가중 이동평균. window를 span으로 보아 α = 2/(window+1).

    같은 window 값으로 다른 필터와 비교할 수 있게 span 관례를 따랐다.
    """
    if window <= 1:
        return list(values)
    alpha = 2.0 / (window + 1)
    out, cur = [], values[0]
    for v in values:
        cur = alpha * v + (1 - alpha) * cur
        out.append(cur)
    return out


def median_filter(values: list[float], window: int) -> list[float]:
    """후행 이동중앙값. 평균과 달리 단발 스파이크에 끌려가지 않는다."""
    if window <= 1:
        return list(values)
    return [statistics.median(values[max(0, i - window + 1):i + 1])
            for i in range(len(values))]


def _endpoint_fit(y: list[float], degree: int) -> float:
    """최근 구간에 다항식을 적합해 마지막 시점의 값을 돌려준다.

    x를 끝점이 0이 되게 잡으면(마지막 = 0, 그 이전이 음수) 적합값의
    끝점 예측이 곧 상수항이 되어 계산이 간단해진다.
    """
    n = len(y)
    if n == 1:
        return y[0]
    x = [i - (n - 1) for i in range(n)]      # ..., -2, -1, 0
    if degree == 1:
        xbar, ybar = statistics.fmean(x), statistics.fmean(y)
        sxx = sum((xi - xbar) ** 2 for xi in x)
        if sxx == 0:
            return ybar
        b = sum((xi - xbar) * (yi - ybar) for xi, yi in zip(x, y)) / sxx
        return ybar + b * (0 - xbar)          # x=0(끝점)에서의 값
    # degree 2: 정규방정식을 직접 푼다 (x=0에서의 값 = 상수항)
    if n < 3:
        return y[-1]
    s = [sum(xi ** p for xi in x) for p in range(5)]
    t = [sum(yi * xi ** p for xi, yi in zip(x, y)) for p in range(3)]
    # [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] @ [c,b,a] = [t0,t1,t2]
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for col in range(3):                      # 가우스 소거
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return statistics.fmean(y)
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, 4):
                m[r][c] -= f * m[col][c]
    return m[0][3] / m[0][0]                  # 상수항 = x=0에서의 적합값


def savgol_endpoint(values: list[float], window: int,
                    degree: int = 1) -> list[float]:
    """국소 다항 적합의 끝점값 (인과 Savitzky-Golay).

    이동평균의 추세 지연을 없앤다 — 직선을 적합해 끝점을 읽으므로
    단조 상승 구간에서 현재값을 과소평가하지 않는다.
    """
    if window <= 1:
        return list(values)
    return [_endpoint_fit(values[max(0, i - window + 1):i + 1], degree)
            for i in range(len(values))]


MIN_HAMPEL_REF = 3   # 참조 구간이 이보다 짧으면 판정하지 않는다


def hampel(values: list[float], window: int,
           n_sigma: float = DEFAULT_HAMPEL_SIGMA) -> list[float]:
    """후행 Hampel 이상치 제거. 직전 구간의 중앙값에서 n_sigma·MAD를 넘게
    벗어나면 그 중앙값으로 대체한다.

    평활이 아니라 스파이크 제거다. 평활 전에 걸어 단발 이상치가 평균·
    회귀를 흔드는 것을 막는다.

    참조 구간에서 **현재 값은 제외**한다. 포함하면 스파이크가 자기 자신의
    산포 추정을 부풀려 판정을 빠져나간다(MAD가 0이 되어 대체값으로 쓴
    표준편차가 스파이크에 오염되는 경우). 구간이 너무 짧으면 산포를 믿을
    수 없으므로 판정을 보류한다.
    """
    if window <= 1:
        return list(values)
    out = []
    for i, v in enumerate(values):
        ref = values[max(0, i - window + 1):i]      # 현재 값 제외
        if len(ref) < MIN_HAMPEL_REF:
            out.append(v)
            continue
        med = statistics.median(ref)
        scale = MAD_TO_SIGMA * statistics.median([abs(r - med) for r in ref])
        limit = n_sigma * scale
        is_outlier = abs(v - med) > limit if limit > 0 else v != med
        out.append(med if is_outlier else v)
    return out


SMOOTHERS = {
    "ma": moving_average,
    "ewma": ewma,
    "median": median_filter,
    "savgol": lambda v, w: savgol_endpoint(v, w, degree=1),
    "savgol2": lambda v, w: savgol_endpoint(v, w, degree=2),
}


def smooth(values: list[float], window: int, method: str = "ma",
           despike: bool = False) -> list[float]:
    """선택한 필터로 평활. despike면 Hampel을 먼저 통과시킨다."""
    if despike:
        values = hampel(values, window)
    return SMOOTHERS[method](values, window)


def smooth_engines(engines: dict[int, dict[str, list[float]]], window: int,
                   method: str = "ma", despike: bool = False
                   ) -> dict[int, dict[str, list[float]]]:
    """모든 유닛·센서에 평활 적용."""
    if window <= 1 and not despike:
        return engines
    return {u: {s: smooth(v, window, method, despike) for s, v in eng.items()}
            for u, eng in engines.items()}


def normalize_per_engine(engines: dict[int, dict[str, list[float]]],
                         baseline_n: int) -> dict[int, dict[str, list[float]]]:
    """엔진별 초기 구간 기준 z-정규화.

    조건별 정규화는 운전 조건 차이를 없앨 뿐 **엔진 개체차**는 남긴다.
    한계치(limits)는 엔진들의 고장 시점 값 중앙값이라 개체 offset이 섞여
    있는데, 엔진마다 기준선이 다르면 같은 한계치가 어떤 엔진엔 가깝고
    어떤 엔진엔 멀어진다. 초기 구간으로 각 엔진을 정규화하면 이 편차가
    사라진다.
    """
    out = {}
    for u, eng in engines.items():
        norm = {}
        for s, series in eng.items():
            base = series[:baseline_n]
            mu = statistics.fmean(base)
            sd = statistics.pstdev(base) or 1e-9
            norm[s] = [(v - mu) / sd for v in series]
        out[u] = norm
    return out
