"""탐지기 평가 — 주입된 결함(ground truth) 대비 탐지 성능.

seed를 바꿔 여러 시나리오를 생성하고, 결함 구간과 탐지 구간의
겹침(overlap) 기준으로 결함별 재현율과 오탐 수를 측정한다.
"""
from pdm.detectors import run_all
from pdm.simulator import simulate

SEEDS = [1, 7, 42, 123, 2026]


def overlaps(a_start, a_end, b_start, b_end, margin=30):
    return a_start <= b_end + margin and b_start - margin <= a_end


def score_scenario(seed: int) -> tuple[int, int, int]:
    """(주입 결함 수, 탐지된 결함 수, 오탐 이벤트 수)."""
    data = simulate(seed=seed)
    events = run_all(data)

    matched = set()
    detected = 0
    for f in data.faults:
        hits = [i for i, e in enumerate(events)
                if overlaps(e.start_idx, e.end_idx, f.start_idx, f.end_idx)]
        matched.update(hits)
        detected += bool(hits)
    return len(data.faults), detected, len(events) - len(matched)


def main():
    total = detected_total = fp_total = 0
    for seed in SEEDS:
        n_faults, detected, fp = score_scenario(seed)
        total += n_faults
        detected_total += detected
        fp_total += fp
        print(f"seed={seed:5}: 결함 {n_faults}건 중 {detected}건 탐지, 오탐 {fp}건")

    recall = detected_total / total * 100
    print(f"\n결함 재현율: {recall:.0f}% ({detected_total}/{total}), "
          f"총 오탐: {fp_total}건 ({len(SEEDS)}개 시나리오)")


if __name__ == "__main__":
    main()
