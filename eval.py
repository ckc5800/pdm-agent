"""탐지기 평가 — 주입된 결함(ground truth) 대비 탐지 성능.

seed를 바꿔 여러 시나리오를 생성하고, 결함 구간과 탐지 구간의
겹침(overlap) 기준으로 결함별 재현율과 오탐 수를 측정한다.
"""
from pdm.detectors import run_all
from pdm.simulator import simulate


def overlaps(a_start, a_end, b_start, b_end, margin=30):
    return a_start <= b_end + margin and b_start - margin <= a_end


def main():
    seeds = [1, 7, 42, 123, 2026]
    total_faults = detected_faults = false_positives = 0

    for seed in seeds:
        data = simulate(seed=seed)
        events = run_all(data)
        matched_events = set()
        for f in data.faults:
            total_faults += 1
            hit = False
            for i, e in enumerate(events):
                if overlaps(e.start_idx, e.end_idx, f.start_idx, f.end_idx):
                    hit = True
                    matched_events.add(i)
            detected_faults += hit
        fp = len(events) - len(matched_events)
        false_positives += fp
        print(f"seed={seed:5}: 결함 {len(data.faults)}건 중 "
              f"{sum(1 for f in data.faults if any(overlaps(e.start_idx, e.end_idx, f.start_idx, f.end_idx) for e in events))}건 탐지, "
              f"오탐 {fp}건")

    recall = detected_faults / total_faults * 100
    print(f"\n결함 재현율: {recall:.0f}% ({detected_faults}/{total_faults}), "
          f"총 오탐: {false_positives}건 (5개 시나리오)")


if __name__ == "__main__":
    main()
