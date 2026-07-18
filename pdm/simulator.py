"""설비 센서 시뮬레이터.

회전기계(모터+베어링)를 단순화한 모델로 진동/온도/전류 시계열을 생성하고,
지정한 시점에 결함(베어링 마모, 과열, 순간 충격)을 주입한다.
주입된 결함은 ground truth로 기록되어 탐지기 평가에 사용된다.
"""
import json
import math
import random
from dataclasses import dataclass, field, asdict

SAMPLE_INTERVAL_SEC = 60  # 1분 간격 샘플


@dataclass
class FaultEvent:
    kind: str          # bearing_wear | overheat | impact
    start_idx: int
    end_idx: int


@dataclass
class MachineData:
    machine_id: str
    vibration: list[float] = field(default_factory=list)   # RMS mm/s
    temperature: list[float] = field(default_factory=list) # °C
    current: list[float] = field(default_factory=list)     # A
    faults: list[FaultEvent] = field(default_factory=list) # ground truth

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "MachineData":
        d = json.loads(s)
        d["faults"] = [FaultEvent(**f) for f in d["faults"]]
        return cls(**d)


def simulate(machine_id: str = "PUMP-01", hours: int = 48,
             seed: int = 42, inject: bool = True) -> MachineData:
    """정상 운전 + (선택) 결함 주입 시계열 생성."""
    rng = random.Random(seed)
    n = hours * 60 // (SAMPLE_INTERVAL_SEC // 60) // 1  # 분 단위 샘플 수
    n = hours * 60  # 1분 간격

    data = MachineData(machine_id=machine_id)

    # 결함 시나리오 결정 (총 구간의 후반부에 배치)
    if inject:
        wear_start = int(n * 0.55)                     # 점진적 베어링 마모
        overheat_start = int(n * 0.75)
        overheat_end = int(n * 0.80)                   # 일시 과열
        impact_idx = int(n * 0.35)                     # 순간 충격 (짧은 스파이크)
        data.faults = [
            FaultEvent("impact", impact_idx, impact_idx + 3),
            FaultEvent("bearing_wear", wear_start, n - 1),
            FaultEvent("overheat", overheat_start, overheat_end),
        ]

    for i in range(n):
        # 기저 신호: 운전 주기(부하 변동) + 노이즈
        load = 0.5 + 0.2 * math.sin(2 * math.pi * i / (8 * 60))  # 8시간 주기 부하
        vib = 1.2 + 0.8 * load + rng.gauss(0, 0.08)
        temp = 45 + 18 * load + rng.gauss(0, 0.5)
        cur = 12 + 6 * load + rng.gauss(0, 0.15)

        if inject:
            # 베어링 마모: 진동이 서서히 증가 + 고주파 성분(지터) 증가
            if i >= data.faults[1].start_idx:
                progress = (i - data.faults[1].start_idx) / (n - data.faults[1].start_idx)
                vib += 2.5 * progress + rng.gauss(0, 0.15 * (1 + 3 * progress))
                temp += 6 * progress
            # 과열: 온도 급상승 구간
            f = data.faults[2]
            if f.start_idx <= i <= f.end_idx:
                temp += 22
                cur += 2.5
            # 순간 충격
            f = data.faults[0]
            if f.start_idx <= i <= f.end_idx:
                vib += 6.0

        data.vibration.append(round(vib, 3))
        data.temperature.append(round(temp, 2))
        data.current.append(round(cur, 3))

    return data


if __name__ == "__main__":
    d = simulate()
    print(f"{d.machine_id}: {len(d.vibration)}개 샘플, 주입 결함 {len(d.faults)}건")
    for f in d.faults:
        print(f"  - {f.kind}: idx {f.start_idx}~{f.end_idx}")
