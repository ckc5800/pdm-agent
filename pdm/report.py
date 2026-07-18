"""센서 차트(SVG) + 진단 리포트(markdown) 생성."""
from pathlib import Path

from .detectors import AnomalyEvent
from .simulator import MachineData

COLORS = {"vibration": "#74c7ec", "temperature": "#f38ba8"}


def _polyline(values, x0, y0, w, h, vmin, vmax, step):
    pts = []
    n = len(values)
    for i in range(0, n, step):
        x = x0 + w * i / n
        y = y0 + h - h * (values[i] - vmin) / (vmax - vmin)
        pts.append(f"{x:.0f},{y:.0f}")
    return " ".join(pts)


def chart_svg(data: MachineData, events: list[AnomalyEvent]) -> str:
    W, H, PAD = 900, 420, 55
    w, h = W - PAD * 2, (H - PAD * 3) // 2
    n = len(data.vibration)
    step = max(1, n // 600)

    def panel(values, sensor, y0, label, unit):
        vmin, vmax = min(values) * 0.95, max(values) * 1.05
        spans = ""
        for e in events:
            if e.sensor != sensor:
                continue
            x1 = PAD + w * e.start_idx / n
            x2 = PAD + w * e.end_idx / n
            color = "#f9e2af" if e.severity == "warning" else "#f38ba8"
            spans += (f'<rect x="{x1:.0f}" y="{y0}" width="{max(x2 - x1, 2):.0f}" '
                      f'height="{h}" fill="{color}" opacity="0.25"/>')
        line = _polyline(values, PAD, y0, w, h, vmin, vmax, step)
        return (spans +
                f'<polyline points="{line}" fill="none" stroke="{COLORS[sensor]}" stroke-width="1.5"/>'
                f'<text x="{PAD}" y="{y0 - 8}" fill="#cdd6f4" font-size="13">{label} ({unit})</text>'
                f'<line x1="{PAD}" y1="{y0 + h}" x2="{PAD + w}" y2="{y0 + h}" stroke="#45475a"/>')

    body = panel(data.vibration, "vibration", PAD + 10, "진동 RMS", "mm/s")
    body += panel(data.temperature, "temperature", PAD * 2 + h + 10, "온도", "°C")

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Consolas, monospace">
  <rect width="{W}" height="{H}" rx="10" fill="#1e1e2e"/>
  <text x="{W / 2:.0f}" y="30" text-anchor="middle" fill="#cdd6f4" font-size="15">{data.machine_id} — 48시간 센서 추이와 탐지된 이상 구간</text>
  {body}
  <rect x="{PAD}" y="{H - 24}" width="12" height="12" fill="#f9e2af" opacity="0.5"/>
  <text x="{PAD + 18}" y="{H - 13}" fill="#9399b2" font-size="12">warning 구간</text>
  <rect x="{PAD + 130}" y="{H - 24}" width="12" height="12" fill="#f38ba8" opacity="0.5"/>
  <text x="{PAD + 148}" y="{H - 13}" fill="#9399b2" font-size="12">critical 구간</text>
</svg>'''


def save_report(data: MachineData, events: list[AnomalyEvent],
                diagnosis: str, out_dir: str = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / "chart.svg").write_text(chart_svg(data, events), encoding="utf-8")

    lines = [
        f"# 예지보전 진단 리포트 — {data.machine_id}",
        "",
        "![chart](chart.svg)",
        "",
        "## 탐지된 이상 징후",
        "",
        "| 심각도 | 센서 | 구간(분) | 탐지기 | 근거 |",
        "|---|---|---|---|---|",
    ]
    for e in events:
        lines.append(f"| {e.severity} | {e.sensor} | {e.start_idx}~{e.end_idx} "
                     f"| {e.detector} | {e.evidence} |")
    lines += ["", "## LLM 진단", "", diagnosis, ""]
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
