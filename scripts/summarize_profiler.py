#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def category(name):
    lower = name.lower()
    if "recurrentgateddeltarule" in lower or "fusedgdngating" in lower:
        return "gated_delta"
    if "flashattention" in lower or "attention" in lower:
        return "attention"
    if "swiglu" in lower:
        return "ffn_activation"
    if "norm" in lower or "l2norm" in lower:
        return "norm"
    if "matmul" in lower:
        return "matmul"
    if "all_reduce" in lower or "allreduce" in lower:
        return "tp_collective_kernel"
    if any(x in lower for x in ("transpose", "slice", "cast", "gather", "tensormove",
                                 "concat", "index", "viewcopy", "reshape")):
        return "data_movement"
    return "other"


def one_csv(folder, prefix):
    files = list(folder.glob(f"{prefix}*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected one {prefix} CSV in {folder}, got {len(files)}")
    return files[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    chart = {}
    for config in ("BASE", "EAGER"):
        folder = args.source / config
        op_rows = list(csv.DictReader(one_csv(folder, "op_statistic_").open(encoding="utf-8-sig")))
        totals = defaultdict(float)
        total = 0.0
        for row in op_rows:
            value = float(row["Total Time(us)"])
            totals[category(row["OP Type"])] += value
            total += value
        chart[config] = {key: value / total * 100 for key, value in totals.items()}
        for key, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            rows.append({"config": config, "scope": "npu_op_category", "name": key,
                         "total_time_us": f"{value:.3f}", "count": "",
                         "share_pct": f"{value / total * 100:.3f}"})
        comm_rows = csv.DictReader(one_csv(folder, "communication_statistic_").open(encoding="utf-8-sig"))
        for row in comm_rows:
            rows.append({"config": config, "scope": "communication", "name": row["OP Type"],
                         "total_time_us": row["Total Time(us)"], "count": row["Count"],
                         "share_pct": row["Ratio(%)"]})
        api_rows = sorted(csv.DictReader(one_csv(folder, "api_statistic_").open(encoding="utf-8-sig")),
                          key=lambda row: float(row["Time(us)"]), reverse=True)[:12]
        api_total = sum(float(row["Time(us)"]) for row in api_rows)
        for row in api_rows:
            rows.append({"config": config, "scope": "host_api_top12",
                         "name": f'{row["Level"]}:{row["API Name"]}',
                         "total_time_us": row["Time(us)"], "count": row["Count"],
                         "share_pct": f'{float(row["Time(us)"]) / api_total * 100:.3f}'})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categories = sorted(set(chart["BASE"]) | set(chart["EAGER"]),
                        key=lambda key: chart["BASE"].get(key, 0), reverse=True)
    width, row_h = 960, 34
    height = 85 + row_h * len(categories)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#172033}.label{font-size:13px}.title{font-size:20px;font-weight:700}.value{font-size:12px}</style>',
           '<text x="20" y="30" class="title">NPU operator category share: BASE vs EAGER</text>',
           '<rect x="680" y="16" width="14" height="14" fill="#2878b5"/><text x="700" y="28" class="label">BASE</text>',
           '<rect x="770" y="16" width="14" height="14" fill="#d95f02"/><text x="790" y="28" class="label">EAGER</text>']
    for index, key in enumerate(categories):
        y = 62 + index * row_h
        base, eager = chart["BASE"].get(key, 0), chart["EAGER"].get(key, 0)
        svg += [f'<text x="20" y="{y + 14}" class="label">{key}</text>',
                f'<rect x="190" y="{y}" width="{base * 7:.1f}" height="12" fill="#2878b5"/>',
                f'<rect x="190" y="{y + 15}" width="{eager * 7:.1f}" height="12" fill="#d95f02"/>',
                f'<text x="{200 + base * 7:.1f}" y="{y + 11}" class="value">{base:.1f}%</text>',
                f'<text x="{200 + eager * 7:.1f}" y="{y + 26}" class="value">{eager:.1f}%</text>']
    svg.append('</svg>')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.figure.write_text("\n".join(svg) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
