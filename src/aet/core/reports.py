import csv
from pathlib import Path

from aet.core.metrics import mean_std, fmt, coerce_na

def write_csv(output_path: Path, rows: list[dict], columns: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: coerce_na(row.get(col)) for col in columns})

def write_ablation_table(
    output_path: Path,
    rows: list[dict],
    numeric_cols: list[str],
    title: str,
) -> None:
    methods = sorted({r["method"] for r in rows})
    lines = [
        f"# Ablation Table: {title}\n",
        "| method | seeds | " + " | ".join(numeric_cols) + " |",
        "|---|---|" + "|".join("---" for _ in numeric_cols) + "|",
    ]
    for method in methods:
        method_rows = [r for r in rows if r["method"] == method]
        seeds = sorted({r["seed"] for r in method_rows})
        cells = []
        for col in numeric_cols:
            vals = [r.get(col) for r in method_rows]
            mean, std = mean_std(vals)
            cells.append(fmt(mean, std))
        lines.append(
            f"| {method} | {','.join(str(s) for s in seeds)} | "
            + " | ".join(cells) + " |"
        )
    if not methods:
        lines.append("| *(no real baseline runs yet)* | — |" + "|".join("NA" for _ in numeric_cols) + "|")
    output_path.write_text("\n".join(lines) + "\n")
