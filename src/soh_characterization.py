from __future__ import annotations

import pandas as pd

from data_io import (
    FULL_DISCHARGE_AH,
    NOMINAL_CAPACITY_AH,
    OUTPUT_DIR,
    discharge_segments,
    load_records,
)


def capacity_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        segs = [s for s in discharge_segments(rec["df"]) if s["ah"] >= FULL_DISCHARGE_AH]
        if not segs:
            continue
        seg = max(segs, key=lambda s: s["ah"])
        rows.append(
            {
                "cell": rec["cell"],
                "test": rec["test"],
                "cycle_index": rec["cycle_index"],
                "regime": rec["regime"],
                "capacity_ah": seg["ah"],
                "soh": seg["ah"] / NOMINAL_CAPACITY_AH,
                "duration_h": seg["duration_h"],
                "v_end": seg["v_end"],
                "i_mean": seg["i_mean"],
            }
        )
    return pd.DataFrame(rows).sort_values(["cell", "cycle_index"]).reset_index(drop=True)


def summary_table(cap: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cap.groupby("cell")
        .agg(
            n_full_discharges=("capacity_ah", "size"),
            capacity_mean_ah=("capacity_ah", "mean"),
            soh_mean=("soh", "mean"),
            soh_min=("soh", "min"),
            soh_max=("soh", "max"),
        )
        .reset_index()
    )
    return summary.round(4)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()

    cap = capacity_table(records)
    summary = summary_table(cap)
    cap.to_csv(OUTPUT_DIR / "soh_capacity.csv", index=False, sep=";", decimal=",")
    summary.to_csv(OUTPUT_DIR / "soh.csv", index=False, sep=";", decimal=",")

    lines = [
        "SoH por capacidade medida",
        "=" * 40,
        f"Capacidade nominal: {NOMINAL_CAPACITY_AH:.0f} Ah",
        f"Descargas completas: {len(cap)}",
        f"SoH global: {cap['soh'].min():.4f} a {cap['soh'].max():.4f}",
        "",
        summary.to_string(index=False),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    print(f"\nArquivos salvos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
