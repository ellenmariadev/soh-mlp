from __future__ import annotations

import numpy as np
import pandas as pd

from data_io import (
    FULL_DISCHARGE_AH,
    OUTPUT_DIR,
    build_manifest,
    discharge_segments,
    load_records,
)

SOC_COLUMNS = [
    "cell",
    "test",
    "cycle_index",
    "regime",
    "segment_id",
    "segment_ah",
    "timestamp",
    "voltage",
    "current",
    "temperature",
    "ah",
    "soc_raw",
    "soc",
]


def build_soc_dataset(
    records: list[dict], full_threshold: float = FULL_DISCHARGE_AH
) -> pd.DataFrame:
    """Extrai trechos de descarga completa e define o alvo SoC físico."""
    frames = []
    for rec in records:
        df = rec["df"]
        for seg in discharge_segments(df):
            if seg["ah"] < full_threshold:
                continue
            sub = df.iloc[seg["start"] : seg["end"] + 1].copy()
            dt_min = (
                sub["timestamp"].diff().dt.total_seconds() / 60.0
            ).fillna(1.0).clip(lower=0.0)
            cum_ah = (sub["current"].abs() * dt_min / 60.0).cumsum()
            sub["soc"] = np.clip(1.0 - cum_ah / seg["ah"], 0.0, 1.0)
            sub["segment_id"] = f"{rec['cell']}|{rec['test']}|{seg['start']}"
            sub["segment_ah"] = seg["ah"]
            sub["cell"] = rec["cell"]
            sub["cell_id"] = rec["cell_id"]
            sub["test"] = rec["test"]
            sub["cycle_index"] = rec["cycle_index"]
            sub["regime"] = rec["regime"]
            frames.append(sub)

    if not frames:
        raise RuntimeError("Nenhuma descarga completa encontrada.")

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(
        ["cell_id", "cycle_index", "segment_id", "timestamp"]
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records()
    build_manifest(records).to_csv(
        OUTPUT_DIR / "manifest.csv", index=False, sep=";", decimal=","
    )

    df = build_soc_dataset(records)
    ends = df.groupby("segment_id")["soc"].agg(["first", "last"])
    print("Dataset de SoC por descarga completa:")
    print(f"  segmentos: {df['segment_id'].nunique()}")
    print(f"  linhas: {len(df)}")
    print(f"  SoC inicial medio: {ends['first'].mean():.4f}")
    print(f"  SoC final medio:   {ends['last'].mean():.4f}")

    path = OUTPUT_DIR / "soc_dataset.csv"
    df[SOC_COLUMNS].to_csv(path, index=False, sep=";", decimal=",")
    print(f"Salvo em {path}")


if __name__ == "__main__":
    main()
