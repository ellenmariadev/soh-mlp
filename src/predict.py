"""Inferência de SoC para um arquivo CSV do dataset.

Uso:
    python src/predict.py "data/Cell 1/Cell 1/Cell 1 Measurements - ....csv"
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_io import OUTPUT_DIR, parse_test_info, read_cell_file
from features import FEATURES, build_features


def estimate_soc(csv_path: str | Path, model_path: str | Path | None = None) -> pd.DataFrame:
    model_path = Path(model_path) if model_path else OUTPUT_DIR / "model_soc_mlp.joblib"
    model = joblib.load(model_path)

    path = Path(csv_path)
    info = parse_test_info(path.name)
    df = read_cell_file(path)
    df["cell"] = info["cell"]
    df["cell_id"] = info["cell_id"]
    df["test"] = info["test"]
    df["cycle_index"] = info["cycle_index"]
    df["regime"] = info["regime"]
    df["soc"] = 0.0 

    feat = build_features(df)

    feat = feat[feat["current"] < 0].copy()
    if feat.empty:
        raise RuntimeError("Nenhuma amostra de descarga encontrada no arquivo.")
    pred = np.clip(model.predict(feat[FEATURES]), 0.0, 1.0)
    return pd.DataFrame(
        {
            "timestamp": feat["timestamp"],
            "voltage": feat["voltage"],
            "current": feat["current"],
            "temperature": feat["temperature"],
            "soc_estimado": pred,
        }
    )


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    result = estimate_soc(sys.argv[1])
    print(result.head(10).to_string(index=False))
    print(f"\n{len(result)} amostras estimadas.")


if __name__ == "__main__":
    main()
