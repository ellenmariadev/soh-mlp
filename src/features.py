from __future__ import annotations

import pandas as pd

FEATURES = ["voltage", "current", "temperature"]
TARGET = "soc"

V_MIN, V_MAX = 1.5, 3.0
I_MIN, I_MAX = -200.0, 200.0
T_MIN, T_MAX = 0.0, 60.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["voltage"].between(V_MIN, V_MAX)]
    df = df[df["current"].between(I_MIN, I_MAX)]
    df = df[df["temperature"].between(T_MIN, T_MAX)]

    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    df.attrs["dropped_rows"] = before - len(df)
    return df
