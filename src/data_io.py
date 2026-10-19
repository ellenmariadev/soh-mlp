from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

NOMINAL_CAPACITY_AH = 1200.0
FULL_DISCHARGE_AH = 1000.0

ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
}


def _to_num(series: pd.Series) -> pd.Series:
    x = (
        series.astype(str)
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    both = x.str.contains(",", regex=False) & x.str.contains(".", regex=False)
    x = x.where(~both, x.str.replace(".", "", regex=False))
    x = x.str.replace(",", ".", regex=False)
    return pd.to_numeric(x, errors="coerce")


def _find(cols: list[str], predicate) -> str | None:
    for c in cols:
        if predicate(c):
            return c
    return None


def _pick_columns(cols: list[str]) -> dict[str, str | None]:
    lower = {c: c.strip().lower() for c in cols}

    def is_timestamp(c: str) -> bool:
        return lower[c] == "timestamp"

    def is_voltage(c: str) -> bool:
        cl = lower[c]
        return bool(re.match(r"^v\d+\s*\(", cl)) and "(v)" in cl and "mv" not in cl

    def is_temperature(c: str) -> bool:
        return bool(re.match(r"^t\d+", lower[c]))

    def is_current(c: str) -> bool:
        cl = lower[c]
        return cl == "current" or (cl.startswith("i") and "(a)" in cl)

    def is_ah(c: str) -> bool:
        return lower[c].strip("() ").lower() == "ah"

    def is_soc(c: str) -> bool:
        return "soc" in lower[c]

    return {
        "timestamp": _find(cols, is_timestamp),
        "voltage": _find(cols, is_voltage),
        "temperature": _find(cols, is_temperature),
        "current": _find(cols, is_current),
        "ah": _find(cols, is_ah),
        "soc_raw": _find(cols, is_soc),
    }


def read_cell_file(path: Path) -> pd.DataFrame:
    """Le um CSV e devolve colunas normalizadas."""
    raw = pd.read_csv(path, sep=";", encoding="cp1252", dtype=str, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    picked = _pick_columns(list(raw.columns))

    if picked["timestamp"] is None or picked["voltage"] is None:
        raise ValueError(f"Colunas essenciais ausentes em {Path(path).name}")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(
        raw[picked["timestamp"]], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    out["voltage"] = _to_num(raw[picked["voltage"]])
    out["current"] = _to_num(raw[picked["current"]]) if picked["current"] else np.nan
    out["temperature"] = (
        _to_num(raw[picked["temperature"]]) if picked["temperature"] else np.nan
    )
    out["ah"] = _to_num(raw[picked["ah"]]) if picked["ah"] else np.nan
    out["soc_raw"] = _to_num(raw[picked["soc_raw"]]) if picked["soc_raw"] else np.nan

    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out


def parse_test_info(filename: str) -> dict:
    stem = Path(filename).name.replace(".csv", "")
    m = re.match(r"Cell\s+(\d+)\s+Measurements\s*-\s*(.*)", stem)
    if not m:
        raise ValueError(f"Nome de arquivo inesperado: {filename}")
    cell = int(m.group(1))
    test = m.group(2).strip()

    first_word = re.split(r"[\s-]", test.lower())[0]
    cycle_index = ORDINALS.get(first_word, -1)

    low = test.lower()
    if "pulse" in low:
        regime = "Pulses"
    elif "wind" in low:
        regime = "Wind"
    elif "gc" in low:
        regime = "GC"
    elif "c20" in low:
        regime = "C20"
    elif "c10" in low:
        regime = "C10"
    else:
        regime = "Other"

    return {
        "cell": f"Cell {cell}",
        "cell_id": cell,
        "test": test,
        "cycle_index": cycle_index,
        "regime": regime,
    }


def load_records() -> list[dict]:
    """Carrega todos os arquivos e ordena cronologicamente por celula."""
    records = []
    for path in sorted(DATA_DIR.glob("Cell */*/*.csv")):
        info = parse_test_info(path.name)
        df = read_cell_file(path)
        if df.empty:
            continue
        records.append({**info, "file": path.name, "path": str(path), "df": df})
    records.sort(key=lambda r: (r["cell_id"], r["df"]["timestamp"].min()))
    return records


def discharge_segments(df: pd.DataFrame, min_ah: float = 1.0) -> list[dict]:
    """Segmentos contiguos de descarga (corrente < 0), com carga integrada."""
    current = df["current"].to_numpy()
    voltage = df["voltage"].to_numpy()
    ts = df["timestamp"].to_numpy()
    active = np.where(np.isnan(current), False, current < -1e-9)

    segments = []
    start = None
    for i in range(len(active) + 1):
        is_active = i < len(active) and active[i]
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            end = i - 1
            seg_i = np.arange(start, end + 1)
            dt_min = (ts[seg_i[-1]] - ts[seg_i[0]]) / np.timedelta64(1, "m")
            ah = float(np.nansum(np.abs(current[seg_i]))) / 60.0
            if ah >= min_ah:
                segments.append(
                    {
                        "start": start,
                        "end": end,
                        "ah": ah,
                        "duration_h": float(dt_min) / 60.0,
                        "v_start": float(np.nanmin(voltage[seg_i[: min(5, len(seg_i))]])),
                        "v_end": float(voltage[seg_i[-1]]),
                        "v_min": float(np.nanmin(voltage[seg_i])),
                        "i_mean": float(np.nanmean(current[seg_i])),
                    }
                )
            start = None
    return segments


def build_manifest(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        df = rec["df"]
        segs = discharge_segments(df)
        dis_ah = sum(s["ah"] for s in segs)
        charge_mask = df["current"] > 0
        chg_ah = float(df.loc[charge_mask, "current"].sum()) / 60.0
        rows.append(
            {
                "cell": rec["cell"],
                "test": rec["test"],
                "cycle_index": rec["cycle_index"],
                "regime": rec["regime"],
                "file": rec["file"],
                "n_rows": len(df),
                "t_start": df["timestamp"].min(),
                "t_end": df["timestamp"].max(),
                "duration_h": (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600.0,
                "charge_ah": chg_ah,
                "discharge_ah": dis_ah,
                "v_min": float(df["voltage"].min()),
                "v_max": float(df["voltage"].max()),
                "i_min": float(df["current"].min()),
                "i_max": float(df["current"].max()),
                "t_min": float(df["temperature"].min()),
                "t_max": float(df["temperature"].max()),
            }
        )
    return pd.DataFrame(rows)
