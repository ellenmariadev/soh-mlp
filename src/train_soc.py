from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_io import OUTPUT_DIR, load_records  # noqa: E402
from features import FEATURES, TARGET, build_features  # noqa: E402
from soc_reconstruct import build_soc_dataset  # noqa: E402

TRAIN_CELLS = (1, 2, 3, 4)
TEST_CELLS = (5, 6)
RANDOM_STATE = 42


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _split(df: pd.DataFrame):
    train = df[df["cell_id"].isin(TRAIN_CELLS)]
    test = df[df["cell_id"].isin(TEST_CELLS)]
    return train, test


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    df = build_soc_dataset(records)
    df = build_features(df)
    print(f"Amostras: {len(df)}  ({df['segment_id'].nunique()} descargas completas)")

    train, test = _split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Treino: celulas {TRAIN_CELLS} -> {len(train)} amostras")
    print(f"Teste : celulas {TEST_CELLS} -> {len(test)} amostras")

    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "MLP (32,16)": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="tanh",
                solver="adam",
                alpha=1e-4,
                max_iter=500,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({"model": name, "n_train": len(train), "n_test": len(test), **_metrics(y_test, pred)})

    lines = [
        "Estimacao de SoC com descargas completas",
        "=" * 45,
        f"Entradas: {', '.join(FEATURES)}",
        "Alvo    : SoC = 1 - Ah_descarga / Q  (1 -> 0)",
        f"Treino  : celulas {TRAIN_CELLS} ({len(train)} amostras)",
        f"Teste   : celulas {TEST_CELLS} ({len(test)} amostras)",
        "",
        f"{'Modelo':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['model']:<20}{row['mae']:>10.4f}{row['rmse']:>10.4f}{row['r2']:>10.4f}"
        )
    text = "\n".join(lines)
    print("\n" + text)

    model_path = OUTPUT_DIR / "model_soc_mlp.joblib"
    try:
        import joblib

        joblib.dump(models["MLP (32,16)"], model_path)
        print(f"\nModelo salvo em {model_path}")
    except Exception as exc:
        print(f"Nao foi possivel salvar o modelo: {exc}")


if __name__ == "__main__":
    main()
