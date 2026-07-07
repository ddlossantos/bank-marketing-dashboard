from __future__ import annotations

import joblib
import sklearn

from .app import MODEL_ARTIFACT_PATH, build_regression_model


def main() -> None:
    model, metrics = build_regression_model()

    # The server only predicts one record at a time, so keep inference lightweight.
    model.set_params(model__n_jobs=1)

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "metrics": metrics,
            "metadata": {
                "artifact_version": 1,
                "scikit_learn_version": sklearn.__version__,
                "source_dataset": "data/bank/bank-full.csv",
            },
        },
        MODEL_ARTIFACT_PATH,
        compress=3,
    )
    print(f"Modelo guardado en {MODEL_ARTIFACT_PATH}")
    print(
        "Metricas: "
        f"R2 test={metrics['r2_test']:.3f}, "
        f"MAE test={metrics['mae_test']:,.0f}, "
        f"RMSE test={metrics['rmse_test']:,.0f}"
    )


if __name__ == "__main__":
    main()
