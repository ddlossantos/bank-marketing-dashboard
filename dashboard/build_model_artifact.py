from __future__ import annotations

import json

from .app import MODEL_ARTIFACT_PATH, build_regression_model


def main() -> None:
    artifact, metrics = build_regression_model()

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
