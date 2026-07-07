from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANK_DATA_PATH = PROJECT_ROOT / "data" / "bank" / "bank-full.csv"
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "dashboard" / "model_artifacts" / "balance_regression_model.json"
MAP_GEOJSON_PATH = PROJECT_ROOT / "data" / "map" / "distritos_inec.geojson"
MAP_PREPARED_GEOJSON_PATH = PROJECT_ROOT / "data" / "map" / "distritos_inec_prepared.geojson"
MAP_VALUES_PATH = PROJECT_ROOT / "data" / "map" / "poblacion_distritos_inec_2023.json"
MAP_VARIABLES_PATH = PROJECT_ROOT / "data" / "map" / "inec_district_variables_selected.json"
MAP_VALUE_DIR = PROJECT_ROOT / "data" / "map"
GEOB_METADATA_PATH = PROJECT_ROOT / "data" / "map" / "geoboundaries_pan_adm2_metadata.json"

COLOR_MAIN = "#18324a"
COLOR_ACCENT = "#19a0a1"
COLOR_MUTED = "#526070"
COLOR_YES = "#19a0a1"
COLOR_NO = "#d95f59"
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["age", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
MISSING_CATEGORY = "__missing__"

DEFAULT_MODEL_METRICS = {
    "r2_train": 0.513998,
    "r2_test": 0.033714,
    "mae_test": 1534.954638,
    "rmse_test": 3076.763544,
}


def clean_label(value: str | float | int | None) -> str:
    if value is None or pd.isna(value):
        return "No informado"
    return str(value).replace("_", " ").title()


@lru_cache(maxsize=1)
def load_bank_data() -> pd.DataFrame:
    df = pd.read_csv(BANK_DATA_PATH, sep=";")
    return df


@lru_cache(maxsize=1)
def load_modeling_data() -> pd.DataFrame:
    return load_bank_data().replace("unknown", np.nan)


def categorical_options(column: str) -> list[dict[str, str]]:
    values = load_bank_data()[column].dropna().astype(str).unique().tolist()
    return [{"label": clean_label(value), "value": value} for value in sorted(values)]


def normalize_category(value) -> str:
    if value is None or pd.isna(value) or value == "unknown":
        return MISSING_CATEGORY
    return str(value)


def _build_feature_matrix(df: pd.DataFrame, artifact: dict) -> np.ndarray:
    columns = [np.ones(len(df), dtype=float)]

    for column, settings in artifact["numeric"].items():
        values = pd.to_numeric(df[column], errors="coerce").fillna(settings["median"]).to_numpy(dtype=float)
        columns.append((values - settings["mean"]) / settings["scale"])

    for column, settings in artifact["categorical"].items():
        values = df[column].map(normalize_category)
        for category in settings["categories"]:
            columns.append((values == category).astype(float).to_numpy())

    return np.column_stack(columns)


def build_regression_model(alpha: float = 100.0):
    df = load_modeling_data()
    target = "balance"
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(df))
    test_count = int(len(df) * 0.2)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    artifact = {
        "model_type": "compact_ridge_linear_balance_v1",
        "target": target,
        "numeric": {},
        "categorical": {},
        "feature_order": ["intercept"],
        "metrics": {},
    }

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(train_df[column], errors="coerce")
        median = float(values.median())
        filled = values.fillna(median)
        mean = float(filled.mean())
        scale = float(filled.std()) or 1.0
        artifact["numeric"][column] = {
            "median": median,
            "mean": mean,
            "scale": scale,
            "coefficient": 0.0,
        }
        artifact["feature_order"].append(f"num:{column}")

    for column in CATEGORICAL_FEATURES:
        categories = sorted(train_df[column].map(normalize_category).unique().tolist())
        artifact["categorical"][column] = {
            "categories": categories,
            "effects": {category: 0.0 for category in categories},
        }
        artifact["feature_order"].extend([f"cat:{column}:{category}" for category in categories])

    x_train = _build_feature_matrix(train_df, artifact)
    y_train = train_df[target].to_numpy(dtype=float)
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ y_train)

    artifact["intercept"] = float(coefficients[0])
    position = 1
    for column in NUMERIC_FEATURES:
        artifact["numeric"][column]["coefficient"] = float(coefficients[position])
        position += 1
    for column in CATEGORICAL_FEATURES:
        for category in artifact["categorical"][column]["categories"]:
            artifact["categorical"][column]["effects"][category] = float(coefficients[position])
            position += 1

    pred_train = _build_feature_matrix(train_df, artifact) @ coefficients
    pred_test = _build_feature_matrix(test_df, artifact) @ coefficients
    y_test = test_df[target].to_numpy(dtype=float)
    ss_res_train = float(np.sum((y_train - pred_train) ** 2))
    ss_tot_train = float(np.sum((y_train - y_train.mean()) ** 2))
    ss_res_test = float(np.sum((y_test - pred_test) ** 2))
    ss_tot_test = float(np.sum((y_test - y_test.mean()) ** 2))
    metrics = {
        "r2_train": 1 - ss_res_train / ss_tot_train if ss_tot_train else 0.0,
        "r2_test": 1 - ss_res_test / ss_tot_test if ss_tot_test else 0.0,
        "mae_test": float(np.mean(np.abs(y_test - pred_test))),
        "rmse_test": float(np.sqrt(np.mean((y_test - pred_test) ** 2))),
    }
    artifact["metrics"] = metrics
    return artifact, metrics


@lru_cache(maxsize=1)
def load_regression_model():
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro el modelo preentrenado en {MODEL_ARTIFACT_PATH}. "
            "Genere el artefacto con: python -m dashboard.build_model_artifact"
        )

    artifact = json.loads(MODEL_ARTIFACT_PATH.read_text(encoding="utf-8"))
    return artifact, artifact.get("metrics", DEFAULT_MODEL_METRICS)


def predict_regression_model(artifact: dict, values: dict) -> float:
    prediction = float(artifact["intercept"])

    for column, settings in artifact["numeric"].items():
        raw_value = values.get(column)
        value = settings["median"] if raw_value is None else float(raw_value)
        prediction += ((value - settings["mean"]) / settings["scale"]) * settings["coefficient"]

    for column, settings in artifact["categorical"].items():
        category = normalize_category(values.get(column))
        prediction += settings["effects"].get(category, 0.0)

    return prediction


def load_model_metrics() -> dict:
    try:
        _, metrics = load_regression_model()
    except Exception as exc:
        logger.warning("No se pudieron cargar las metricas del modelo de balance: %s", exc)
        return DEFAULT_MODEL_METRICS
    return metrics


@lru_cache(maxsize=1)
def load_map_base():
    geojson = json.loads(MAP_PREPARED_GEOJSON_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(GEOB_METADATA_PATH.read_text(encoding="utf-8"))
    return geojson, metadata


@lru_cache(maxsize=1)
def load_map_variables() -> tuple[dict[int, dict], list[dict]]:
    variables = json.loads(MAP_VARIABLES_PATH.read_text(encoding="utf-8"))
    simple_ids = {179, 82, 83, 84, 86, 90, 91, 92, 94}
    variables = [item for item in variables if int(item["id"]) in simple_ids]
    variable_map = {int(item["id"]): item for item in variables}
    options = [{"label": item["nombre"], "value": int(item["id"])} for item in variables]
    return variable_map, options


@lru_cache(maxsize=16)
def load_map_data(variable_id: int = 179):
    geojson, metadata = load_map_base()
    variable_map, _ = load_map_variables()
    variable = variable_map.get(int(variable_id), variable_map[179])
    table_path = MAP_VALUE_DIR / f"map_table_variable_{int(variable['id'])}.json"
    table = pd.DataFrame(json.loads(table_path.read_text(encoding="utf-8")))

    value_lookup = table.set_index("join_code")["value"].to_dict()
    unit = variable.get("unidad_medida") or ""
    variable_name = variable["nombre"]
    geojson = json.loads(json.dumps(geojson))
    for feature in geojson["features"]:
        join_code = feature["properties"].get("join_code")
        feature["properties"]["value"] = value_lookup.get(join_code, 0)
        feature["properties"]["unit"] = unit
        feature["properties"]["variable_name"] = variable_name

    return geojson, table, metadata, variable


def filter_data(selected_jobs, age_range, month, housing, loan) -> pd.DataFrame:
    df = load_bank_data().copy()
    if selected_jobs:
        df = df[df["job"].isin(selected_jobs)]
    if age_range:
        df = df[df["age"].between(age_range[0], age_range[1])]
    if month and month != "all":
        df = df[df["month"] == month]
    if housing and housing != "all":
        df = df[df["housing"] == housing]
    if loan and loan != "all":
        df = df[df["loan"] == loan]
    return df


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font={"size": 16})
    fig.update_layout(template="plotly_white", xaxis={"visible": False}, yaxis={"visible": False})
    return fig


def acceptance_by_job_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Sin datos para los filtros seleccionados")
    grouped = (
        df.assign(accepted=(df["y"] == "yes").astype(int))
        .groupby("job", as_index=False)
        .agg(acceptance_rate=("accepted", "mean"), clients=("accepted", "size"))
        .sort_values("acceptance_rate", ascending=False)
    )
    grouped["acceptance_rate_pct"] = grouped["acceptance_rate"] * 100
    fig = px.bar(
        grouped,
        x="acceptance_rate_pct",
        y="job",
        orientation="h",
        color="clients",
        color_continuous_scale="Teal",
        labels={"acceptance_rate_pct": "Aceptación (%)", "job": "Ocupación", "clients": "Clientes"},
        title="Tasa de aceptación por ocupación",
    )
    fig.update_layout(template="plotly_white", yaxis={"categoryorder": "total ascending"}, margin=dict(l=20, r=20, t=60, b=30))
    return fig


def age_distribution_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Sin datos para los filtros seleccionados")
    fig = px.histogram(
        df,
        x="age",
        color="y",
        nbins=32,
        barmode="overlay",
        color_discrete_map={"yes": COLOR_YES, "no": COLOR_NO},
        labels={"age": "Edad", "count": "Clientes", "y": "Aceptó"},
        title="Distribución de edad por aceptación",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=30))
    return fig


def duration_box_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Sin datos para los filtros seleccionados")
    fig = px.box(
        df,
        x="y",
        y="duration",
        color="y",
        color_discrete_map={"yes": COLOR_YES, "no": COLOR_NO},
        labels={"duration": "Duración de llamada (segundos)", "y": "Aceptó"},
        title="Duración de llamada según aceptación",
    )
    fig.update_layout(template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=60, b=30))
    return fig


def correlation_figure(df: pd.DataFrame) -> go.Figure:
    if df.empty or len(df) < 5:
        return empty_figure("No hay suficientes datos para correlación")
    corr_df = df.copy()
    corr_df["y_binary"] = (corr_df["y"] == "yes").astype(int)
    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous", "y_binary"]
    corr = corr_df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlación de variables numéricas",
    )
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=30))
    return fig


def format_map_value(value: float, unit: str) -> str:
    if "%" in unit:
        return f"{value:.1f}%"
    if "dólares" in unit.lower() or "dolares" in unit.lower():
        return f"${value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def map_figure(variable_id: int = 179) -> go.Figure:
    geojson, district_table, _, variable = load_map_data(int(variable_id))
    unit = variable.get("unidad_medida") or ""
    fig = px.choropleth_mapbox(
        district_table,
        geojson=geojson,
        locations="join_code",
        featureidkey="properties.join_code",
        color="value",
        hover_name="district_name",
        hover_data={
            "join_code": False,
            "value": ":,.2f" if "%" in unit else ":,.0f",
            "area_km2": ":,.1f",
        },
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        center={"lat": 8.45, "lon": -80.1},
        zoom=6.35,
        opacity=0.80,
        labels={"value": unit or "Valor", "area_km2": "Área km2"},
        title=f"{variable['nombre']} por distrito de Panamá",
        custom_data=["join_code", "district_name", "PROV_NOMB", "value", "unit", "area_km2"],
    )
    fig.update_traces(marker_line_width=0.35, marker_line_color="#ffffff")
    fig.update_layout(margin=dict(l=0, r=0, t=55, b=0), coloraxis_colorbar_title=unit or "Valor")
    return fig


def map_summary_figure(variable_id: int = 179) -> go.Figure:
    _, district_table, _, variable = load_map_data(int(variable_id))
    unit = variable.get("unidad_medida") or ""
    is_total_variable = not bool(variable.get("is_porcentaje")) and unit.lower() == "personas"
    agg_name = "Total provincial" if is_total_variable else "Promedio distrital"
    grouped = (
        district_table.groupby("PROV_NOMB", as_index=False)
        .agg(value=("value", "sum" if is_total_variable else "mean"), districts=("join_code", "count"))
        .sort_values("value", ascending=True)
    )
    fig = px.bar(
        grouped,
        x="value",
        y="PROV_NOMB",
        orientation="h",
        color="value",
        color_continuous_scale="Viridis",
        labels={"value": unit or "Valor", "PROV_NOMB": "Provincia"},
        title="Resumen por provincia",
        hover_data={"districts": True, "value": ":,.2f" if "%" in unit else ":,.0f"},
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


def selected_district_text(click_data, variable_id: int = 179):
    _, district_table, _, variable = load_map_data(int(variable_id))
    unit = variable.get("unidad_medida") or ""
    if click_data and click_data.get("points"):
        join_code = click_data["points"][0]["customdata"][0]
        row = district_table[district_table["join_code"] == join_code]
        item = row.iloc[0] if not row.empty else district_table.sort_values("value", ascending=False).iloc[0]
    else:
        item = district_table.sort_values("value", ascending=False).iloc[0]

    return [
        html.P("Distrito seleccionado", className="kpi-label"),
        html.P(item["district_name"], className="map-detail-title"),
        html.P(f"Valor: {format_map_value(float(item['value']), unit)} {unit}", className="map-detail-line"),
        html.P(f"Área aproximada: {float(item['area_km2']):,.1f} km2", className="map-detail-line"),
    ]


bank_df = load_bank_data()
model_metrics = load_model_metrics()
geojson_map, district_table, geob_meta, default_map_variable = load_map_data(179)
map_variable_map, map_variable_options = load_map_variables()

job_options = categorical_options("job")
month_options = [{"label": "Todos", "value": "all"}] + categorical_options("month")
yes_no_options = [{"label": "Todos", "value": "all"}, {"label": "Sí", "value": "yes"}, {"label": "No", "value": "no"}]

app = Dash(__name__, title="Dashboard Bank Marketing")
server = app.server

app.layout = html.Div(
    className="app-shell",
    children=[
        html.Header(
            className="topbar",
            children=[
                html.H1("Bank Marketing Dashboard"),
                html.P(
                    "Análisis interactivo de campañas de telemarketing bancario, modelos predictivos y mapa sociodemográfico distrital de Panamá."
                ),
            ],
        ),
        html.Main(
            className="content",
            children=[
                html.Section(
                    className="panel control-panel",
                    children=[
                        html.H2("Filtros de análisis", className="section-title"),
                        html.Div(
                            className="controls-grid",
                            children=[
                                html.Div(
                                    className="control-block",
                                    children=[
                                        html.Label("Ocupación"),
                                        dcc.Dropdown(
                                            id="job-filter",
                                            options=job_options,
                                            value=[],
                                            multi=True,
                                            placeholder="Todas las ocupaciones",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-block",
                                    children=[
                                        html.Label("Rango de edad"),
                                        dcc.RangeSlider(
                                            id="age-filter",
                                            min=int(bank_df["age"].min()),
                                            max=int(bank_df["age"].max()),
                                            value=[int(bank_df["age"].min()), int(bank_df["age"].max())],
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-block",
                                    children=[html.Label("Mes"), dcc.Dropdown(id="month-filter", options=month_options, value="all", clearable=False)],
                                ),
                                html.Div(
                                    className="control-block",
                                    children=[html.Label("Hipoteca"), dcc.Dropdown(id="housing-filter", options=yes_no_options, value="all", clearable=False)],
                                ),
                                html.Div(
                                    className="control-block",
                                    children=[html.Label("Préstamo"), dcc.Dropdown(id="loan-filter", options=yes_no_options, value="all", clearable=False)],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="kpi-grid",
                    children=[
                        html.Div(className="panel kpi", children=[html.P("Clientes filtrados", className="kpi-label"), html.P(id="kpi-clients", className="kpi-value")]),
                        html.Div(className="panel kpi", children=[html.P("Aceptación", className="kpi-label"), html.P(id="kpi-acceptance", className="kpi-value")]),
                        html.Div(className="panel kpi", children=[html.P("Duración media", className="kpi-label"), html.P(id="kpi-duration", className="kpi-value")]),
                        html.Div(className="panel kpi", children=[html.P("Balance medio", className="kpi-label"), html.P(id="kpi-balance", className="kpi-value")]),
                    ],
                ),
                html.Section(
                    className="viz-grid",
                    children=[
                        html.Div(className="panel chart-panel", children=[dcc.Graph(id="job-chart", config={"displayModeBar": False})]),
                        html.Div(className="panel chart-panel", children=[dcc.Graph(id="age-chart", config={"displayModeBar": False})]),
                        html.Div(className="panel chart-panel", children=[dcc.Graph(id="duration-chart", config={"displayModeBar": False})]),
                        html.Div(className="panel chart-panel", children=[dcc.Graph(id="corr-chart", config={"displayModeBar": False})]),
                    ],
                ),
                html.Section(
                    className="predictor-layout",
                    style={"marginTop": "18px"},
                    children=[
                        html.Div(
                            className="panel predictor-panel",
                            children=[
                                html.H2("Predicción de balance", className="section-title"),
                                html.Div(
                                    className="predictor-grid",
                                    children=[
                                        html.Div(className="predictor-field", children=[html.Label("Edad"), dcc.Input(id="p-age", type="number", value=35, min=18, max=95, step=1)]),
                                        html.Div(className="predictor-field", children=[html.Label("Ocupación"), dcc.Dropdown(id="p-job", options=job_options, value="management", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Estado civil"), dcc.Dropdown(id="p-marital", options=categorical_options("marital"), value="married", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Educación"), dcc.Dropdown(id="p-education", options=categorical_options("education"), value="secondary", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Incumplimiento"), dcc.Dropdown(id="p-default", options=categorical_options("default"), value="no", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Hipoteca"), dcc.Dropdown(id="p-housing", options=categorical_options("housing"), value="yes", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Préstamo"), dcc.Dropdown(id="p-loan", options=categorical_options("loan"), value="no", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Contacto"), dcc.Dropdown(id="p-contact", options=categorical_options("contact"), value="cellular", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Día"), dcc.Input(id="p-day", type="number", value=15, min=1, max=31, step=1)]),
                                        html.Div(className="predictor-field", children=[html.Label("Mes"), dcc.Dropdown(id="p-month", options=categorical_options("month"), value="may", clearable=False)]),
                                        html.Div(className="predictor-field", children=[html.Label("Duración"), dcc.Input(id="p-duration", type="number", value=260, min=0, step=10)]),
                                        html.Div(className="predictor-field", children=[html.Label("Contactos campaña"), dcc.Input(id="p-campaign", type="number", value=2, min=1, step=1)]),
                                        html.Div(className="predictor-field", children=[html.Label("Días previos"), dcc.Input(id="p-pdays", type="number", value=-1, step=1)]),
                                        html.Div(className="predictor-field", children=[html.Label("Contactos previos"), dcc.Input(id="p-previous", type="number", value=0, min=0, step=1)]),
                                        html.Div(className="predictor-field", children=[html.Label("Resultado previo"), dcc.Dropdown(id="p-poutcome", options=categorical_options("poutcome"), value="unknown", clearable=False)]),
                                    ],
                                ),
                                html.Button("Predecir balance", id="predict-button", n_clicks=0, className="predict-button"),
                                html.Div(id="prediction-output", className="prediction-result"),
                                html.P(
                                    f"Modelo usado: regresion compacta preentrenada. R2 test: {model_metrics['r2_test']:.3f}; MAE test: {model_metrics['mae_test']:,.0f}. "
                                    "El notebook mostró desempeño limitado para esta variable, por lo que la predicción debe interpretarse como referencia exploratoria.",
                                    className="predictor-note",
                                ),
                            ],
                        ),
                        html.Div(
                            className="panel predictor-panel",
                            children=[
                                html.H2("Desempeño del modelo de regresión", className="section-title"),
                                html.Div(
                                    className="model-metrics-grid",
                                    children=[
                                        html.Div(className="mini-metric", children=[html.P("R2 prueba"), html.Strong(f"{model_metrics['r2_test']:.3f}")]),
                                        html.Div(className="mini-metric", children=[html.P("MAE prueba"), html.Strong(f"{model_metrics['mae_test']:,.0f}")]),
                                        html.Div(className="mini-metric", children=[html.P("RMSE prueba"), html.Strong(f"{model_metrics['rmse_test']:,.0f}")]),
                                    ],
                                ),
                                html.P(
                                    "La regresión de balance se mantiene como componente exploratorio. El resultado debe interpretarse como una estimación de apoyo, no como una predicción financiera exacta.",
                                    className="predictor-note",
                                ),
                            ],
                        ),
                    ],
                ),
                html.Section(
                    className="panel chart-panel map-panel",
                    style={"marginTop": "18px"},
                    children=[
                        html.Div(
                            className="map-header",
                            children=[
                                html.Div(
                                    children=[
                                        html.H2("Mapa sociodemográfico de Panamá", className="section-title"),
                                        html.P(
                                            "Seleccione una variable del INEC MAPI. El mapa muestra el valor por distrito y la gráfica resume la misma variable por provincia.",
                                            className="predictor-note",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    className="control-block map-variable-control",
                                    children=[
                                        html.Label("Variable sociodemográfica"),
                                        dcc.Dropdown(
                                            id="map-variable-filter",
                                            options=map_variable_options,
                                            value=179,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="map-kpi-grid",
                            children=[
                                html.Div(className="map-mini-card", children=[html.P("Distritos"), html.P(id="map-kpi-districts", className="kpi-value")]),
                                html.Div(className="map-mini-card", children=[html.P("Valor máximo"), html.P(id="map-kpi-max", className="kpi-value")]),
                                html.Div(className="map-mini-card", children=[html.P("Valor mínimo"), html.P(id="map-kpi-min", className="kpi-value")]),
                                html.Div(className="map-mini-card", id="map-selected-district"),
                            ],
                        ),
                        html.Div(
                            className="map-grid",
                            children=[
                                html.Div(className="map-figure-wrap", children=[dcc.Graph(id="panama-map", figure=map_figure(179), style={"height": "590px"})]),
                                html.Div(className="summary-figure-wrap", children=[dcc.Graph(id="map-summary-chart", figure=map_summary_figure(179), config={"displayModeBar": False})]),
                            ],
                        ),
                    ],
                ),
                html.P(
                    "Fuentes: Bank Marketing Dataset de UCI Machine Learning Repository; INEC MAPI, variables sociodemográficas distritales 2023; geoBoundaries PAN ADM2 descargado para documentación geográfica del proyecto. "
                    f"geoBoundaries reporta {geob_meta.get('admUnitCount')} unidades ADM2 para su versión {geob_meta.get('boundaryYearRepresented')}.",
                    className="source-note",
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("kpi-clients", "children"),
    Output("kpi-acceptance", "children"),
    Output("kpi-duration", "children"),
    Output("kpi-balance", "children"),
    Output("job-chart", "figure"),
    Output("age-chart", "figure"),
    Output("duration-chart", "figure"),
    Output("corr-chart", "figure"),
    Input("job-filter", "value"),
    Input("age-filter", "value"),
    Input("month-filter", "value"),
    Input("housing-filter", "value"),
    Input("loan-filter", "value"),
)
def update_analysis(selected_jobs, age_range, month, housing, loan):
    df = filter_data(selected_jobs, age_range, month, housing, loan)
    if df.empty:
        return "0", "0.0%", "0 s", "0", empty_figure("Sin datos"), empty_figure("Sin datos"), empty_figure("Sin datos"), empty_figure("Sin datos")

    acceptance = (df["y"] == "yes").mean() * 100
    duration = df["duration"].mean()
    balance = df["balance"].mean()
    return (
        f"{len(df):,}",
        f"{acceptance:.1f}%",
        f"{duration:,.0f} s",
        f"{balance:,.0f}",
        acceptance_by_job_figure(df),
        age_distribution_figure(df),
        duration_box_figure(df),
        correlation_figure(df),
    )


@app.callback(
    Output("panama-map", "figure"),
    Output("map-summary-chart", "figure"),
    Output("map-kpi-districts", "children"),
    Output("map-kpi-max", "children"),
    Output("map-kpi-min", "children"),
    Output("map-selected-district", "children"),
    Input("map-variable-filter", "value"),
    Input("panama-map", "clickData"),
)
def update_map_section(variable_id, click_data):
    variable_id = int(variable_id or 179)
    _, district_table, _, variable = load_map_data(variable_id)
    unit = variable.get("unidad_medida") or ""
    max_row = district_table.sort_values("value", ascending=False).iloc[0]
    min_row = district_table.sort_values("value", ascending=True).iloc[0]
    return (
        map_figure(variable_id),
        map_summary_figure(variable_id),
        f"{len(district_table):,}",
        format_map_value(float(max_row["value"]), unit),
        format_map_value(float(min_row["value"]), unit),
        selected_district_text(click_data, variable_id),
    )


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    Input("p-age", "value"),
    Input("p-job", "value"),
    Input("p-marital", "value"),
    Input("p-education", "value"),
    Input("p-default", "value"),
    Input("p-housing", "value"),
    Input("p-loan", "value"),
    Input("p-contact", "value"),
    Input("p-day", "value"),
    Input("p-month", "value"),
    Input("p-duration", "value"),
    Input("p-campaign", "value"),
    Input("p-pdays", "value"),
    Input("p-previous", "value"),
    Input("p-poutcome", "value"),
)
def predict_balance(n_clicks, age, job, marital, education, default, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome):
    if any(value is None for value in [age, day, duration, campaign, pdays, previous]):
        return "Complete los valores numericos para calcular el balance."

    instance = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }
    try:
        model, _ = load_regression_model()
        prediction = predict_regression_model(model, instance)
    except Exception:
        logger.exception("No se pudo calcular la prediccion de balance")
        return "No se pudo calcular el balance en este momento. Revise los logs del servidor."

    prefix = "Predicción inicial" if not n_clicks else "Balance estimado"
    return f"{prefix}: {prediction:,.0f}"


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
