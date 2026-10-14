"""
Hispanic Heritage Historical Data Explorer
Streamlit app: World Bank historical data, polynomial fitting, context/function lab,
calculus analysis, editable data, and CSV/XLSX/PDF exports.
"""
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
WORLD_BANK_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
COUNTRIES = {
    "Mexico (MX)": "MX", "Brazil (BR)": "BR", "Argentina (AR)": "AR",
    "Colombia (CO)": "CO", "Chile (CL)": "CL", "Peru (PE)": "PE",
    "Venezuela (VE)": "VE", "Ecuador (EC)": "EC", "Uruguay (UY)": "UY",
}
INDICATORS = {
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN", "unit": "years",
        "definition": "The number of years a newborn would live if current age-specific mortality patterns persisted.",
        "constraint": "positive", "source": "https://data.worldbank.org/indicator/SP.DYN.LE00.IN",
    },
    "Unemployment, national estimate": {
        "code": "SL.UEM.TOTL.NE.ZS", "unit": "% of labor force",
        "definition": "The share of the labor force that is without work but available for and seeking employment, using national estimates.",
        "constraint": "percent", "source": "https://data.worldbank.org/indicator/SL.UEM.TOTL.NE.ZS",
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD", "unit": "current US$ per person",
        "definition": "Gross domestic product divided by midyear population, in current US dollars. It is not average wealth or household income.",
        "constraint": "positive", "source": "https://data.worldbank.org/indicator/NY.GDP.PCAP.CD",
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN", "unit": "births per 1,000 people per year",
        "definition": "Annual live births per 1,000 people. This is a rate, not a birth count.",
        "constraint": "positive", "source": "https://data.worldbank.org/indicator/SP.DYN.CBRT.IN",
    },
    "Net migration": {
        "code": "SM.POP.NETM", "unit": "people over the period reported",
        "definition": "The net number of migrants during the period. It is not an emigration count; positive and negative values are possible.",
        "constraint": "any", "source": "https://data.worldbank.org/indicator/SM.POP.NETM",
    },
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5", "unit": "per 100,000 people",
        "definition": "Intentional homicide victims per 100,000 people, as reported by the World Bank indicator source.",
        "constraint": "nonnegative", "source": "https://data.worldbank.org/indicator/VC.IHR.PSRC.P5",
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG", "unit": "% annual change",
        "definition": "Annual percentage change in the consumer price index. Negative values can occur.",
        "constraint": "any", "source": "https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG",
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS", "unit": "% of GDP",
        "definition": "Central government debt expressed as a percentage of GDP.",
        "constraint": "nonnegative", "source": "https://data.worldbank.org/indicator/GC.DOD.TOTL.GD.ZS",
    },
    "Total reserves including gold": {
        "code": "FI.RES.TOTL.CD", "unit": "current US$",
        "definition": "Total reserves including gold, valued in current US dollars.",
        "constraint": "nonnegative", "source": "https://data.worldbank.org/indicator/FI.RES.TOTL.CD",
    },
    "External balance on goods and services": {
        "code": "NE.RSB.GNFS.ZS", "unit": "% of GDP",
        "definition": "Exports minus imports of goods and services, expressed as a share of GDP.",
        "constraint": "any", "source": "https://data.worldbank.org/indicator/NE.RSB.GNFS.ZS",
    },
}
CONTEXTS = {
    "Business": ("Revenue over time", "time", "Revenue", "year", "USD", "Examine how revenue changes over time."),
    "Engineering": ("Bridge deflection under load", "load", "deflection", "kN", "mm", "Model a measured engineering response; use real measurements for a real design."),
    "Medical": ("Measured response over time", "time", "response", "days", "measurement units", "Illustrative only; not for diagnosis or treatment."),
    "Sports science": ("Athlete performance versus training time", "training time", "performance", "hours", "performance units", "Illustrative relationship; avoid treating a fit as a guarantee."),
    "Biology": ("Population or growth over time", "time", "population", "days", "organisms", "Use a biological mechanism and data appropriate to the species."),
    "Bioengineering": ("Biomaterial response versus strain", "strain", "response", "%", "response units", "Illustrative model; validate with laboratory measurements."),
    "Economic forecasting": ("Economic indicator over time", "year", "indicator", "year", "indicator units", "Forecasts are uncertain; polynomial extrapolation can be unstable."),
    "Weather": ("Temperature over time", "time", "temperature", "day", "°C", "Illustrative function only; not a weather forecast."),
    "Custom": ("Custom model", "x", "y", "x units", "y units", "Describe the system and assumptions."),
}
SUGGESTED = {
    "Business": "a*x + b",
    "Engineering": "a*x**2 + b*x + c",
    "Medical": "a*exp(-b*x) + c",
    "Sports science": "a*sin(b*x) + c",
    "Biology": "a*exp(b*x)",
    "Bioengineering": "a*x/(b + x)",
    "Economic forecasting": "a*x**2 + b*x + c",
    "Weather": "a*sin(2*pi*x/365) + c",
    "Custom": "a*x + b",
}
SAFE_FUNCTIONS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi,
}

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("HISPANIC HERITAGE DATA LAB • historical evidence + mathematical modeling")
st.info("Research question and project choices can be customized below. Brazil is included as a Latin American comparison country; it is not a Spanish-speaking country.")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country: str, indicator: str, start_year: int, end_year: int) -> tuple[pd.DataFrame, str]:
    """Return source observations and a readable retrieval status."""
    url = WORLD_BANK_API.format(country=country, indicator=indicator)
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": 20000}
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this selection."
        rows = []
        for item in payload[1]:
            year = item.get("date")
            value = item.get("value")
            if year is not None and value is not None:
                try:
                    rows.append({"year": int(year), "value": float(value)})
                except (TypeError, ValueError):
                    continue
        frame = pd.DataFrame(rows, columns=["year", "value"]).sort_values("year") if rows else pd.DataFrame(columns=["year", "value"])
        if frame.empty:
            return frame, "The API response contained no non-null year/value observations."
        return frame.reset_index(drop=True), f"Retrieved {len(frame)} non-null observations."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Check your connection and retry."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not read the API response: {exc}"

def validate_table(frame: pd.DataFrame, constraint: str, min_year: int, max_year: int) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    if not {"year", "value"}.issubset(frame.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    data = frame[["year", "value"]].copy()
    numeric_year = pd.to_numeric(data["year"], errors="coerce")
    numeric_value = pd.to_numeric(data["value"], errors="coerce")
    if numeric_year.isna().any():
        issues.append("Every year must be an integer.")
    if numeric_year.notna().any() and not np.equal(numeric_year.dropna(), np.floor(numeric_year.dropna())).all():
        issues.append("Years must be whole numbers.")
    data["year"] = numeric_year
    data["value"] = numeric_value
    if data["year"].duplicated().any():
        issues.append("Duplicate years found. Keep one row per year.")
    if data["value"].isna().any():
        issues.append("Values must be numeric; blank values are not valid in the editable table.")
    if data["year"].notna().any() and ((data["year"] < min_year) | (data["year"] > max_year)).any():
        issues.append(f"Years must be between {min_year} and {max_year}.")
    vals = data["value"].dropna()
    if constraint in ("positive", "percent") and (vals < 0).any():
        issues.append("This indicator should not contain negative values.")
    if constraint == "nonnegative" and (vals < 0).any():
        issues.append("This indicator should not contain negative values.")
    if len(data) < 1:
        issues.append("At least one valid observation is required.")
    cleaned = data.dropna(subset=["year", "value"]).copy()
    if not cleaned.empty:
        cleaned["year"] = cleaned["year"].astype(int)
        cleaned = cleaned.sort_values("year").reset_index(drop=True)
    return cleaned, issues

def polynomial_fit(data: pd.DataFrame, degree: int) -> dict[str, Any]:
    if data["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {data['year'].nunique()} are available.")
    base_year = int(data["year"].min())
    t = data["year"].to_numpy(dtype=float) - base_year
    y = data["value"].to_numpy(dtype=float)
    if not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError("The fit contains non-finite values.")
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    residuals = y - pred
    return {"base": base_year, "coeff": coeff, "degree": degree, "pred": pred, "r2": r2, "mae": mae,
            "residuals": residuals, "n": len(y), "min_year": int(data["year"].min()),
            "max_year": int(data["year"].max()), "mean": float(np.mean(y))}

def poly_predict(model: dict[str, Any], years: Any) -> np.ndarray:
    # Apply the centered-time transformation exactly once.
    return np.polyval(model["coeff"], np.asarray(years, dtype=float) - model["base"])

def equation_text(model: dict[str, Any], y_name: str = "y") -> str:
    terms = []
    degree = model["degree"]
    for i, c in enumerate(model["coeff"]):
        power = degree - i
        if abs(c) < 1e-12:
            continue
        if power == 0:
            term = f"{c:.10g}"
        elif power == 1:
            term = f"{c:.10g}t"
        else:
            term = f"{c:.10g}t^{power}"
        terms.append(term)
    return f"{y_name} = " + (" + ".join(terms).replace("+ -", "- ") if terms else "0") + f", where t = year - {model['base']}"

def holdout_score(data: pd.DataFrame, degree: int) -> str:
    # Chronological holdout: train on earliest ~80%, test on latest observations.
    ordered = data.sort_values("year").reset_index(drop=True)
    n = len(ordered)
    test_n = max(1, int(math.ceil(n * 0.2)))
    train = ordered.iloc[:-test_n]
    test = ordered.iloc[-test_n:]
    if len(train) < degree + 1 or len(test) < 2:
        return "Not enough data for a meaningful chronological holdout (needs degree+1 training years and at least 2 test years)."
    base = int(train["year"].min())
    coeff = np.polyfit(train["year"].to_numpy() - base, train["value"].to_numpy(), degree)
    predicted = np.polyval(coeff, test["year"].to_numpy() - base)
    return f"Chronological holdout MAE: {mean_absolute_error(test['value'], predicted):.6g} {st.session_state.get('unit_label','units')} across {len(test)} later years. Separate from in-sample metrics."

def safe_custom_function(expression: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Evaluate restricted familiar notation without eval or arbitrary Python."""
    expr = expression.replace("^", "**")
    if not re.fullmatch(r"[0-9a-zA-Z_+\-*/().,\s]*", expr):
        raise ValueError("Use only numbers, x, parameter names, parentheses, + - * /, powers, and supported functions.")
    names = {"x": x, **params, **SAFE_FUNCTIONS}
    # Permit only explicitly listed names; reject attribute access, indexing, and unknown identifiers.
    identifiers = set(re.findall(r"[A-Za-z_]\w*", expr))
    allowed = set(names)
    unknown = identifiers - allowed
    if unknown:
        raise ValueError("Unknown name(s): " + ", ".join(sorted(unknown)))
    try:
        result = eval(expr, {"__builtins__": {}}, names)  # restricted globals and validated identifiers
        arr = np.asarray(result, dtype=float)
        if arr.ndim == 0:
            arr = np.full_like(x, float(arr), dtype=float)
        if arr.shape != x.shape:
            arr = np.broadcast_to(arr, x.shape).astype(float)
        return arr
    except Exception as exc:
        raise ValueError(f"Could not evaluate function: {exc}") from exc

def make_excel(summary: pd.DataFrame, data: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        data.to_excel(writer, sheet_name="Working data", index=False)
        results.to_excel(writer, sheet_name="Calculated years", index=False)
        for sheet, df in [("Summary", summary), ("Working data", data), ("Calculated years", results)]:
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 0)
            ws.set_column(0, max(0, len(df.columns)-1), 22)
    return buffer.getvalue()

def make_pdf(title: str, summary_lines: list[str], chart_bytes: bytes | None, data: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch, leftMargin=0.45*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in summary_lines:
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    if chart_bytes:
        story.extend([Spacer(1, 8), RLImage(io.BytesIO(chart_bytes), width=8.7*inch, height=4.0*inch), Spacer(1, 8)])
    for label, df in [("Data preview", data.head(20)), ("Calculated-year results", results.head(30))]:
        story.append(Paragraph(label, styles["Heading2"]))
        if df.empty:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
        else:
            rows = [list(map(str, df.columns))] + df.astype(str).values.tolist()
            table = Table(rows, repeatRows=1)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2d35e")),
                ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(table)
        story.append(Spacer(1, 10))
    doc.build(story)
    return buffer.getvalue()

# Project setup and research controls
with st.sidebar:
    st.header("Project setup")
    research_question = st.text_input("Research question", "How has life expectancy changed over time in selected Latin American countries?")
    pathway = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"])
    country_label = st.selectbox("Featured country", list(COUNTRIES.keys()), index=0)
    country_code = COUNTRIES[country_label]
    indicator_name = st.selectbox("Indicator", list(INDICATORS.keys()), index=0)
    indicator = INDICATORS[indicator_name]
    degree = st.slider("Polynomial degree", min_value=3, max_value=8, value=3)
    sample_step = st.slider("Graph sample increment (years)", min_value=1, max_value=10, value=5)
    horizon = st.slider("Forward display horizon (years)", min_value=0, max_value=50, value=10)
    comparison_labels = st.multiselect("Comparison countries (same indicator)", [x for x in COUNTRIES if x != country_label], default=["Brazil (BR)"] if country_label != "Brazil (BR)" else ["Mexico (MX)"])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

current_year = date.today().year
start_year, end_year = current_year - 70, current_year - 1
st.write(f"**Research question:** {research_question}")
st.write(f"**Historical window requested:** {start_year}–{end_year} inclusive (70 calendar years). Actual available observations may be fewer.")
st.caption(f"Source: World Bank API • retrieved on {date.today().isoformat()} • {indicator['code']}")
st.markdown(f"**Definition:** {indicator['definition']}  \n**Unit:** {indicator['unit']}  \n**Source link:** [{indicator['source']}]({indicator['source']})")

source_df, fetch_status = fetch_world_bank(country_code, indicator["code"], start_year, end_year)
st.caption(fetch_status)
if source_df.empty:
    st.error("No source observations are available for this selection. The app will not invent or substitute data.")
    source_df = pd.DataFrame(columns=["year", "value"])

data_key = f"source_{country_code}_{indicator['code']}_{start_year}_{end_year}"
if data_key not in st.session_state:
    st.session_state[data_key] = source_df.copy()
if "active_data_key" not in st.session_state or st.session_state.active_data_key != data_key:
    st.session_state.active_data_key = data_key
    st.session_state.working_data = source_df.copy()
    st.session_state.edited = False
if "calc_results" not in st.session_state:
    st.session_state.calc_results = pd.DataFrame(columns=["year", "estimate", "unit", "range_classification"])
if "calc_signature" not in st.session_state:
    st.session_state.calc_signature = None

col_a, col_b, col_c = st.columns(3)
col_a.metric("Non-null source observations", len(source_df))
col_b.metric("Distinct source years", source_df["year"].nunique() if not source_df.empty else 0)
col_c.metric("Coverage", f"{int(source_df.year.min())}–{int(source_df.year.max())}" if not source_df.empty else "None")

st.subheader("Working data (editable)")
st.caption("Edit the year/value table below. Original source data remains separate. Edited values are a user-modified scenario, not official observations.")
if st.button("Reset to source"):
    st.session_state.working_data = st.session_state[data_key].copy()
    st.session_state.edited = False
    st.session_state.calc_results = pd.DataFrame(columns=["year", "estimate", "unit", "range_classification"])
    st.session_state.calc_signature = None
edited_frame = st.data_editor(st.session_state.working_data, num_rows="dynamic", use_container_width=True, key=f"editor_{data_key}")
if not edited_frame.equals(st.session_state.working_data):
    st.session_state.edited = True
st.session_state.working_data = edited_frame.copy()
working_data, validation_issues = validate_table(st.session_state.working_data, indicator["constraint"], start_year, end_year)
if st.session_state.edited:
    st.warning("USER-MODIFIED SCENARIO: calculations and exports use the edited table until Reset to source is selected.")
if validation_issues:
    for issue in validation_issues:
        st.error(issue)
else:
    st.success(f"Working table valid: {len(working_data)} observations.")

model = None
fit_error = None
if not validation_issues and len(working_data) >= degree + 1:
    try:
        model = polynomial_fit(working_data, degree)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        fit_error = str(exc)
elif not validation_issues:
    fit_error = f"Degree {degree} requires at least {degree + 1} distinct years; current valid count is {working_data['year'].nunique()}."
if fit_error:
    st.warning(f"Model not fitted: {fit_error}")

results = st.session_state.calc_results.copy()
signature = (country_code, indicator["code"], degree, tuple(working_data.get("year", pd.Series(dtype=int)).tolist()), tuple(working_data.get("value", pd.Series(dtype=float)).tolist()))
if st.session_state.calc_signature is not None and signature != st.session_state.calc_signature:
    st.session_state.calc_results = pd.DataFrame(columns=["year", "estimate", "unit", "range_classification"])
    results = st.session_state.calc_results.copy()
    st.info("Calculated-year results were cleared because the underlying selection or data changed.")

if model is not None:
    st.subheader("Polynomial fit")
    st.write(f"**Equation:** `{equation_text(model, indicator_name)}`")
    st.caption(f"Centered time variable: t = year − {model['base']}. The same centered-time transformation is used for fitting and prediction.")
    met1, met2, met3, met4 = st.columns(4)
    met1.metric("R² (in-sample)", f"{model['r2']:.5f}")
    met2.metric("MAE (in-sample)", f"{model['mae']:.5g} {indicator['unit']}")
    met3.metric("Observations", model["n"])
    met4.metric("Mean", f"{model['mean']:.6g} {indicator['unit']}")
    st.write(f"**Fitted historical range:** {model['min_year']}–{model['max_year']}  • **Degree:** {degree}")
    st.write(holdout_score(working_data, degree))
    residual_table = working_data[["year", "value"]].copy()
    residual_table["fitted_value"] = model["pred"]
    residual_table["residual (actual - fitted)"] = model["residuals"]
    with st.expander("Residuals and fitted values"):
        st.dataframe(residual_table, use_container_width=True)
    st.caption("R² and MAE are in-sample fit metrics, not proof of causation. MAE and residuals describe historical fit error; no validated 95% prediction interval is claimed.")

    # Graph
    st.subheader("Historical data and fitted curve")
    max_display_year = min(end_year + horizon, 2100)
    curve_years = np.linspace(model["min_year"], max_display_year, max(250, (max_display_year-model["min_year"]+1)*3))
    curve_values = poly_predict(model, curve_years)
    sample_years = np.arange(model["min_year"], max_display_year + 1, sample_step)
    sample_values = poly_predict(model, sample_years)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.scatter(working_data["year"], working_data["value"], label="Observed / working data", color="#1f77b4", zorder=3)
    hist_mask = curve_years <= model["max_year"]
    ext_mask = curve_years > model["max_year"]
    ax.plot(curve_years[hist_mask], curve_values[hist_mask], label="Fitted curve (historical range)", color="#d97706", linewidth=2)
    if ext_mask.any():
        ax.plot(curve_years[ext_mask], curve_values[ext_mask], label="Extrapolated segment (uncertain)", color="#b91c1c", linestyle="--", linewidth=2)
    ax.scatter(sample_years, sample_values, marker="x", s=20, label=f"Model sample every {sample_step} year(s)", color="#333333")
    ax.axvline(model["max_year"], color="gray", linestyle=":", label="Historical boundary")
    if not results.empty:
        ax.scatter(results["year"], results["estimate"], marker="D", s=55, label="Calculated estimates", color="#7c3aed")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{indicator_name} ({indicator['unit']})")
    ax.set_title(f"{country_label}: {indicator_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", dpi=160, bbox_inches="tight")
    chart_bytes = chart_buf.getvalue()
    plt.close(fig)

    # Flag potentially implausible extrapolations; do not clip.
    if horizon > 0:
        future_years = np.arange(model["max_year"] + 1, max_display_year + 1)
        future_vals = poly_predict(model, future_years)
        if len(future_vals) and (not np.isfinite(future_vals).all() or np.max(np.abs(future_vals)) > max(1e6, 1000 * max(1, np.nanmax(np.abs(working_data["value"].to_numpy()))))):
            st.warning("Extrapolation warning: the polynomial produces extreme or non-finite values. Values are not clipped; do not treat them as reliable forecasts.")

    # Single and batch calculation
    st.subheader("Calculate model estimates")
    st.caption("These are fitted model estimates, not necessarily actual observed values. At years with source data, the estimate is labeled separately from the observation.")
    with st.form("single_year_form"):
        single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=min(max(model["max_year"], 1950), 2100), step=1)
        calc_single = st.form_submit_button("Calculate Value")
    if calc_single:
        est = float(poly_predict(model, np.array([single_year]))[0])
        in_data = bool((working_data["year"] == int(single_year)).any())
        classification = "within fitted historical range" if model["min_year"] <= single_year <= model["max_year"] else "outside fitted historical range"
        row = pd.DataFrame([{"year": int(single_year), "estimate": est, "unit": indicator["unit"], "range_classification": classification, "actual_observation_exists": in_data}])
        st.session_state.calc_results = pd.concat([st.session_state.calc_results, row], ignore_index=True).drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)
        st.session_state.calc_signature = signature
    with st.form("batch_year_form"):
        batch_mode = st.radio("Batch input type", ["Year range", "Custom comma-separated years"], horizontal=True)
        if batch_mode == "Year range":
            r1, r2, r3 = st.columns(3)
            batch_start = r1.number_input("Start year", 1950, 2100, int(max(1950, model["min_year"])), 1)
            batch_end = r2.number_input("End year", 1950, 2100, int(min(2100, model["max_year"])), 1)
            batch_step = r3.slider("Range step (years)", 1, 10, 1)
            custom_text = ""
        else:
            custom_text = st.text_input("Custom years, e.g. 1950, 1980, 2025")
            batch_start, batch_end, batch_step = 1950, 1950, 1
        calc_batch = st.form_submit_button("Calculate batch")
    if calc_batch:
        try:
            if batch_mode == "Year range":
                if batch_end < batch_start:
                    raise ValueError("End year must be greater than or equal to start year.")
                requested = list(range(int(batch_start), int(batch_end)+1, int(batch_step)))
            else:
                if not custom_text.strip():
                    raise ValueError("Enter one or more years separated by commas.")
                tokens = [part.strip() for part in custom_text.split(",")]
                if any(not re.fullmatch(r"\d{4}", token) for token in tokens):
                    raise ValueError("Each custom year must be a four-digit integer, separated by commas.")
                requested = [int(token) for token in tokens]
                if any(y < 1950 or y > 2100 for y in requested):
                    raise ValueError("Custom years must be between 1950 and 2100.")
            requested = list(dict.fromkeys(requested))
            estimates = poly_predict(model, np.array(requested, dtype=float))
            actual_years = set(working_data["year"].tolist())
            batch = pd.DataFrame({
                "year": requested, "estimate": estimates, "unit": indicator["unit"],
                "range_classification": ["within fitted historical range" if model["min_year"] <= y <= model["max_year"] else "outside fitted historical range" for y in requested],
                "actual_observation_exists": [y in actual_years for y in requested],
            })
            st.session_state.calc_results = pd.concat([st.session_state.calc_results, batch], ignore_index=True).drop_duplicates(subset=["year"], keep="last").sort_values("year").reset_index(drop=True)
            st.session_state.calc_signature = signature
            st.success(f"Calculated {len(requested)} valid requested year(s).")
        except ValueError as exc:
            st.error(f"Invalid batch input: {exc}")
    results = st.session_state.calc_results.copy()
    if not results.empty:
        results["actual_observation"] = results["year"].map(working_data.set_index("year")["value"])
        st.dataframe(results, use_container_width=True)
    else:
        st.info("No calculated-year results yet. Use either calculation form above.")

    # Pathway-specific mathematical analysis
    st.subheader(f"{pathway} analysis")
    years_dense = np.linspace(model["min_year"], model["max_year"], 500)
    vals_dense = poly_predict(model, years_dense)
    avg_rate = (float(poly_predict(model, np.array([model["max_year"]]))[0]) - float(poly_predict(model, np.array([model["min_year"]]))[0])) / max(1, model["max_year"] - model["min_year"])
    st.write(f"**Domain used for historical fit:** years {model['min_year']}–{model['max_year']}. **Observed-data range:** {working_data['value'].min():.6g} to {working_data['value'].max():.6g} {indicator['unit']}.")
    st.write(f"**Average rate of change across the fitted historical interval:** {avg_rate:.6g} {indicator['unit']} per year.")
    st.write("**End behavior:** a polynomial's long-run behavior is controlled by its leading term. Outside the observed range, high-degree polynomials can grow rapidly and may not describe real-world limits.")
    if pathway in ("Precalculus", "Calculus"):
        st.write(f"**Degree and coefficients:** degree {degree} allows up to {degree-1} turning points. Coefficient signs and magnitudes affect shape and scale; changing them can alter turning points and concavity.")
        first_coeff = np.polyder(model["coeff"], 1)
        second_coeff = np.polyder(model["coeff"], 2)
        critical = np.roots(first_coeff) if len(first_coeff) > 1 else np.array([])
        real_critical = sorted([float(z.real + model["base"]) for z in critical if abs(z.imag) < 1e-8 and model["min_year"] <= z.real + model["base"] <= model["max_year"]])
        st.write("**Turning-point candidates in the historical interval:** " + (", ".join(f"{x:.3f}" for x in real_critical) if real_critical else "none detected"))
        st.write("Changing average rates across subintervals helps reveal concavity: increasing slopes suggest concave up; decreasing slopes suggest concave down.")
    if pathway == "Calculus":
        d1 = np.polyder(model["coeff"], 1)
        d2 = np.polyder(model["coeff"], 2)
        st.write(f"**First derivative:** d({indicator_name})/dt = `{np.poly1d(d1)}`")
        st.write(f"**Second derivative:** d²({indicator_name})/dt² = `{np.poly1d(d2)}`")
        if len(d1) > 1:
            crit_t = np.roots(d1)
            crit_years = sorted([float(z.real + model["base"]) for z in crit_t if abs(z.imag) < 1e-8 and model["min_year"] <= z.real + model["base"] <= model["max_year"]])
        else:
            crit_years = []
        if len(d2) > 1:
            inf_t = np.roots(d2)
            inf_years = sorted([float(z.real + model["base"]) for z in inf_t if abs(z.imag) < 1e-8 and model["min_year"] <= z.real + model["base"] <= model["max_year"]])
        else:
            inf_years = []
        st.write("**Critical points in interval:** " + (", ".join(f"{y:.3f}" for y in crit_years) if crit_years else "none"))
        st.write("**Inflection-point candidates in interval:** " + (", ".join(f"{y:.3f}" for y in inf_years) if inf_years else "none"))
        st.caption("Critical points are classified using derivative sign changes; endpoints must also be checked for interval extrema. Inflection candidates require a concavity sign change.")
        # Numerically classify critical points and interval extrema
        candidates = [model["min_year"], model["max_year"]] + crit_years
        candidate_vals = poly_predict(model, np.array(candidates))
        st.write(f"**Interval minimum among endpoints and critical points:** year {candidates[int(np.argmin(candidate_vals))]:.3f}, estimate {np.min(candidate_vals):.6g}.")
        st.write(f"**Interval maximum among endpoints and critical points:** year {candidates[int(np.argmax(candidate_vals))]:.3f}, estimate {np.max(candidate_vals):.6g}.")
else:
    chart_bytes = None
    st.info("A valid polynomial model is required for calculations, graphing, and model exports.")

# Country comparison
st.subheader("Same-indicator country comparison")
if comparison_labels:
    compare_frames = []
    for label in comparison_labels:
        code = COUNTRIES[label]
        cdf, cstatus = fetch_world_bank(code, indicator["code"], start_year, end_year)
        if not cdf.empty:
            cdf["country"] = label
            compare_frames.append(cdf)
        else:
            st.caption(f"{label}: {cstatus}")
    if compare_frames:
        comp = pd.concat(compare_frames, ignore_index=True)
        figc, axc = plt.subplots(figsize=(10, 4))
        if not source_df.empty:
            axc.plot(source_df["year"], source_df["value"], marker="o", markersize=3, label=country_label)
        for label, group in comp.groupby("country"):
            axc.plot(group["year"], group["value"], marker="o", markersize=3, label=label)
        axc.set_xlabel("Year")
        axc.set_ylabel(f"{indicator_name} ({indicator['unit']})")
        axc.set_title(f"Official source series: {indicator_name}")
        axc.grid(True, alpha=0.25)
        axc.legend()
        figc.tight_layout()
        st.pyplot(figc, use_container_width=True)
        st.caption("Countries can have different observation coverage or missing years. A gap means no reported value in this API series, not zero.")
        plt.close(figc)
    else:
        st.info("No comparison observations were returned for the selected countries.")
else:
    st.caption("Choose one or more comparison countries in the sidebar.")

# Context and function lab
st.header("Context and function simulation lab")
st.write("This section is a mathematical sandbox. Unless you select the fitted historical model, the function is a user-defined or illustrative model—not official historical data.")
context = st.selectbox("Context", list(CONTEXTS.keys()))
suggested_name, default_x, default_y, default_x_unit, default_y_unit, context_note = CONTEXTS[context]
context_desc = st.text_input("Context description", context_note)
v1, v2, v3, v4 = st.columns(4)
x_name = v1.text_input("Independent variable", default_x)
y_name = v2.text_input("Dependent variable", default_y)
x_unit = v3.text_input("Independent-variable unit", default_x_unit)
y_unit = v4.text_input("Dependent-variable unit", default_y_unit)
analysis_cols = st.columns(2)
x_min = analysis_cols[0].number_input("Analysis interval start", value=0.0)
x_max = analysis_cols[1].number_input("Analysis interval end", value=10.0)
function_source = st.radio("Function source", ["Use fitted curve from historical data", "Use suggested function", "Build with parameters", "Enter my own function"], horizontal=False)
suggested_expr = SUGGESTED[context]
st.caption(f"Suggested for {context}: `{suggested_expr}`")
params = {}
if function_source == "Use fitted curve from historical data":
    st.write("Uses the polynomial fit above, with x interpreted as calendar year.")
elif function_source == "Use suggested function":
    expression = suggested_expr
    params = {"a": 1.0, "b": 0.2, "c": 0.0}
elif function_source == "Build with parameters":
    expression = st.selectbox("Function family", ["Polynomial: a*x**2 + b*x + c", "Rational: a*x/(b + x)", "Exponential: a*exp(b*x) + c", "Logarithmic: a*log(x) + b", "Trigonometric: a*sin(b*x) + c"])
    expression = {
        "Polynomial: a*x**2 + b*x + c": "a*x**2 + b*x + c",
        "Rational: a*x/(b + x)": "a*x/(b + x)",
        "Exponential: a*exp(b*x) + c": "a*exp(b*x) + c",
        "Logarithmic: a*log(x) + b": "a*log(x) + b",
        "Trigonometric: a*sin(b*x) + c": "a*sin(b*x) + c",
    }[expression]
    pa, pb, pc = st.columns(3)
    params["a"] = pa.number_input("Parameter a", value=1.0)
    params["b"] = pb.number_input("Parameter b", value=1.0)
    params["c"] = pc.number_input("Parameter c", value=0.0)
elif function_source == "Enter my own function":
    expression = st.text_input("Function f(x) =", "a*x + b")
    pa, pb, pc = st.columns(3)
    params["a"] = pa.number_input("Parameter a", value=1.0, key="custom_a")
    params["b"] = pb.number_input("Parameter b", value=0.0, key="custom_b")
    params["c"] = pc.number_input("Parameter c", value=0.0, key="custom_c")

if x_max <= x_min:
    st.error("Analysis interval end must be greater than its start.")
else:
    xs = np.linspace(x_min, x_max, 500)
    try:
        if function_source == "Use fitted curve from historical data":
            if model is None:
                raise ValueError("A valid historical polynomial fit is not available. Fix data or choose another function source.")
            ys = poly_predict(model, xs)
            sim_label = "Fitted historical polynomial (x = calendar year)"
        else:
            ys = safe_custom_function(expression, xs, params)
            sim_label = expression
        if not np.isfinite(ys).any():
            raise ValueError("Function has no finite values on this interval.")
        ys = np.where(np.isfinite(ys), ys, np.nan)
        figf, axf = plt.subplots(figsize=(10, 4.4))
        axf.plot(xs, ys, label=sim_label)
        axf.set_xlabel(f"{x_name} ({x_unit})")
        axf.set_ylabel(f"{y_name} ({y_unit})")
        axf.set_title(f"{context}: {context_desc}")
        axf.grid(True, alpha=0.25)
        axf.legend()
        figf.tight_layout()
        st.pyplot(figf, use_container_width=True)
        plt.close(figf)
        st.write(f"**Simulation range:** x from {x_min:g} to {x_max:g} {x_unit}; finite y values range from {np.nanmin(ys):.6g} to {np.nanmax(ys):.6g} {y_unit}.")
        st.write("**Interpretation reminder:** function shape describes the entered mathematical model. It does not establish a real-world causal relationship.")
        if pathway == "Calculus":
            if function_source == "Use fitted curve from historical data":
                coeff_for_calc = model["coeff"]
                deriv1 = np.polyder(coeff_for_calc)
                deriv2 = np.polyder(coeff_for_calc, 2)
                st.write(f"**Derivative with respect to centered time:** `{np.poly1d(deriv1)}`")
                st.write(f"**Second derivative:** `{np.poly1d(deriv2)}`")
            else:
                st.caption("For custom rational, exponential, logarithmic, or trigonometric functions, the graph and numerical range are shown; symbolic derivatives are not generated automatically.")
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError) as exc:
        st.error(f"Function validation/evaluation error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected function error: {exc}")

# Printer view
if printer_mode:
    st.subheader("Printer-friendly report")
    st.write(f"**Question:** {research_question}")
    st.write(f"**Selection:** {country_label} • {indicator_name} • {indicator['code']} • {indicator['unit']}")
    st.write(f"**Retrieved:** {date.today().isoformat()} • **Data status:** {'user-modified scenario' if st.session_state.edited else 'source data'}")
    if model is not None:
        st.write(f"**Model:** {equation_text(model, indicator_name)}")
        st.write(f"**In-sample R²:** {model['r2']:.6f}; **in-sample MAE:** {model['mae']:.6g}; **n:** {model['n']}.")
    st.write("**Limitations:** missing years remain missing; polynomial fit is descriptive, not causal; extrapolations are uncertain; no 95% prediction interval is claimed.")

# Exports
st.header("Export current selection")
export_data = working_data.copy() if not validation_issues else st.session_state.working_data.copy()
export_data["data_status"] = "user-modified scenario" if st.session_state.edited else "World Bank source data"
export_results = st.session_state.calc_results.copy()
summary_rows = [
    ("App", APP_TITLE), ("Research question", research_question), ("Country", country_label),
    ("Indicator", indicator_name), ("Indicator code", indicator["code"]), ("Unit", indicator["unit"]),
    ("Definition", indicator["definition"]), ("Source URL", indicator["source"]),
    ("Retrieval date", date.today().isoformat()), ("Requested window", f"{start_year}-{end_year}"),
    ("Actual non-null observations", len(source_df)), ("Data status", "user-modified scenario" if st.session_state.edited else "source data"),
    ("Pathway", pathway), ("Polynomial degree", degree), ("Sample increment (years)", sample_step), ("Forward horizon (years)", horizon),
]
if model is not None:
    summary_rows.extend([("Equation", equation_text(model, indicator_name)), ("Base year", model["base"]),
                         ("In-sample R-squared", model["r2"]), ("In-sample MAE", model["mae"]),
                         ("Observation count", model["n"]), ("Historical range", f"{model['min_year']}-{model['max_year']}"),
                         ("Mean", model["mean"]), ("Holdout", holdout_score(working_data, degree))])
summary_df = pd.DataFrame(summary_rows, columns=["field", "value"])
csv_bytes = export_data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", data=csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
try:
    xlsx_bytes = make_excel(summary_df, export_data, export_results)
    st.download_button("Download Excel workbook (.xlsx)", data=xlsx_bytes, file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as exc:
    st.error(f"Excel export failed: {exc}")
try:
    pdf_chart = chart_bytes if model is not None else None
    pdf_bytes = make_pdf(APP_TITLE, [f"{k}: {v}" for k, v in summary_rows], pdf_chart, export_data, export_results)
    st.download_button("Download PDF report", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
except Exception as exc:
    st.error(f"PDF export failed: {exc}")

st.divider()
st.caption("Data provider: World Bank Open Data API. This educational tool is not an official forecast or professional decision system.")
