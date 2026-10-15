from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "pollution": ROOT / "data/raw/pm25_new_delhi_recent_90d.csv",
    "combined": ROOT / "data/processed/pm25_weather_combined.csv",
    "predictions": ROOT / "reports/hourly_model_predictions.csv",
    "importance": ROOT / "reports/feature_importance.csv",
}

st.set_page_config(
    page_title="AirAware | Air quality analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# AirQo-inspired visual direction: white cards, blue accents, clear hierarchy.
st.markdown(
    """
    <style>
    :root { --aa-blue:#2463eb; --aa-ink:#263445; --aa-muted:#788797; }
    .stApp { background:#f7f9fc; }
    [data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e7ebf1; }
    [data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
    .block-container { max-width:1500px; padding-top:1.35rem; padding-bottom:2.5rem; }
    .brand {font-size:1.65rem;font-weight:800;color:#1f344a;margin:0 0 .2rem 0;}
    .brand-sub {font-size:.78rem;color:#8592a1;margin-bottom:1.4rem;}
    .hero {
      background:linear-gradient(115deg,#e7f4f1 0%,#eaf2ff 100%);
      border:1px solid #dbe7ec;border-radius:18px;padding:1.25rem 1.45rem;
      margin-bottom:1.1rem;
    }
    .hero h1 {font-size:2rem;line-height:1.2;margin:0 0 .45rem 0;color:#1f344a;}
    .hero p {margin:0;color:#536579;font-size:.98rem;line-height:1.55;}
    .eyebrow {font-size:.76rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#52728d;}
    .section-title {font-size:1.18rem;font-weight:750;color:#2b3949;margin:.25rem 0 .2rem 0;}
    .muted {color:#7a8796;font-size:.86rem;}
    .metric-card {
      background:#fff;border:1px solid #e5eaf0;border-radius:15px;
      padding:1rem 1rem .85rem 1rem;min-height:130px;
      box-shadow:0 2px 8px rgba(30,55,90,.025);
    }
    .metric-label {font-size:.86rem;color:#7b8998;margin-bottom:.4rem;}
    .metric-value {font-size:1.75rem;line-height:1.15;font-weight:800;color:#293747;}
    .metric-foot {font-size:.78rem;color:#8592a1;margin-top:.45rem;}
    .panel {
      background:#fff;border:1px solid #e5eaf0;border-radius:16px;
      padding:1rem 1.1rem .8rem 1.1rem;margin-bottom:1rem;
      box-shadow:0 2px 8px rgba(30,55,90,.025);
    }
    .pill {display:inline-block;border-radius:999px;padding:.25rem .6rem;font-size:.75rem;font-weight:700;}
    .pill-blue {background:#eaf1ff;color:#2458be;}
    .pill-green {background:#e8f8ef;color:#16814c;}
    .pill-amber {background:#fff5d9;color:#8c6500;}
    div[data-testid="stMetric"] {
      background:#fff;border:1px solid #e5eaf0;border-radius:14px;
      padding:.85rem 1rem;box-shadow:0 2px 8px rgba(30,55,90,.025);
    }
    div[data-testid="stMetricLabel"] p {font-size:.82rem;color:#7b8998;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;color:#293747;}
    .stTabs [data-baseweb="tab-list"] {gap:1.5rem;border-bottom:1px solid #e6ebf1;}
    .stTabs [data-baseweb="tab"] {height:3rem;padding:0 .15rem;font-weight:650;}
    .stTabs [aria-selected="true"] {color:#2463eb;}
    [data-testid="stDataFrame"] {border:1px solid #e5eaf0;border-radius:12px;overflow:hidden;}
    footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load source files safely.
missing = [str(p.relative_to(ROOT)) for p in FILES.values() if not p.exists()]
if missing:
    st.error("AirAware could not find the files needed to render the analytics.")
    st.write("Missing files:")
    for item in missing:
        st.code(item)
    st.info("Confirm the data/ and reports/ folders exist in the project and are committed to GitHub.")
    st.stop()

pollution = pd.read_csv(FILES["pollution"])
combined = pd.read_csv(FILES["combined"])
predictions = pd.read_csv(FILES["predictions"])
importance = pd.read_csv(FILES["importance"])

# Normalize columns used by the dashboard.
if "datetime_utc" not in pollution or "pm25" not in pollution:
    st.error("The pollution dataset needs 'datetime_utc' and 'pm25' columns.")
    st.stop()

pollution["datetime_utc"] = pd.to_datetime(pollution["datetime_utc"], utc=True, errors="coerce")
pollution["pm25"] = pd.to_numeric(pollution["pm25"], errors="coerce")
pollution = (
    pollution.dropna(subset=["datetime_utc", "pm25"])
    .sort_values("datetime_utc")
    .drop_duplicates(subset="datetime_utc")
)
if pollution.empty:
    st.error("No valid PM2.5 observations were found in the saved dataset.")
    st.stop()

if "time_local" in combined.columns:
    combined["time_local"] = pd.to_datetime(combined["time_local"], errors="coerce")
for col in ["pm25", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

latest = pollution.iloc[-1]
first_time = pollution["datetime_utc"].min()
last_time = pollution["datetime_utc"].max()
observed_days = max(1, (last_time - first_time).days + 1)
latest_pm = float(latest["pm25"])
median_pm = float(pollution["pm25"].median())
mean_pm = float(pollution["pm25"].mean())
latest_time = latest["datetime_utc"].strftime("%d %b %Y · %H:%M UTC")

# Sidebar navigation keeps the existing multipage Research Findings page intact.
with st.sidebar:
    st.markdown('<div class="brand">🌿 AirAware</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">AIR QUALITY • NEW DELHI</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Workspace**")
    st.markdown("📊  Analytics overview")
    st.markdown("🌫️  PM2.5 history")
    st.markdown("🌦️  Weather relationships")
    st.markdown("🤖  Forecast evaluation")
    st.markdown("---")
    st.caption("🎓 Student research prototype")
    st.caption("Historical dataset · Not a live alert")

# Header hero
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">AIR QUALITY ANALYTICS · NEW DELHI</div>
      <h1>Understand the air, one reading at a time 🌿</h1>
      <p>Explore historical fine-particle pollution, see how it moves with weather,
      and compare simple one-hour-ahead forecasting approaches.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Status and time-range context
st.markdown(
    f'<span class="pill pill-blue">📚 Historical data</span> '
    f'<span class="pill pill-amber">Not live monitoring</span> '
    f'<span class="muted">Data window: {first_time.strftime("%d %b %Y")} – {last_time.strftime("%d %b %Y")}</span>',
    unsafe_allow_html=True,
)
st.write("")

# KPI row inspired by the reference design, using actual project data only.
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">🌫️ Latest saved PM2.5</div>
    <div class="metric-value">{latest_pm:.1f}<span style="font-size:.85rem;font-weight:600"> µg/m³</span></div>
    <div class="metric-foot">🕒 {latest_time}</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📍 Median PM2.5</div>
    <div class="metric-value">{median_pm:.1f}<span style="font-size:.85rem;font-weight:600"> µg/m³</span></div>
    <div class="metric-foot">Middle value across saved readings</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">🧪 Valid observations</div>
    <div class="metric-value">{len(pollution):,}</div>
    <div class="metric-foot">After removing invalid rows</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-label">📅 Data span</div>
    <div class="metric-value">{observed_days}<span style="font-size:.85rem;font-weight:600"> days</span></div>
    <div class="metric-foot">Coverage may contain gaps</div></div>""", unsafe_allow_html=True)

st.write("")
overview_tab, explore_tab, weather_tab, model_tab, quality_tab = st.tabs([
    "✨ Overview", "📈 Explore PM2.5", "🌦️ Weather", "🤖 Forecast results", "🛡️ Data quality"
])

def style_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color="#e8edf3", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dce3eb")
    ax.spines["bottom"].set_color("#dce3eb")
    ax.tick_params(colors="#718096", labelsize=9)
    ax.xaxis.label.set_color("#718096")
    ax.yaxis.label.set_color("#718096")

with overview_tab:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        st.markdown('<div class="section-title">📈 Air quality over time</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Each point is a saved PM2.5 reading. The smooth line highlights the broader trend.</div>', unsafe_allow_html=True)
        series = pollution.set_index("datetime_utc")["pm25"].sort_index()
        hourly_idx = pd.date_range(series.index.min(), series.index.max(), freq="h")
        hourly = series.reindex(hourly_idx)
        smooth = hourly.rolling(24, min_periods=6).mean()
        fig, ax = plt.subplots(figsize=(9, 4.0), dpi=110)
        fig.patch.set_facecolor("#ffffff")
        ax.plot(hourly.index, hourly.values, color="#a8c8fb", linewidth=1.0, alpha=.9, label="Saved readings")
        ax.plot(smooth.index, smooth.values, color="#2463eb", linewidth=2.5, label="24-hour rolling average")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xlabel("Time (UTC)")
        style_axis(ax)
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown('<div class="section-title">🧭 At a glance</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">A quick summary of the saved measurements.</div>', unsafe_allow_html=True)
        st.metric("Average PM2.5", f"{mean_pm:.1f} µg/m³")
        st.metric("Highest saved reading", f"{pollution['pm25'].max():.1f} µg/m³")
        st.metric("Lowest saved reading", f"{pollution['pm25'].min():.1f} µg/m³")
        with st.expander("What is PM2.5?"):
            st.write("PM2.5 is fine particulate matter with a diameter of 2.5 micrometres or less. This dashboard shows concentration in micrograms per cubic metre (µg/m³). It is a research view, not a health alert.")
    st.markdown('<div class="section-title">💡 How to read this dashboard</div>', unsafe_allow_html=True)
    tip1, tip2, tip3 = st.columns(3, gap="medium")
    with tip1:
        st.markdown('<div class="panel"><b>🌫️ Pollution</b><br><span class="muted">Use the trend to spot peaks and quieter periods.</span></div>', unsafe_allow_html=True)
    with tip2:
        st.markdown('<div class="panel"><b>🌦️ Weather</b><br><span class="muted">Explore associations; correlation does not prove cause.</span></div>', unsafe_allow_html=True)
    with tip3:
        st.markdown('<div class="panel"><b>🤖 Forecasts</b><br><span class="muted">Lower MAE/RMSE means predictions were closer on this test set.</span></div>', unsafe_allow_html=True)

with explore_tab:
    st.markdown('<div class="section-title">🔎 Explore the pollution history</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Filter the date window to focus on a particular part of the saved dataset.</div>', unsafe_allow_html=True)
    min_date = pollution["datetime_utc"].min().date()
    max_date = pollution["datetime_utc"].max().date()
    date_range = st.date_input("Date range (UTC)", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = pollution[
            pollution["datetime_utc"].dt.date.between(start_date, end_date)
        ]
    else:
        filtered = pollution
    if filtered.empty:
        st.warning("No observations fall inside this date range.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings in range", f"{len(filtered):,}")
        c2.metric("Median in range", f"{filtered['pm25'].median():.1f} µg/m³")
        c3.metric("Peak in range", f"{filtered['pm25'].max():.1f} µg/m³")
        fig, ax = plt.subplots(figsize=(11, 4.1), dpi=110)
        fig.patch.set_facecolor("#fff")
        ax.plot(filtered["datetime_utc"], filtered["pm25"], color="#2463eb", linewidth=1.4)
        ax.fill_between(filtered["datetime_utc"], filtered["pm25"], color="#9bbcf8", alpha=.18)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_xlabel("Time (UTC)")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        with st.expander("View filtered observations"):
            st.dataframe(filtered[["datetime_utc", "pm25"]].tail(100), use_container_width=True, hide_index=True)

with weather_tab:
    st.markdown('<div class="section-title">🌦️ Does weather move with pollution?</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Select a weather measure to explore its relationship with PM2.5. This is descriptive analysis, not proof of causation.</div>', unsafe_allow_html=True)
    weather_choices = {
        "🌡️ Temperature (°C)": "temperature_2m",
        "💧 Relative humidity (%)": "relative_humidity_2m",
        "🌧️ Precipitation (mm)": "precipitation",
        "💨 Wind speed (km/h)": "wind_speed_10m",
    }
    available = {label: col for label, col in weather_choices.items() if col in combined.columns and "pm25" in combined.columns}
    if not available:
        st.warning("The combined dataset does not contain the expected PM2.5 and weather columns.")
    else:
        label = st.selectbox("Weather variable", list(available))
        variable = available[label]
        valid = combined[["pm25", variable]].dropna()
        corr = valid["pm25"].corr(valid[variable]) if len(valid) > 1 else float("nan")
        left, right = st.columns([1.7, 1], gap="large")
        with left:
            fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=110)
            fig.patch.set_facecolor("#fff")
            ax.scatter(valid[variable], valid["pm25"], s=22, alpha=.45, color="#2463eb", edgecolors="none")
            ax.set_xlabel(label)
            ax.set_ylabel("PM2.5 (µg/m³)")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.metric("Pearson correlation", f"{corr:+.3f}" if pd.notna(corr) else "Not available")
            st.metric("Matched records", f"{len(valid):,}")
            st.info("Correlation near +1 indicates a positive linear association; near −1 indicates a negative one; near 0 indicates little linear association.")
        cols = ["pm25"] + list(available.values())
        corr_table = combined[cols].corr()["pm25"].drop("pm25").rename("Pearson correlation").to_frame().round(3)
        corr_table.index = [x.replace("_", " ").title() for x in corr_table.index]
        st.markdown("**All available weather correlations**")
        st.dataframe(corr_table, use_container_width=True)

with model_tab:
    st.markdown('<div class="section-title">🤖 How well did the forecasts perform?</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">The experiment predicts PM2.5 one hour ahead. Lower MAE and RMSE are better.</div>', unsafe_allow_html=True)
    actual_col = "target_pm25_next_hour"
    model_cols = {"Persistence baseline": "baseline_prediction", "Random Forest": "random_forest_prediction"}
    needed = [actual_col] + list(model_cols.values())
    if not all(col in predictions.columns for col in needed):
        st.warning("Prediction report is missing one or more expected columns.")
    else:
        valid_pred = predictions.dropna(subset=needed)
        rows = []
        for name, pred_col in model_cols.items():
            error = valid_pred[actual_col] - valid_pred[pred_col]
            rows.append({"Model": name, "MAE (µg/m³)": error.abs().mean(), "RMSE (µg/m³)": (error.pow(2).mean()) ** .5})
        metric_df = pd.DataFrame(rows).set_index("Model")
        best = metric_df["MAE (µg/m³)"].idxmin()
        c1, c2, c3 = st.columns(3)
        c1.metric("Best MAE", f"{metric_df['MAE (µg/m³)'].min():.2f} µg/m³", best)
        c2.metric("Test examples", f"{len(valid_pred):,}")
        c3.metric("Forecast horizon", "1 hour")
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            st.markdown("**Model comparison**")
            st.dataframe(metric_df.round(3), use_container_width=True)
            fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=110)
            fig.patch.set_facecolor("#fff")
            metric_df.plot(kind="bar", ax=ax, color=["#2463eb", "#9bbcf8"], width=.65)
            ax.set_ylabel("Error (µg/m³)")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0)
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("**🧠 Plain-English takeaway**")
            baseline_mae = metric_df.loc["Persistence baseline", "MAE (µg/m³)"] if "Persistence baseline" in metric_df.index else float("nan")
            rf_mae = metric_df.loc["Random Forest", "MAE (µg/m³)"] if "Random Forest" in metric_df.index else float("nan")
            if pd.notna(baseline_mae) and pd.notna(rf_mae) and baseline_mae < rf_mae:
                st.warning("The persistence baseline beat Random Forest on this test period. The more complex model did not improve accuracy here.")
            else:
                st.success("Random Forest had lower MAE than the baseline on this test period.")
            st.caption("This result applies to the saved chronological test period; it does not guarantee future accuracy.")
        if {"feature", "importance"}.issubset(importance.columns):
            with st.expander("🔍 What inputs did Random Forest use most?"):
                top = importance.nlargest(8, "importance").sort_values("importance")
                fig, ax = plt.subplots(figsize=(8, 3.5), dpi=110)
                fig.patch.set_facecolor("#fff")
                ax.barh(top["feature"], top["importance"], color="#2463eb")
                ax.set_xlabel("Relative feature importance")
                style_axis(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

with quality_tab:
    st.markdown('<div class="section-title">🛡️ Can we trust the coverage?</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">These checks make the data limitations visible rather than hiding them.</div>', unsafe_allow_html=True)
    timestamps = pollution["datetime_utc"].sort_values()
    expected_hours = int((timestamps.max() - timestamps.min()).total_seconds() / 3600) + 1
    coverage = len(timestamps) / expected_hours * 100 if expected_hours > 0 else 0
    largest_gap = timestamps.diff().dt.total_seconds().div(3600).max()
    q1, q2, q3 = st.columns(3)
    q1.metric("Valid readings", f"{len(pollution):,}")
    q2.metric("Approx. hourly coverage", f"{coverage:.1f}%")
    q3.metric("Largest gap", f"{largest_gap:.0f} hours" if pd.notna(largest_gap) else "Not available")
    st.markdown("**Important limitations**")
    st.warning(
        "This dashboard uses saved historical observations and estimated weather data. "
        "Readings are unevenly spaced, gaps can affect interpretation, and the forecast experiment "
        "is a one-hour-ahead test—not a live alert or next-day forecast."
    )
    st.markdown("**Recent observations**")
    recent = pollution.tail(15)[["datetime_utc", "pm25"]].copy()
    recent["datetime_utc"] = recent["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M UTC")
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("AirAware · Student research prototype · New Delhi · Historical data only")
