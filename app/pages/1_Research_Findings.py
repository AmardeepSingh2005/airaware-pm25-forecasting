from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = ROOT / "reports" / "weather_ablation_comparison.csv"

st.set_page_config(
    page_title="AirAware | Research Findings",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Findings")
st.caption("AirAware | Weather feature ablation experiment")

st.subheader("Research question")
st.markdown(
    "**Does adding weather data improve forecasting performance?**"
)

st.write(
    "We compared two Random Forest models using the same "
    "chronological training and test split."
)

if not RESULTS_FILE.exists():
    st.error(f"Results file not found: {RESULTS_FILE}")
    st.stop()

results = pd.read_csv(RESULTS_FILE)

without_weather = results.iloc[0]
with_weather = results.iloc[1]

mae_change = (
    (with_weather["MAE"] - without_weather["MAE"])
    / without_weather["MAE"] * 100
)

rmse_change = (
    (with_weather["RMSE"] - without_weather["RMSE"])
    / without_weather["RMSE"] * 100
)

a, b, c = st.columns(3)
a.metric("Training examples", "1,216")
b.metric("Test examples", "304")
c.metric("Test period", "Sep–Oct 2026")

st.subheader("Model comparison")

display = results.rename(columns={
    "model": "Model",
    "feature_count": "Number of features",
    "MAE": "MAE (µg/m³)",
    "RMSE": "RMSE (µg/m³)",
})

st.dataframe(
    display.round(3),
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)

with left:
    st.metric(
        "Change in MAE after adding weather",
        f"{mae_change:+.2f}%"
    )

with right:
    st.metric(
        "Change in RMSE after adding weather",
        f"{rmse_change:+.2f}%"
    )

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

labels = ["Without weather", "With weather"]
colors = [without_weather["MAE"], with_weather["MAE"]]

axes[0].bar(labels, colors)
axes[0].set_title("Mean Absolute Error")
axes[0].set_ylabel("MAE (µg/m³)")

rmse_values = [
    without_weather["RMSE"],
    with_weather["RMSE"],
]
axes[1].bar(labels, rmse_values)
axes[1].set_title("Root Mean Squared Error")
axes[1].set_ylabel("RMSE (µg/m³)")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.25)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")

if mae_change < 0 and rmse_change < 0:
    st.success(
        "Both error metrics improved after adding weather features "
        "on this test set."
    )
elif mae_change > 0 and rmse_change > 0:
    st.warning(
        "Both error metrics worsened after adding weather features "
        "on this test set. Recent pollution history may already "
        "provide a strong short-term predictive signal."
    )
else:
    st.info(
        "The two error metrics moved in different directions. "
        "The practical trade-off requires further analysis."
    )

st.markdown(
    """
    ### Limitations

    - This is one chronological holdout evaluation.
    - The dataset covers one sensor and a limited historical period.
    - Hourly observations contain substantial gaps.
    - Weather values are historical model-based estimates.
    - The experiment does not show that weather is unhelpful for all
      forecasting models, cities, or time horizons.
    """
)

st.caption(
    "AirAware is an educational research prototype. "
    "Results are preliminary and should not be used for health decisions."
)
