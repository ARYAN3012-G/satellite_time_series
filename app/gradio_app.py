"""
app/gradio_app.py — Agricultural Drought Detection
Model  : Transformer (Acc 81.1%, Macro-F1 0.6995, ROC-AUC 0.9242)
Input  : Raw weather CSV — takes last 30 days automatically
Output : Next day drought prediction — Healthy / Moderate / Severe

Run: python app\gradio_app.py
     then open http://localhost:7860
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config
from inference.multi_predictor import MultiModelPredictor

# ── Load Transformer once at startup ─────────────────────────────────────────
_predictor = MultiModelPredictor(model_name="Transformer")

CLASS_COLORS = {"Healthy": "#2E7D32", "Moderate": "#E65100", "Severe": "#B71C1C"}
CLASS_EMOJI  = {"Healthy": "🟢", "Moderate": "🟡", "Severe": "🔴"}
CLASS_BG     = {"Healthy": "#E8F5E9", "Moderate": "#FFF3E0", "Severe": "#FFEBEE"}


def _resolve_date(df):
    if config.DATE_COLUMN in df.columns:
        return df
    for alias in config.DATE_COLUMN_ALIASES:
        if alias in df.columns:
            return df.rename(columns={alias: config.DATE_COLUMN})
    return df


def _make_prob_chart(probs: dict):
    classes = ["Healthy", "Moderate", "Severe"]
    values  = [probs.get(c, 0) * 100 for c in classes]
    colors  = [CLASS_COLORS[c] for c in classes]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    bars = ax.barh(classes, values, color=colors, height=0.5, edgecolor="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title("Transformer — Class Probabilities", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(min(val + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def predict(file_obj):
    if file_obj is None:
        return "<p style='color:#888;text-align:center;'>Upload a CSV file to get prediction.</p>", None

    try:
        df = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"<p style='color:red;'>❌ Cannot read file: {e}</p>", None

    df = _resolve_date(df)
    if config.DATE_COLUMN not in df.columns:
        return f"<p style='color:red;'>❌ No date column found. Need one of: {config.DATE_COLUMN_ALIASES}</p>", None

    try:
        df[config.DATE_COLUMN] = pd.to_datetime(df[config.DATE_COLUMN], format="mixed")
    except Exception as e:
        return f"<p style='color:red;'>❌ Date error: {e}</p>", None

    df = df.sort_values(config.DATE_COLUMN).reset_index(drop=True)

    missing = [c for c in config.MODEL_INPUT_FEATURES if c not in df.columns]
    if missing:
        return f"<p style='color:red;'>❌ Missing columns: {missing}</p>", None

    if len(df) < config.SEQUENCE_LENGTH_DAYS:
        return f"<p style='color:red;'>❌ Need at least 30 rows. Got {len(df)}.</p>", None

    window   = df.tail(config.SEQUENCE_LENGTH_DAYS).reset_index(drop=True)
    from_date = window[config.DATE_COLUMN].min().strftime("%d %b %Y")
    to_date   = window[config.DATE_COLUMN].max().strftime("%d %b %Y")
    predict_date = (window[config.DATE_COLUMN].max() + pd.Timedelta(days=1)).strftime("%d %b %Y")

    try:
        result = _predictor.predict(window)
    except Exception as e:
        return f"<p style='color:red;'>❌ Prediction error: {e}</p>", None

    cls   = result["predicted_class"]
    conf  = result["confidence"]
    probs = result["class_probabilities"]
    color = CLASS_COLORS[cls]
    bg    = CLASS_BG[cls]
    emoji = CLASS_EMOJI[cls]

    html = f"""
<div style="border:2px solid {color};border-radius:14px;overflow:hidden;margin-bottom:8px;">

  <div style="background:{color};padding:20px 24px;text-align:center;">
    <div style="font-size:2.2em;font-weight:800;color:white;letter-spacing:1px;">
      {emoji} &nbsp; {cls.upper()} &nbsp; {emoji}
    </div>
    <div style="color:white;opacity:0.9;font-size:1em;margin-top:4px;">
      Predicted drought status for <b>{predict_date}</b>
    </div>
  </div>

  <div style="background:{bg};padding:16px 24px;">
    <table style="width:100%;border-collapse:collapse;font-size:0.97em;">
      <tr>
        <td style="padding:6px 12px;color:#555;">🎯 Confidence</td>
        <td style="padding:6px 12px;font-weight:700;color:{color};">{conf*100:.1f}%</td>
        <td style="padding:6px 12px;color:#555;">📅 Input window</td>
        <td style="padding:6px 12px;font-weight:600;">{from_date} → {to_date}</td>
      </tr>
      <tr>
        <td style="padding:6px 12px;color:#555;">🤖 Model</td>
        <td style="padding:6px 12px;font-weight:600;">Transformer</td>
        <td style="padding:6px 12px;color:#555;">📊 Input days</td>
        <td style="padding:6px 12px;font-weight:600;">30 days</td>
      </tr>
    </table>
  </div>

</div>

<div style="background:#f5f5f5;border-radius:10px;padding:12px 20px;font-size:0.88em;color:#666;margin-top:4px;">
  <b>SPEI thresholds:</b> &nbsp;
  🟢 Healthy (SPEI ≥ 0) &nbsp;|&nbsp;
  🟡 Moderate (−1.5 ≤ SPEI &lt; 0) &nbsp;|&nbsp;
  🔴 Severe (SPEI &lt; −1.5)
</div>
"""
    return html, _make_prob_chart(probs)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Agricultural Drought Detection") as demo:

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1b4332,#2d6a4f);
                color:white;padding:22px 32px;border-radius:12px;margin-bottom:12px;">
      <h1 style="margin:0;font-size:1.8em;font-weight:800;">
        🌍 Agricultural Drought Detection
      </h1>
      <p style="margin:6px 0 0;opacity:0.85;font-size:0.95em;">
        Transformer · Predicts next day drought status from 30-day weather window
        · Anantapur / Kurnool / Sri Sathya Sai, Andhra Pradesh
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("""
### How to use
1. Upload your weather CSV
2. App reads the **last 30 days** automatically
3. Predicts **tomorrow's** drought status

### Model performance
| Metric | Score |
|--------|-------|
| Accuracy | 81.1% |
| Macro F1 | 0.6995 |
| ROC-AUC | 0.9242 |

### Required columns (9)
- Wind_Speed_10m_Mean_24h
- Temperature_Air_2m_Max_24h
- Temperature_Air_2m_Mean_24h
- Temperature_Air_2m_Min_24h
- Derived_Relative_Humidity_2m_Max_24h
- Derived_Relative_Humidity_2m_Min_24h
- Precipitation_Flux
- ReferenceET_PenmanMonteith_FAO56
- Solar_Radiation_Flux
""")

        with gr.Column(scale=3):
            file_input  = gr.File(label="📂 Upload Weather CSV", file_types=[".csv"])
            predict_btn = gr.Button("▶  Predict Drought Status", variant="primary", size="lg")
            result_html = gr.HTML()
            prob_chart  = gr.Plot(label="Class Probabilities", show_label=False)

    predict_btn.click(fn=predict, inputs=[file_input], outputs=[result_html, prob_chart])
    file_input.change(fn=predict, inputs=[file_input], outputs=[result_html, prob_chart])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
