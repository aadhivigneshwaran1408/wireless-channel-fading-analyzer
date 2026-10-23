import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Wireless Channel Fading Analyzer",
    page_icon="📡",
    layout="wide"
)

st.title("📡 Wireless Channel Fading Analyzer")
st.caption("Interactive simulation of multipath fading, received power, SNR, and outage probability.")

# Sidebar controls
st.sidebar.header("Simulation Settings")
model = st.sidebar.selectbox("Fading model", ["Rayleigh", "Rician"])
n_samples = st.sidebar.slider("Number of samples", 500, 10000, 3000, step=500)
tx_power_dbm = st.sidebar.slider("Transmit power (dBm)", -10, 30, 20)
path_loss_db = st.sidebar.slider("Path loss (dB)", 40, 140, 80)
noise_power_dbm = st.sidebar.slider("Noise power (dBm)", -120, -60, -100)
snr_threshold_db = st.sidebar.slider("Outage SNR threshold (dB)", -10, 30, 5)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

rician_k_db = 6.0
if model == "Rician":
    rician_k_db = st.sidebar.slider("Rician K-factor (dB)", -5, 20, 6)

# Generate normalized fading envelope
rng = np.random.default_rng(int(seed))
if model == "Rayleigh":
    h = (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)) / np.sqrt(2)
else:
    k = 10 ** (rician_k_db / 10)
    diffuse = (rng.normal(size=n_samples) + 1j * rng.normal(size=n_samples)) / np.sqrt(2)
    specular = np.sqrt(k / (k + 1))
    h = np.sqrt(1 / (k + 1)) * diffuse + specular

power_linear = np.abs(h) ** 2
power_db = 10 * np.log10(np.maximum(power_linear, 1e-12))
mean_fading_power = float(np.mean(power_linear))
# Link budget: average received power before fading, then apply instantaneous fading power
avg_rx_dbm = tx_power_dbm - path_loss_db
rx_power_dbm = avg_rx_dbm + power_db
snr_db = rx_power_dbm - noise_power_dbm
outage_probability = float(np.mean(snr_db < snr_threshold_db) * 100)
mean_rx_dbm = float(np.mean(rx_power_dbm))
mean_snr_db = float(np.mean(snr_db))

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Average received power", f"{mean_rx_dbm:.2f} dBm")
c2.metric("Average SNR", f"{mean_snr_db:.2f} dB")
c3.metric("Outage probability", f"{outage_probability:.2f}%")
c4.metric("Mean normalized fading power", f"{mean_fading_power:.3f}")

st.info(
    "This is a software simulation, not a live measurement from a physical wireless receiver. "
    "The fading channel is normalized to approximately unit average power."
)

df = pd.DataFrame({
    "Sample": np.arange(1, n_samples + 1),
    "Fading power (dB)": power_db,
    "Received power (dBm)": rx_power_dbm,
    "SNR (dB)": snr_db,
    "Outage": snr_db < snr_threshold_db
})

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Signal & Power", "📊 Fading Distribution", "📶 SNR & Outage", "📋 Data"
])

with tab1:
    st.subheader("Received signal power over time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Sample"], y=df["Received power (dBm)"],
        mode="lines", name="Received power", line=dict(width=1)
    ))
    fig.add_hline(
        y=avg_rx_dbm, line_dash="dash",
        annotation_text="Average power before fading", annotation_position="top left"
    )
    fig.update_layout(
        xaxis_title="Sample index", yaxis_title="Received power (dBm)",
        height=430, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Instantaneous fading power")
    fig2 = px.line(df, x="Sample", y="Fading power (dB)", title="Normalized fading power")
    fig2.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader(f"{model} fading envelope distribution")
    envelope = np.abs(h)
    hist = go.Figure()
    hist.add_trace(go.Histogram(x=envelope, histnorm="probability density", nbinsx=50, name="Simulated envelope"))
    hist.update_layout(
        xaxis_title="Normalized envelope amplitude", yaxis_title="Probability density",
        height=430, margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(hist, use_container_width=True)
    st.write(
        "**Interpretation:** Rayleigh fading models many reflected paths without a dominant line-of-sight path. "
        "Rician fading includes a dominant component in addition to scattered paths; a larger K-factor means a stronger dominant component."
    )

with tab3:
    st.subheader("Signal-to-noise ratio")
    fig3 = px.line(df, x="Sample", y="SNR (dB)", title="Instantaneous SNR across samples")
    fig3.add_hline(
        y=snr_threshold_db, line_dash="dash",
        annotation_text=f"Outage threshold: {snr_threshold_db} dB"
    )
    fig3.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("SNR distribution")
    fig4 = px.histogram(df, x="SNR (dB)", nbins=50, title="Distribution of instantaneous SNR")
    fig4.add_vline(x=snr_threshold_db, line_dash="dash", annotation_text="Threshold")
    fig4.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        f"**Outage probability = {outage_probability:.2f}%** of simulated samples have SNR below "
        f"the selected threshold of **{snr_threshold_db} dB**."
    )

with tab4:
    st.subheader("Simulation data")
    st.dataframe(df.head(500), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download full results as CSV",
        data=csv,
        file_name="wireless_fading_results.csv",
        mime="text/csv"
    )
    st.caption("The table previews the first 500 rows. The CSV download contains all simulated samples.")

with st.expander("Model assumptions and formulas"):
    st.markdown("""
    - **Rayleigh model:** complex channel coefficient is generated from independent Gaussian in-phase and quadrature components.
    - **Rician model:** combines a dominant component with a diffuse Gaussian component. The K-factor is the ratio of dominant to diffuse power.
    - **Average received power:** transmit power − path loss.
    - **Instantaneous received power:** average received power + fading power in dB.
    - **SNR:** instantaneous received power − noise power.
    - **Outage probability:** percentage of samples where instantaneous SNR is below the selected threshold.

    This simplified link budget does not include antenna gains, shadowing, bandwidth-dependent noise calculations, or a specific modulation scheme/BER model.
    """)
