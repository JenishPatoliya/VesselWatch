
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="VesselWatch",
    page_icon="⚓",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #070b14; }
    .main .block-container { 
        padding: 1rem 2rem;
        background-color: #070b14;
    }
    [data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2d45;
    }
    [data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1e2d45;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {
        color: #4a5568 !important;
        font-size: 10px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase;
        font-family: monospace;
    }
    [data-testid="metric-container"] 
    [data-testid="metric-value"] {
        color: white !important;
        font-family: monospace !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    h1 { 
        color: #00d4ff !important; 
        font-family: monospace !important;
        font-size: 22px !important;
        letter-spacing: 3px !important;
    }
    h2, h3 { 
        color: #e2e8f0 !important;
        font-family: monospace !important;
        font-size: 12px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
    }
    p, div, span, label {
        color: #e2e8f0;
        font-family: monospace;
    }
    hr { border-color: #1e2d45; }
    .stCaption { 
        color: #4a5568 !important; 
        font-family: monospace !important;
        font-size: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        return pd.read_csv("final_results.csv")
    except:
        return pd.read_parquet(
            "/content/drive/MyDrive/VesselWatch/data/processed/final_results_v2.parquet"
        )

@st.cache_data
def load_shap():
    try:
        return pd.read_csv("shap_explanations.csv")
    except:
        return pd.read_parquet(
            "/content/drive/MyDrive/VesselWatch/data/processed/shap_explanations_v2.parquet"
        )

ML_COLS = [
    "speed_mean","speed_std","speed_min","speed_max",
    "speed_variance","total_distance_km","total_time_hrs",
    "loitering_score","max_gap_hrs","total_gaps","gap_flag",
    "position_jump_km","dist_from_port_km",
    "behavioral_score","speed_consistency"
]

with st.spinner("Loading VesselWatch..."):
    features = load_data()
    shap_df  = load_shap()

# ── SIDEBAR ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;
                gap:10px;margin-bottom:8px">
        <div style="background:linear-gradient(
                    135deg,#00d4ff,#0077ff);
                    width:36px;height:36px;
                    border-radius:8px;display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:18px">⚓</div>
        <div>
            <div style="color:#00d4ff;
                        font-family:monospace;
                        font-weight:700;font-size:16px;
                        letter-spacing:2px">VESSELWATCH</div>
            <div style="color:#4a5568;
                        font-family:monospace;
                        font-size:9px">
                        Maritime Anomaly Detection</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### FILTERS")

    vessel_types = ["All"] + sorted(
        features["VesselTypeLabel"].unique().tolist()
    )
    selected_type = st.selectbox("Ship Type", vessel_types)

    anomaly_options = [
        "All Anomalies","Dark Event",
        "Rendezvous","High Risk Only"
    ]
    selected_anomaly = st.selectbox(
        "Anomaly Type", anomaly_options
    )

    risk_threshold = st.slider(
        "Risk Score Threshold",
        0.0, 1.0, 0.5, 0.05
    )

    st.divider()

    # Apply filters
    filt = features.copy()
    if selected_type != "All":
        filt = filt[filt["VesselTypeLabel"]==selected_type]
    if selected_anomaly == "Dark Event":
        filt = filt[filt["gap_flag"]==1]
    elif selected_anomaly == "Rendezvous":
        filt = filt[filt["rendezvous_flag"]==1]
    elif selected_anomaly == "High Risk Only":
        filt = filt[filt["final_risk_score"]>0.7]

    flagged = filt[filt["final_risk_score"]>=risk_threshold]

    # Sidebar stats
    st.markdown("### STATISTICS")
    total  = len(filt)
    high   = len(filt[filt["final_risk_score"]>0.7])
    medium = len(filt[(filt["final_risk_score"]>=0.5)&(filt["final_risk_score"]<=0.7)])
    low    = len(filt[filt["final_risk_score"]<0.5])

    st.metric("Vessels Monitored", f"{total:,}")
    st.metric("Anomalies Detected", f"{len(flagged):,}")
    st.metric("High Risk > 0.7",    f"{high:,}")
    st.metric("Dark Events",
              f"{int(filt['gap_flag'].sum()):,}")

    st.divider()
    st.markdown("### RISK BREAKDOWN")
    st.markdown(f"""
    <div style="font-family:monospace;font-size:11px">
        <div style="display:flex;justify-content:space-between;
                    padding:4px 0;border-bottom:1px solid #1e2d45">
            <span style="color:#ff3b3b">🔴 High Risk</span>
            <span style="color:#ff3b3b;font-weight:700">
                {high:,} ({high/total*100:.1f}%)</span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    padding:4px 0;border-bottom:1px solid #1e2d45">
            <span style="color:#ff8c00">🟠 Medium Risk</span>
            <span style="color:#ff8c00;font-weight:700">
                {medium:,} ({medium/total*100:.1f}%)</span>
        </div>
        <div style="display:flex;justify-content:space-between;
                    padding:4px 0">
            <span style="color:#3b82f6">🔵 Normal</span>
            <span style="color:#3b82f6;font-weight:700">
                {low:,} ({low/total*100:.1f}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── NAVBAR ────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;
            justify-content:space-between;
            background:#0d1220;
            border:1px solid #1e2d45;
            border-radius:10px;
            padding:12px 20px;margin-bottom:16px">
    <div style="color:#00d4ff;font-family:monospace;
                font-weight:700;font-size:20px;
                letter-spacing:3px">⚓ VESSELWATCH</div>
    <div style="color:#4a5568;font-family:monospace;
                font-size:11px">
                Maritime Anomaly Detection System</div>
    <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;
                    background:#00ff88;
                    border-radius:50%"></div>
        <div style="color:#00ff88;font-family:monospace;
                    font-size:11px">LIVE ANALYSIS</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ───────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("VESSELS MONITORED", f"{len(features):,}")
c2.metric("ANOMALIES DETECTED",
    f"{len(features[features['final_risk_score']>=0.5]):,}")
c3.metric("HIGH RISK > 0.7",
    f"{len(features[features['final_risk_score']>0.7]):,}")
c4.metric("DARK EVENTS",
    f"{int(features['gap_flag'].sum()):,}")

st.divider()

# ── MAP + PANEL ───────────────────────────────────────
map_col, panel_col = st.columns([7,3])

with map_col:
    st.markdown("### 🗺️ VESSEL TRACKING MAP")
    st.caption(
        "Click any vessel • Red=High Risk • "
        "Orange=Medium • Blue=Normal"
    )

    # Unique key forces map to re-render on filter change
    map_key = f"{selected_type}_{selected_anomaly}_{risk_threshold}"

    m = folium.Map(
        location=[28,-88],
        zoom_start=4,
        tiles="CartoDB dark_matter"
    )

    # Only plot filtered vessels
    plot_df = filt.dropna(subset=["last_lat","last_lon"])
    plot_df = plot_df[
        (plot_df["last_lat"].between(-90,90)) &
        (plot_df["last_lon"].between(-180,180))
    ].head(300)  # Reduced for faster rendering

    for _, row in plot_df.iterrows():
        risk = row["final_risk_score"]
        if risk > 0.7:
            color,radius = "red", 9
        elif risk >= 0.5:
            color,radius = "orange", 7
        else:
            color,radius = "blue", 4

        reasons = []
        if row["gap_flag"]==1:
            reasons.append(
                f"AIS gap {row['max_gap_hrs']:.1f}hrs"
            )
        if row["position_jump_km"]>100:
            reasons.append(
                f"Jump {row['position_jump_km']:.0f}km"
            )
        if row["rendezvous_flag"]==1:
            reasons.append("Rendezvous")
        reason_str = " · ".join(reasons)                      if reasons else "Behavioral anomaly"

        popup_html = f"""
        <div style="font-family:monospace;
                    background:#0d1220;color:#e2e8f0;
                    padding:12px;min-width:220px;
                    border-radius:8px">
            <div style="color:{'#ff3b3b' if risk>0.7 
                         else '#ff8c00' if risk>=0.5 
                         else '#3b82f6'};
                        font-weight:700;font-size:13px;
                        margin-bottom:6px">
                ⚠ {row["VesselName"]}
            </div>
            <div style="color:#4a5568;font-size:10px;
                        margin-bottom:8px">
                MMSI: {row["MMSI"]} | 
                {row["VesselTypeLabel"]}
            </div>
            <div style="background:#1e2d45;height:4px;
                        border-radius:2px;
                        margin-bottom:8px">
                <div style="background:{'#ff3b3b' 
                            if risk>0.7 else '#ff8c00'};
                            height:4px;
                            width:{int(risk*100)}%;
                            border-radius:2px"></div>
            </div>
            <div style="font-size:10px;color:#ff3b3b">
                ⚠ {reason_str}
            </div>
            <div style="font-size:10px;
                        color:#4a5568;margin-top:4px">
                Risk: {risk:.3f} | 
                Gap: {row["max_gap_hrs"]:.1f}hrs | 
                Jump: {row["position_jump_km"]:.0f}km
            </div>
        </div>
        """
        folium.CircleMarker(
            location=[row["last_lat"],row["last_lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(
                popup_html, max_width=280
            ),
            tooltip=f"{row['VesselName']} | "
                    f"Risk:{risk:.2f} | "
                    f"{row['VesselTypeLabel']}"
        ).add_to(m)

    st_folium(
        m, 
        width=None, 
        height=420,
        key=map_key,
        returned_objects=[]
    )

with panel_col:
    st.markdown("### ⚠️ FLAGGED VESSELS")
    st.caption(f"Showing {len(flagged)} vessels above threshold {risk_threshold}")

    if len(flagged) == 0:
        st.info("No vessels flagged at current threshold. Lower the Risk Score Threshold slider.")
    else:
        # Scrollable container
        vessel_container = st.container(height=450)
        with vessel_container:
            top_flagged = flagged.nlargest(50,"final_risk_score")
            for _, row in top_flagged.iterrows():
                risk = row["final_risk_score"]
                badge = "🔴" if risk>0.7 else "🟠"
                with st.expander(
                    f"{badge} {row['VesselName']}  {risk:.3f}"
                ):
                    st.markdown(f"""
                    <div style="font-family:monospace;
                                font-size:11px">
                        <div style="color:#4a5568">
                            MMSI: {row["MMSI"]}</div>
                        <div style="color:#4a5568">
                            Type: {row["VesselTypeLabel"]}</div>
                        <br>
                        <div style="color:#ff3b3b">
                            ⚠ AIS Gap: 
                            {row["max_gap_hrs"]:.1f} hrs</div>
                        <div style="color:#ff3b3b">
                            ⚠ Jump: 
                            {row["position_jump_km"]:.0f} km</div>
                        <div style="color:#ff8c00">
                            Port: 
                            {row["dist_from_port_km"]:.0f} km</div>
                        <div style="color:#4a5568">
                            Behavioral: 
                            {row["behavioral_score"]:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

st.divider()

# ── CHARTS ────────────────────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("### 📊 DETECTIONS BY VESSEL TYPE")
    type_data = filt[
        filt["final_risk_score"]>=0.5
    ]["VesselTypeLabel"].value_counts().reset_index()
    type_data.columns = ["Type","Count"]

    if len(type_data) > 0:
        fig1 = px.bar(
            type_data,x="Count",y="Type",
            orientation="h",
            color="Count",
            color_continuous_scale="Reds"
        )
        fig1.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="white",family="monospace"),
            height=300,
            margin=dict(l=10,r=10,t=10,b=10),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#1e2d45"),
            yaxis=dict(gridcolor="#1e2d45")
        )
        st.plotly_chart(fig1,use_container_width=True)
    else:
        st.info("No anomalies detected for current filter")

with ch2:
    st.markdown("### 🍩 ANOMALY BREAKDOWN")
    dark   = int(filt["gap_flag"].sum())
    rendez = int(filt["rendezvous_flag"].sum())
    loiter = int((filt["loitering_score"]>0.9).sum())
    speed  = int((filt["speed_variance"]>10).sum())
    total_anomalies = dark+rendez+loiter+speed

    if total_anomalies > 0:
        fig2 = go.Figure(go.Pie(
            labels=["Dark Event","Rendezvous",
                    "Loitering","Speed Anomaly"],
            values=[dark,rendez,loiter,speed],
            hole=0.6,
            marker_colors=["#ff3b3b","#ffd700",
                           "#ff8c00","#00d4ff"]
        ))
        fig2.update_layout(
            paper_bgcolor="#111827",
            font=dict(color="white",family="monospace"),
            height=300,
            margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(
                font=dict(color="white",size=10),
                bgcolor="#111827"
            )
        )
        fig2.add_annotation(
            text=f"<b>{total_anomalies}</b>",
            x=0.5,y=0.5,
            font=dict(size=16,color="white",
                      family="monospace"),
            showarrow=False
        )
        st.plotly_chart(fig2,use_container_width=True)
    else:
        st.info("No anomalies for current filter")

st.divider()

# ── SHAP SECTION ──────────────────────────────────────
st.markdown(
    "### 🔍 WHY WAS THIS VESSEL FLAGGED? — SHAP"
)

top_vessels = filt.nlargest(50,"final_risk_score")
vessel_options = top_vessels["VesselName"].tolist()

if len(vessel_options) == 0:
    st.info("No vessels to explain at current filter settings.")
else:
    selected = st.selectbox(
        "Select vessel to explain:", vessel_options
    )

    if selected:
        v = filt[filt["VesselName"]==selected].iloc[0]
        vs = shap_df[shap_df["VesselName"]==selected]

        ic, sc = st.columns([1,2])

        with ic:
            risk = v["final_risk_score"]
            st.markdown(f"""
            <div style="background:#111827;
                        border:1px solid #1e2d45;
                        border-radius:8px;padding:16px;
                        font-family:monospace">
                <div style="color:{'#ff3b3b' 
                             if risk>0.7 else '#ff8c00'};
                            font-size:15px;
                            font-weight:700;
                            margin-bottom:12px">
                    ⚠ {v["VesselName"]}
                </div>
                <div style="color:#4a5568;font-size:10px">
                    TYPE</div>
                <div style="color:white;margin-bottom:8px">
                    {v["VesselTypeLabel"]}</div>
                <div style="color:#4a5568;font-size:10px">
                    MMSI</div>
                <div style="color:white;margin-bottom:8px">
                    {v["MMSI"]}</div>
                <div style="color:#4a5568;font-size:10px">
                    RISK SCORE</div>
                <div style="color:{'#ff3b3b' 
                             if risk>0.7 else '#ff8c00'};
                            font-size:24px;
                            font-weight:700;
                            margin-bottom:8px">
                    {risk:.4f}</div>
                <div style="background:#1e2d45;height:6px;
                            border-radius:3px;
                            margin-bottom:14px">
                    <div style="background:linear-gradient(
                                90deg,#ff8c00,#ff3b3b);
                                height:6px;
                                width:{int(risk*100)}%;
                                border-radius:3px"></div>
                </div>
                <div style="color:#ff3b3b;font-size:11px">
                    ⚠ AIS Gap: 
                    {v["max_gap_hrs"]:.1f} hrs<br>
                    ⚠ Jump: 
                    {v["position_jump_km"]:.0f} km<br>
                    ⚠ Port: 
                    {v["dist_from_port_km"]:.0f} km<br>
                    ⚠ Behavioral: 
                    {v["behavioral_score"]:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sc:
            if len(vs) > 0:
                sv = vs[ML_COLS].iloc[0]
                df_plot = pd.DataFrame({
                    "Feature": ML_COLS,
                    "Value": [-x for x in sv],
                    "Direction": [
                        "Pushes to Anomaly"
                        if x<0 else "Pushes to Normal"
                        for x in sv
                    ]
                }).sort_values("Value")

                fig3 = px.bar(
                    df_plot,x="Value",y="Feature",
                    color="Direction",orientation="h",
                    color_discrete_map={
                        "Pushes to Anomaly":"#ff3b3b",
                        "Pushes to Normal":"#3b82f6"
                    }
                )
                fig3.update_layout(
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="white",
                              family="monospace",
                              size=11),
                    height=420,
                    margin=dict(l=10,r=10,t=10,b=10),
                    xaxis=dict(
                        title="SHAP Value",
                        gridcolor="#1e2d45",
                        zerolinecolor="#4a5568"
                    ),
                    yaxis=dict(gridcolor="#1e2d45"),
                    legend=dict(
                        font=dict(color="white",size=10),
                        bgcolor="#111827",title=""
                    )
                )
                st.plotly_chart(
                    fig3,use_container_width=True
                )
                st.caption(
                    "Red = pushed risk UP | "
                    "Blue = pushed risk DOWN"
                )
            else:
                st.info(
                    "SHAP data available for top 50 "
                    "vessels only. Select from list above."
                )

st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;
            font-family:monospace;font-size:10px">
    VesselWatch v2.0 | Isolation Forest + 
    DBSCAN + LSTM Autoencoder | 
    SHAP Explainability | 
    5.6M AIS Records | 16,937 Vessels Analyzed
</div>
""", unsafe_allow_html=True)
