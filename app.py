
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="VesselWatch",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #070b14; }
    .main .block-container { 
        padding: 1rem 2rem;
        background-color: #070b14;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1220;
        border-right: 1px solid #1e2d45;
    }
    
    /* Metrics */
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
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: white !important;
        font-family: monospace !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    /* Headers */
    h1 { 
        color: #00d4ff !important; 
        font-family: monospace !important;
        font-size: 24px !important;
        letter-spacing: 3px !important;
    }
    h2, h3 { 
        color: #e2e8f0 !important;
        font-family: monospace !important;
        font-size: 13px !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
    }
    
    /* Text */
    p, div, span, label {
        color: #e2e8f0;
        font-family: monospace;
    }
    
    /* Selectbox and slider */
    [data-testid="stSelectbox"] > div {
        background-color: #111827;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        font-family: monospace;
        font-size: 12px;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background-color: #111827;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        margin-bottom: 4px;
    }
    
    /* Divider */
    hr { border-color: #1e2d45; }
    
    /* Caption */
    .stCaption { 
        color: #4a5568 !important; 
        font-family: monospace !important;
        font-size: 10px !important;
    }
    
    /* Scrollable vessel list */
    .vessel-list {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 4px;
    }
    
    /* Risk badge */
    .badge-high {
        background: rgba(255,59,59,0.15);
        color: #ff3b3b;
        border: 1px solid rgba(255,59,59,0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-medium {
        background: rgba(255,140,0,0.15);
        color: #ff8c00;
        border: 1px solid rgba(255,140,0,0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 11px;
        font-weight: 700;
    }
    
    /* Vessel card */
    .vessel-card {
        background: #111827;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 6px;
        cursor: pointer;
    }
    .vessel-card:hover {
        border-color: #00d4ff;
        background: rgba(0,212,255,0.05);
    }
    .vessel-card-selected {
        border-left: 3px solid #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ───────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("final_results.csv")

@st.cache_data
def load_shap():
    return pd.read_csv("shap_explanations.csv")

features = load_data()
shap_df  = load_shap()

ML_COLS = [
    "speed_mean","speed_std","speed_min","speed_max",
    "speed_variance","total_distance_km","total_time_hrs",
    "loitering_score","max_gap_hrs","total_gaps","gap_flag",
    "position_jump_km","dist_from_port_km",
    "behavioral_score","speed_consistency"
]

# ─── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="background:linear-gradient(135deg,#00d4ff,#0077ff);
                    width:36px;height:36px;border-radius:8px;
                    display:flex;align-items:center;
                    justify-content:center;font-size:18px">⚓</div>
        <div>
            <div style="color:#00d4ff;font-family:monospace;
                        font-weight:700;font-size:16px;
                        letter-spacing:2px">VESSELWATCH</div>
            <div style="color:#4a5568;font-family:monospace;
                        font-size:9px">Maritime Anomaly Detection</div>
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

    st.markdown("### STATISTICS")
    st.metric("Vessels Monitored", f"{len(filt):,}")
    st.metric("Anomalies Detected", f"{len(flagged):,}")
    st.metric("High Risk  > 0.7",
              f"{len(filt[filt['final_risk_score']>0.7]):,}")
    st.metric("Dark Events",
              f"{int(filt['gap_flag'].sum()):,}")

# ─── NAVBAR ──────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;
            justify-content:space-between;
            background:#0d1220;border:1px solid #1e2d45;
            border-radius:10px;padding:12px 20px;
            margin-bottom:16px">
    <div style="color:#00d4ff;font-family:monospace;
                font-weight:700;font-size:20px;
                letter-spacing:3px">⚓ VESSELWATCH</div>
    <div style="color:#4a5568;font-family:monospace;
                font-size:11px">
        Maritime Anomaly Detection System
    </div>
    <div style="display:flex;align-items:center;gap:8px">
        <div style="width:8px;height:8px;background:#00ff88;
                    border-radius:50%;
                    animation:pulse 2s infinite"></div>
        <div style="color:#00ff88;font-family:monospace;
                    font-size:11px">LIVE ANALYSIS</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP METRICS ─────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("VESSELS MONITORED", f"{len(features):,}")
c2.metric("ANOMALIES DETECTED",
          f"{len(features[features['final_risk_score']>=0.5]):,}")
c3.metric("HIGH RISK  > 0.7",
          f"{len(features[features['final_risk_score']>0.7]):,}")
c4.metric("DARK EVENTS",
          f"{int(features['gap_flag'].sum()):,}")

st.divider()

# ─── MAP + FLAGGED VESSELS ────────────────────────────
map_col, panel_col = st.columns([7, 3])

with map_col:
    st.markdown("### 🗺️ VESSEL TRACKING MAP")
    st.caption("Click any vessel to see details • Red = High Risk • Orange = Medium • Blue = Normal")

    m = folium.Map(
        location=[28, -88],
        zoom_start=4,
        tiles="CartoDB dark_matter"
    )

    # Add vessels to map
    plot_df = filt.dropna(subset=["last_lat","last_lon"])
    plot_df = plot_df[
        (plot_df["last_lat"].between(-90,90)) &
        (plot_df["last_lon"].between(-180,180))
    ].head(600)

    for _, row in plot_df.iterrows():
        risk  = row["final_risk_score"]
        
        if risk > 0.7:
            color, radius, fill_op = "red", 9, 0.9
        elif risk >= 0.5:
            color, radius, fill_op = "orange", 7, 0.8
        else:
            color, radius, fill_op = "blue", 4, 0.6

        # Anomaly reason
        reasons = []
        if row["gap_flag"]==1:
            reasons.append(f"AIS gap {row['max_gap_hrs']:.1f}hrs")
        if row["position_jump_km"]>100:
            reasons.append(f"Jump {row['position_jump_km']:.0f}km")
        if row["rendezvous_flag"]==1:
            reasons.append("Rendezvous detected")
        if row["loitering_score"]>0.9:
            reasons.append("Loitering")
        reason_str = " · ".join(reasons) if reasons else "Behavioral anomaly"

        popup_html = f"""
        <div style="font-family:monospace;
                    background:#0d1220;color:#e2e8f0;
                    padding:12px;min-width:220px;
                    border-radius:8px">
            <div style="color:{'#ff3b3b' if risk>0.7 else '#ff8c00' if risk>=0.5 else '#3b82f6'};
                        font-weight:700;font-size:13px;
                        margin-bottom:6px">
                ⚠ {row["VesselName"]}
            </div>
            <div style="color:#4a5568;font-size:10px;
                        margin-bottom:8px">
                MMSI: {row["MMSI"]} | {row["VesselTypeLabel"]}
            </div>
            <div style="display:flex;justify-content:space-between;
                        margin-bottom:4px">
                <span style="color:#4a5568;font-size:10px">RISK SCORE</span>
                <span style="color:{'#ff3b3b' if risk>0.7 else '#ff8c00'};
                             font-weight:700">{risk:.3f}</span>
            </div>
            <div style="background:#1e2d45;height:4px;
                        border-radius:2px;margin-bottom:8px">
                <div style="background:{'#ff3b3b' if risk>0.7 else '#ff8c00'};
                            height:4px;width:{int(risk*100)}%;
                            border-radius:2px"></div>
            </div>
            <div style="font-size:10px;color:#ff3b3b">
                ⚠ {reason_str}
            </div>
            <div style="font-size:10px;color:#4a5568;margin-top:4px">
                AIS Gap: {row["max_gap_hrs"]:.1f}hrs | 
                Jump: {row["position_jump_km"]:.0f}km |
                Port: {row["dist_from_port_km"]:.0f}km
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row["last_lat"], row["last_lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=fill_op,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{row['VesselName']} | Risk: {risk:.2f} | {row['VesselTypeLabel']}"
        ).add_to(m)

    st_folium(m, width=None, height=380,
              returned_objects=[])

with panel_col:
    st.markdown("### ⚠️ FLAGGED VESSELS")
    
    top_flagged = flagged.nlargest(30, "final_risk_score")
    
    for _, row in top_flagged.iterrows():
        risk = row["final_risk_score"]
        badge = "🔴" if risk > 0.7 else "🟠"
        tag   = "DARK EVENT" if row["gap_flag"]==1 else                 "RENDEZVOUS" if row["rendezvous_flag"]==1 else                 "ANOMALY"
        
        with st.expander(
            f"{badge} {row['VesselName']}  {risk:.3f}"
        ):
            st.markdown(f"""
            <div style="font-family:monospace;font-size:11px">
                <div style="color:#4a5568">MMSI: {row['MMSI']}</div>
                <div style="color:#4a5568">
                    Type: {row['VesselTypeLabel']}</div>
                <br>
                <div style="color:#ff3b3b">
                    ⚠ AIS Gap: {row['max_gap_hrs']:.1f} hrs</div>
                <div style="color:#ff3b3b">
                    ⚠ Jump: {row['position_jump_km']:.0f} km</div>
                <div style="color:#ff8c00">
                    Port Dist: {row['dist_from_port_km']:.0f} km</div>
            </div>
            """, unsafe_allow_html=True)

st.divider()

# ─── CHARTS ──────────────────────────────────────────
ch1, ch2 = st.columns(2)

with ch1:
    st.markdown("### 📊 DETECTIONS BY VESSEL TYPE")
    type_data = features[
        features["final_risk_score"]>=0.5
    ]["VesselTypeLabel"].value_counts().reset_index()
    type_data.columns = ["Type","Count"]
    
    fig1 = px.bar(
        type_data, x="Count", y="Type",
        orientation="h",
        color="Count",
        color_continuous_scale="Reds",
        title=""
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
    st.plotly_chart(fig1, use_container_width=True)

with ch2:
    st.markdown("### 🍩 ANOMALY TYPE BREAKDOWN")
    
    dark    = int(features["gap_flag"].sum())
    rendez  = int(features["rendezvous_flag"].sum())
    loiter  = int((features["loitering_score"]>0.9).sum())
    speed   = int((features["speed_variance"]>10).sum())
    
    fig2 = go.Figure(go.Pie(
        labels=["Dark Event","Rendezvous",
                "Loitering","Speed Anomaly"],
        values=[dark, rendez, loiter, speed],
        hole=0.6,
        marker_colors=["#ff3b3b","#ffd700",
                       "#ff8c00","#00d4ff"]
    ))
    fig2.update_layout(
        paper_bgcolor="#111827",
        font=dict(color="white",family="monospace"),
        height=300,
        margin=dict(l=10,r=10,t=10,b=10),
        showlegend=True,
        legend=dict(
            font=dict(color="white",size=10),
            bgcolor="#111827"
        )
    )
    fig2.add_annotation(
        text=f"<b>{dark+rendez+loiter+speed}</b><br>total",
        x=0.5, y=0.5,
        font=dict(size=14,color="white",
                  family="monospace"),
        showarrow=False
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── SHAP SECTION ────────────────────────────────────
st.markdown("### 🔍 WHY WAS THIS VESSEL FLAGGED? (SHAP EXPLAINABILITY)")

vessel_options = features.nlargest(
    50,"final_risk_score"
)["VesselName"].tolist()

selected = st.selectbox(
    "Select vessel to explain:", vessel_options
)

if selected:
    v_data  = features[
        features["VesselName"]==selected
    ].iloc[0]
    v_shap  = shap_df[
        shap_df["VesselName"]==selected
    ]
    
    info_col, shap_col = st.columns([1,2])
    
    with info_col:
        risk = v_data["final_risk_score"]
        st.markdown(f"""
        <div style="background:#111827;border:1px solid #1e2d45;
                    border-radius:8px;padding:16px;
                    font-family:monospace">
            <div style="color:{'#ff3b3b' if risk>0.7 else '#ff8c00'};
                        font-size:16px;font-weight:700;
                        margin-bottom:12px">
                ⚠ {v_data["VesselName"]}
            </div>
            <div style="color:#4a5568;font-size:10px;
                        margin-bottom:4px">MMSI</div>
            <div style="color:white;margin-bottom:10px">
                {v_data["MMSI"]}</div>
            <div style="color:#4a5568;font-size:10px;
                        margin-bottom:4px">TYPE</div>
            <div style="color:white;margin-bottom:10px">
                {v_data["VesselTypeLabel"]}</div>
            <div style="color:#4a5568;font-size:10px;
                        margin-bottom:4px">RISK SCORE</div>
            <div style="color:{'#ff3b3b' if risk>0.7 else '#ff8c00'};
                        font-size:24px;font-weight:700;
                        margin-bottom:8px">{risk:.4f}</div>
            <div style="background:#1e2d45;height:6px;
                        border-radius:3px;margin-bottom:14px">
                <div style="background:linear-gradient(90deg,#ff8c00,#ff3b3b);
                            height:6px;width:{int(risk*100)}%;
                            border-radius:3px"></div>
            </div>
            <div style="color:#ff3b3b;font-size:11px">
                ⚠ AIS Gap: {v_data["max_gap_hrs"]:.1f} hrs<br>
                ⚠ Jump: {v_data["position_jump_km"]:.0f} km<br>
                ⚠ Port: {v_data["dist_from_port_km"]:.0f} km away<br>
                ⚠ Behavioral score: {v_data["behavioral_score"]:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with shap_col:
        if len(v_shap) > 0:
            shap_vals = v_shap[ML_COLS].iloc[0]
            
            df_plot = pd.DataFrame({
                "Feature": ML_COLS,
                "Value": [-v for v in shap_vals],
                "Direction": [
                    "Pushes to Anomaly" if v<0 
                    else "Pushes to Normal" 
                    for v in shap_vals
                ]
            }).sort_values("Value")
            
            fig3 = px.bar(
                df_plot,
                x="Value", y="Feature",
                color="Direction",
                orientation="h",
                color_discrete_map={
                    "Pushes to Anomaly":"#ff3b3b",
                    "Pushes to Normal":"#3b82f6"
                }
            )
            fig3.update_layout(
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                font=dict(color="white",
                          family="monospace",size=11),
                height=420,
                margin=dict(l=10,r=10,t=10,b=10),
                xaxis=dict(
                    title="SHAP Contribution Value",
                    gridcolor="#1e2d45",
                    zerolinecolor="#4a5568"
                ),
                yaxis=dict(gridcolor="#1e2d45"),
                legend=dict(
                    font=dict(color="white",size=10),
                    bgcolor="#111827",
                    title=""
                )
            )
            
            # Axis labels
            fig3.add_annotation(
                x=df_plot["Value"].min(),
                y=-0.8,
                text="← Pushes to Normal",
                showarrow=False,
                font=dict(color="#3b82f6",
                          size=9,family="monospace")
            )
            fig3.add_annotation(
                x=df_plot["Value"].max(),
                y=-0.8,
                text="Pushes to Anomaly →",
                showarrow=False,
                font=dict(color="#ff3b3b",
                          size=9,family="monospace")
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            st.caption(
                "Red bars = pushed risk score UP | "
                "Blue bars = pushed risk score DOWN"
            )
        else:
            st.info(
                "Select a vessel from the top 50 list "
                "to see its SHAP explanation"
            )

st.divider()
st.markdown("""
<div style="text-align:center;color:#4a5568;
            font-family:monospace;font-size:10px;
            padding:10px">
    VesselWatch v2.0 | 
    Isolation Forest + DBSCAN + LSTM Autoencoder | 
    SHAP Explainability | 
    Built with Python · Scikit-learn · TensorFlow · Streamlit
</div>
""", unsafe_allow_html=True)
