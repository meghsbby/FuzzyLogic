"""
Smart Home Climate & Energy Optimizer - Interactive Streamlit Application.
State-of-the-art Fuzzy Logic Decision Support System.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from smart_home import (
    create_smart_climate_system,
    PRESET_SCENARIOS,
    plot_linguistic_variable,
    plot_defuzzification,
    plot_3d_control_surface,
)

# Page configuration
st.set_page_config(
    page_title="Smart Home Climate & Energy Fuzzy Optimizer",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphic, modern typography, rich cards)
st.markdown(
    """
    <style>
    /* Global Base */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom Title Header */
    .hero-container {
        padding: 1.5rem 2rem;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.0rem;
        margin-bottom: 0;
    }

    /* Output KPI Cards */
    .kpi-card {
        padding: 1.25rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .kpi-blue {
        background: linear-gradient(145deg, rgba(30, 58, 138, 0.35), rgba(15, 23, 42, 0.7));
        border-top: 3px solid #38bdf8;
    }
    .kpi-orange {
        background: linear-gradient(145deg, rgba(154, 52, 18, 0.35), rgba(15, 23, 42, 0.7));
        border-top: 3px solid #fb923c;
    }
    .kpi-purple {
        background: linear-gradient(145deg, rgba(88, 28, 135, 0.35), rgba(15, 23, 42, 0.7));
        border-top: 3px solid #a855f7;
    }
    .kpi-green {
        background: linear-gradient(145deg, rgba(6, 78, 59, 0.35), rgba(15, 23, 42, 0.7));
        border-top: 3px solid #34d399;
    }
    .kpi-val {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.2rem 0;
        color: #f8fafc;
    }
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
    }
    .kpi-status {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 4px;
    }

    /* Rule Badge */
    .rule-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize engine in session state
@st.cache_resource
def get_engine():
    return create_smart_climate_system()

engine = get_engine()

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("Adjust live sensor inputs or select a preset scenario.")

    # Preset Scenarios Selector
    scenario_names = ["-- Custom Manual Inputs --"] + list(PRESET_SCENARIOS.keys())
    selected_scenario = st.selectbox("📍 Load Real-World Scenario", scenario_names)

    # State management for sliders
    if selected_scenario != "-- Custom Manual Inputs --":
        sc_data = PRESET_SCENARIOS[selected_scenario]
        st.info(sc_data["description"])
        default_temp = sc_data["inputs"]["Temperature"]
        default_humidity = sc_data["inputs"]["Humidity"]
        default_occ = sc_data["inputs"]["Occupancy"]
        default_tariff = sc_data["inputs"]["ElectricityTariff"]
    else:
        default_temp = 28.0
        default_humidity = 65.0
        default_occ = 3.0
        default_tariff = 3.0

    st.markdown("---")
    st.markdown("#### 🌡️ Live Environmental Inputs")

    input_temp = st.slider(
        "Indoor Temperature (°C)",
        min_value=10.0,
        max_value=40.0,
        value=float(default_temp),
        step=0.5,
        help="Current indoor ambient temperature.",
    )

    input_humidity = st.slider(
        "Relative Humidity (%)",
        min_value=10.0,
        max_value=95.0,
        value=float(default_humidity),
        step=1.0,
        help="Indoor air relative humidity percentage.",
    )

    input_occ = st.slider(
        "Room Occupancy (people)",
        min_value=0.0,
        max_value=10.0,
        value=float(default_occ),
        step=1.0,
        help="Number of people currently present in the room.",
    )

    input_tariff = st.slider(
        "Electricity Tariff Rate",
        min_value=1.0,
        max_value=5.0,
        value=float(default_tariff),
        step=0.5,
        help="Tier 1 = Cheap Off-Peak, Tier 3 = Standard, Tier 5 = Super-Peak Demand.",
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Inference Settings")
    defuzz_method = st.selectbox(
        "Defuzzification Algorithm",
        options=["centroid", "bisector", "mom", "som", "lom"],
        index=0,
        format_func=lambda x: {
            "centroid": "Center of Gravity (Centroid)",
            "bisector": "Bisector of Area",
            "mom": "Mean of Maximum (MOM)",
            "som": "Smallest of Maximum (SOM)",
            "lom": "Largest of Maximum (LOM)",
        }[x],
        help="Method used to defuzzify aggregated fuzzy sets into crisp numbers.",
    )

# Run Inference
current_inputs = {
    "Temperature": input_temp,
    "Humidity": input_humidity,
    "Occupancy": input_occ,
    "ElectricityTariff": input_tariff,
}
result = engine.compute(current_inputs, defuzz_method=defuzz_method)

# Helper for status text
def get_cooling_status(val: float) -> tuple[str, str]:
    if val < 10:
        return "Idle / Off", "#94A3B8"
    elif val < 40:
        return "Low Cooling", "#38BDF8"
    elif val < 70:
        return "Moderate Cooling", "#60A5FA"
    else:
        return "Heavy Chill", "#3B82F6"

def get_heater_status(val: float) -> tuple[str, str]:
    if val < 10:
        return "Standby / Off", "#94A3B8"
    elif val < 40:
        return "Gentle Warmth", "#FDBA74"
    elif val < 70:
        return "Active Heating", "#FB923C"
    else:
        return "Maximum Heat", "#EA580C"

def get_fan_status(val: float) -> tuple[str, str]:
    if val < 10:
        return "Off", "#94A3B8"
    elif val < 35:
        return "Whisper Quiet", "#C084FC"
    elif val < 70:
        return "Medium Breeze", "#A855F7"
    else:
        return "Turbo Airflow", "#7E22CE"

def get_eco_status(val: float) -> tuple[str, str]:
    if val < 30:
        return "High Demand", "#F87171"
    elif val < 65:
        return "Balanced", "#FBBF24"
    else:
        return "Max EcoSaver", "#34D399"

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Smart Home Climate & Energy Optimizer</div>
        <div class="hero-subtitle">
            Autonomous Mamdani Fuzzy Inference Engine for dynamic thermal comfort, multi-zone HVAC modulation, and electricity grid demand response.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Output Metric Cards
c1, c2, c3, c4 = st.columns(4)

ac_val = result.crisp_outputs["AC_Cooling"]
ac_status, ac_color = get_cooling_status(ac_val)
with c1:
    st.markdown(
        f"""
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">❄️ AC Cooling Power</div>
            <div class="kpi-val">{ac_val:.1f}%</div>
            <div class="kpi-status" style="background: rgba(56, 189, 248, 0.2); color: {ac_color};">{ac_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

ht_val = result.crisp_outputs["Heater"]
ht_status, ht_color = get_heater_status(ht_val)
with c2:
    st.markdown(
        f"""
        <div class="kpi-card kpi-orange">
            <div class="kpi-label">🔥 Heating Power</div>
            <div class="kpi-val">{ht_val:.1f}%</div>
            <div class="kpi-status" style="background: rgba(251, 146, 60, 0.2); color: {ht_color};">{ht_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fn_val = result.crisp_outputs["Fan_Speed"]
fn_status, fn_color = get_fan_status(fn_val)
with c3:
    st.markdown(
        f"""
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">💨 Fan Speed</div>
            <div class="kpi-val">{fn_val:.1f}%</div>
            <div class="kpi-status" style="background: rgba(168, 85, 247, 0.2); color: {fn_color};">{fn_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

ec_val = result.crisp_outputs["Eco_Index"]
ec_status, ec_color = get_eco_status(ec_val)
with c4:
    st.markdown(
        f"""
        <div class="kpi-card kpi-green">
            <div class="kpi-label">🌱 Eco Savings Index</div>
            <div class="kpi-val">{ec_val:.1f}%</div>
            <div class="kpi-status" style="background: rgba(52, 211, 153, 0.2); color: {ec_color};">{ec_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Live Decision & Math Trace",
    "🌐 3D Fuzzy Control Surfaces",
    "📊 Batch Scenario Comparison",
    "📖 Fuzzy Theory & Architecture",
])

# =============================================================================
# TAB 1: LIVE DECISION & MATH TRACE
# =============================================================================
with tab1:
    st.markdown("#### 1. Active Fuzzy Rules & Mamdani Firing Strengths")
    active_rules = result.active_rules

    if active_rules:
        rule_rows = []
        for r in active_rules:
            strength_pct = r.firing_strength * 100
            rule_rows.append({
                "Rule #": r.index,
                "Firing Strength (α)": f"{r.firing_strength:.3f} ({strength_pct:.1f}%)",
                "Natural Language Rule": r.description,
                "Consequents": ", ".join(r.consequents),
            })
        df_rules = pd.DataFrame(rule_rows)
        st.dataframe(df_rules, use_container_width=True, hide_index=True)
    else:
        st.warning("No specific rules fired for this combination. Fallback neutral state applied.")

    st.markdown("---")
    st.markdown("#### 2. Fuzzification Breakdown (Membership Degrees μ)")
    st.caption("Inspect how crisp numerical sensor values map into overlapping linguistic terms:")

    col_mf1, col_mf2 = st.columns(2)
    with col_mf1:
        fig_temp = plot_linguistic_variable(engine.inputs["Temperature"], input_temp)
        st.plotly_chart(fig_temp, use_container_width=True)

        fig_occ = plot_linguistic_variable(engine.inputs["Occupancy"], input_occ)
        st.plotly_chart(fig_occ, use_container_width=True)

    with col_mf2:
        fig_hum = plot_linguistic_variable(engine.inputs["Humidity"], input_humidity)
        st.plotly_chart(fig_hum, use_container_width=True)

        fig_tar = plot_linguistic_variable(engine.inputs["ElectricityTariff"], input_tariff)
        st.plotly_chart(fig_tar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Defuzzification Integration & Aggregated Fuzzy Sets")
    st.caption(f"Visualizing the union of clipped consequents and the crisp defuzzification center ({defuzz_method.upper()}):")

    col_out1, col_out2 = st.columns(2)
    with col_out1:
        fig_ac = plot_defuzzification(
            result.output_universes["AC_Cooling"],
            result.output_distributions["AC_Cooling"],
            result.crisp_outputs["AC_Cooling"],
            "AC Cooling Power",
            "%",
            defuzz_method,
        )
        st.plotly_chart(fig_ac, use_container_width=True)

        fig_fan = plot_defuzzification(
            result.output_universes["Fan_Speed"],
            result.output_distributions["Fan_Speed"],
            result.crisp_outputs["Fan_Speed"],
            "Fan Speed",
            "%",
            defuzz_method,
        )
        st.plotly_chart(fig_fan, use_container_width=True)

    with col_out2:
        fig_ht = plot_defuzzification(
            result.output_universes["Heater"],
            result.output_distributions["Heater"],
            result.crisp_outputs["Heater"],
            "Heater Power",
            "%",
            defuzz_method,
        )
        st.plotly_chart(fig_ht, use_container_width=True)

        fig_eco = plot_defuzzification(
            result.output_universes["Eco_Index"],
            result.output_distributions["Eco_Index"],
            result.crisp_outputs["Eco_Index"],
            "Eco Savings Index",
            "%",
            defuzz_method,
        )
        st.plotly_chart(fig_eco, use_container_width=True)


# =============================================================================
# TAB 2: 3D FUZZY CONTROL SURFACES
# =============================================================================
with tab2:
    st.markdown("#### 🌐 Interactive 3D Control Surface")
    st.caption(
        "Fuzzy inference produces smooth, continuous, nonlinear control surfaces without the abrupt step discontinuities of classical if-else statements."
    )

    cs_col1, cs_col2, cs_col3 = st.columns(3)
    with cs_col1:
        x_axis = st.selectbox("X-Axis Input", options=["Temperature", "Humidity", "Occupancy", "ElectricityTariff"], index=0)
    with cs_col2:
        y_axis = st.selectbox("Y-Axis Input", options=["Humidity", "Occupancy", "ElectricityTariff", "Temperature"], index=1)
    with cs_col3:
        target_out = st.selectbox("Target Output Surface", options=["AC_Cooling", "Heater", "Fan_Speed", "Eco_Index"], index=0)

    if x_axis == y_axis:
        st.error("Please pick two distinct input variables for the X and Y axes.")
    else:
        with st.spinner("Generating 3D surface mesh..."):
            fig_3d = plot_3d_control_surface(
                engine=engine,
                x_var_name=x_axis,
                y_var_name=y_axis,
                output_var_name=target_out,
                fixed_inputs=current_inputs,
                grid_size=25,
            )
            st.plotly_chart(fig_3d, use_container_width=True)


# =============================================================================
# TAB 3: BATCH SCENARIO COMPARISON
# =============================================================================
with tab3:
    st.markdown("#### 📊 Real-World Scenario Comparative Evaluation")
    st.caption("Side-by-side evaluation of all preset scenarios to observe how the fuzzy system adapts autonomously across diverse climate and occupancy conditions:")

    scenario_results = []
    for s_name, s_data in PRESET_SCENARIOS.items():
        s_res = engine.compute(s_data["inputs"], defuzz_method=defuzz_method)
        scenario_results.append({
            "Scenario": s_name,
            "Temp (°C)": s_data["inputs"]["Temperature"],
            "Humidity (%)": s_data["inputs"]["Humidity"],
            "People": s_data["inputs"]["Occupancy"],
            "Tariff": s_data["inputs"]["ElectricityTariff"],
            "AC Cooling (%)": s_res.crisp_outputs["AC_Cooling"],
            "Heater (%)": s_res.crisp_outputs["Heater"],
            "Fan Speed (%)": s_res.crisp_outputs["Fan_Speed"],
            "Eco Index (%)": s_res.crisp_outputs["Eco_Index"],
            "Active Rules": len(s_res.active_rules),
        })

    df_scenarios = pd.DataFrame(scenario_results)
    st.dataframe(
        df_scenarios.style.background_gradient(subset=["AC Cooling (%)"], cmap="Blues")
        .background_gradient(subset=["Heater (%)"], cmap="Oranges")
        .background_gradient(subset=["Fan Speed (%)"], cmap="Purples")
        .background_gradient(subset=["Eco Index (%)"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )

    # Radar Chart Comparison
    st.markdown("##### 🕸️ Multi-Scenario System Modulation Radar")
    fig_radar = go.Figure()
    categories = ["AC Cooling", "Heater", "Fan Speed", "Eco Index"]

    for sc in scenario_results[:4]:  # Top 4 scenarios
        vals = [
            sc["AC Cooling (%)"],
            sc["Heater (%)"],
            sc["Fan Speed (%)"],
            sc["Eco Index (%)"],
        ]
        vals.append(vals[0])  # close loop
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill="toself",
                name=sc["Scenario"].split(" ")[1],
            )
        )

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8", gridcolor="#334155"),
            angularaxis=dict(color="#E2E8F0", gridcolor="#334155"),
            bgcolor="rgba(15, 23, 42, 0.4)",
        ),
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        template="plotly_dark",
        height=450,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# =============================================================================
# TAB 4: FUZZY THEORY & ARCHITECTURE
# =============================================================================
with tab4:
    st.markdown("### 🧠 The Science Behind Fuzzy Logic Decision Systems")
    st.markdown(
        """
        Conventional computing uses **Boolean (crisp) logic**, where propositions are strictly either `True (1)` or `False (0)`.
        However, real-world human reasoning and physical phenomena are inherently vague, continuous, and imprecise.
        
        Introduced by **Lotfi A. Zadeh (1965)**, **Fuzzy Logic** allows intermediate degrees of truth $\\mu \\in [0, 1]$.
        """
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(
            r"""
            #### 1. Fuzzification & Membership Functions $\mu(x)$
            Maps crisp physical sensor readings (e.g., $T = 27.5^\circ\text{C}$) into fuzzy linguistic values:
            
            - **Triangular Membership Function**:
            $$\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$
            
            - **Trapezoidal Membership Function**:
            $$\mu(x; a, b, c, d) = \max\left(0, \min\left(\frac{x - a}{b - a}, 1, \frac{d - x}{d - c}\right)\right)$$
            
            #### 2. Fuzzy Operators & Mamdani Rule Implication
            Rules take the form:
            $$\text{IF } X_1 \text{ is } A_1 \text{ AND } X_2 \text{ is } A_2 \text{ THEN } Y \text{ is } B$$
            
            - **T-Norm (Fuzzy AND)**: Minimum operator:
            $$\alpha = \min(\mu_{A_1}(x_1), \mu_{A_2}(x_2))$$
            - **Implication (Clipping)**:
            $$\mu_B'(y) = \min(\alpha, \mu_B(y))$$
            """
        )

    with t_col2:
        st.markdown(
            r"""
            #### 3. Aggregation (Fuzzy OR)
            Combines the clipped fuzzy outputs of all fired rules using the Maximum S-Norm operator:
            $$\mu_{\text{agg}}(y) = \max_{k=1}^N \mu_{B'_k}(y)$$
            
            #### 4. Defuzzification (Centroid / Center of Gravity)
            Converts the final aggregated fuzzy region back into an actionable, crisp engineering command:
            $$y^* = \frac{\int y \cdot \mu_{\text{agg}}(y) \, dy}{\int \mu_{\text{agg}}(y) \, dy} \approx \frac{\sum_{i=1}^M y_i \cdot \mu_{\text{agg}}(y_i)}{\sum_{i=1}^M \mu_{\text{agg}}(y_i)}$$
            
            #### 💡 Why Fuzzy Outperforms Classical Bang-Bang Thermostats
            - **Eliminates Actuator Chattering**: No harsh cycling on/off around a single setpoint.
            - **Multi-Objective Optimization**: Simultaneously balances human thermal comfort, humidity index, occupant count, and electricity grid tariff pricing.
            - **Complete Explainability**: Every decision can be traced directly to human-interpretable linguistic rules rather than an opaque black-box neural weight matrix.
            """
        )
