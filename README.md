# 🌡️ Smart Home Climate & Energy Optimizer (Fuzzy Logic Decision System)

An intelligent, explainable **Mamdani Fuzzy Inference System** built in Python with an interactive **Streamlit Dashboard**, high-performance **CLI**, and real-time **Plotly visualizations** (membership curves, defuzzification centroids, and 3D control surfaces).

---

## 🌟 Key Features

- **Pure, Explainable Fuzzy Logic Engine**: Built from first principles using NumPy. Zero opaque dependencies, 100% mathematically transparent.
- **Comprehensive Membership Functions**: Triangular (`trimf`), Trapezoidal (`trapmf`), Gaussian (`gaussmf`), and Sigmoidal (`sigmf`).
- **Multiple Defuzzification Strategies**:
  - Center of Gravity / Centroid (COG)
  - Bisector of Area (BOA)
  - Mean of Maxima (MOM)
  - Smallest of Maxima (SOM) / Largest of Maxima (LOM)
- **Multi-Objective Smart Home Model**:
  - **Inputs (Antecedents)**: Temperature (°C), Relative Humidity (%), Room Occupancy (headcount), Electricity Grid Tariff Tier (1=Off-peak to 5=Critical peak).
  - **Outputs (Consequents)**: AC Cooling Power (%), Heater Power (%), Fan Speed (%), Eco Savings Index (%).
  - **Knowledge Base**: 28+ domain-expert Mamdani fuzzy rules accounting for thermal comfort, occupant heat dissipation, humidity dampening, and peak tariff demand shedding.
- **Modern Interactive Web App (Streamlit + Plotly)**:
  - Real-time sliders & KPI cards with dynamic status badges.
  - Interactive membership function charts highlighting live input membership degrees $\mu$.
  - Shaded aggregated defuzzification curve visualizer.
  - Interactive 3D Fuzzy Control Surface explorer with orbit rotation.
  - Batch Scenario Comparison & Radar analysis across real-world situations.
- **Fast CLI & Speed Benchmark**: Runs nearly **1,000 full Mamdani inference cycles per second** (~1.0 ms latency per decision).

---

## 📁 Project Structure

```
fuzzylogic/
├── fuzzy_engine/             # Core reusable Fuzzy Inference Engine
│   ├── __init__.py           # Package exports
│   ├── membership.py         # Triangular, Trapezoidal, Gaussian, Sigmoid MFs
│   ├── variable.py           # LinguisticVariable and universe discretization
│   ├── rule.py               # FuzzyRule, AntecedentClause, ConsequentClause
│   ├── defuzzifier.py        # Centroid, Bisector, MOM, SOM, LOM algorithms
│   └── inference.py          # MamdaniEngine execution & mathematical tracer
├── smart_home/               # Smart Home Domain Model
│   ├── __init__.py           # Package exports
│   ├── model.py              # Variables, terms, and 28+ Mamdani expert rules
│   ├── scenarios.py          # Preset real-world scenarios (Empty House, Party, etc.)
│   └── visualizer.py         # Plotly membership, defuzzification & 3D surface charts
├── tests/
│   └── test_fuzzy_engine.py  # Unit tests with 100% test pass rate
├── app.py                    # Streamlit Interactive Web Application
├── run_cli.py                # Standalone Command-Line Interface & Benchmark
├── requirements.txt          # Python dependencies
└── README.md                 # Complete system documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run via Command-Line Interface (CLI)
You can run immediate one-off inferences or test preset scenarios directly in your terminal:

```bash
# Manual sensor inputs
python run_cli.py --temp 32.5 --humidity 75 --occupancy 4 --tariff 4

# Load a preset scenario
python run_cli.py --scenario "Summer Evening Party"

# List all available scenarios
python run_cli.py --list-scenarios

# Run inference speed benchmark
python run_cli.py --benchmark
```

### 4. Run Unit Tests
```bash
python -m unittest discover tests
```

---

## 📐 Mathematical Formulation

### 1. Fuzzification
A crisp sensor input $x_0$ is mapped into a fuzzy degree of truth $\mu_A(x_0) \in [0, 1]$ across linguistic terms.
For a Triangular membership function with parameters $(a, b, c)$:
$$\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

### 2. Fuzzy Rule Implication (Mamdani Min T-Norm)
For rule $k$: `IF X1 is A1 AND X2 is A2 THEN Y is B`:
$$\alpha_k = \min(\mu_{A_1}(x_1), \mu_{A_2}(x_2))$$
$$\mu_{B'_k}(y) = \min(\alpha_k \cdot w_k, \mu_B(y))$$

### 3. Output Aggregation (Max S-Norm)
The combined fuzzy output set across all $N$ rules is:
$$\mu_{\text{agg}}(y) = \max_{k=1}^N \mu_{B'_k}(y)$$

### 4. Defuzzification (Center of Gravity / Centroid)
The continuous aggregated fuzzy set is defuzzified into a single crisp control output $y^*$:
$$y^* = \frac{\int_{y_{\min}}^{y_{\max}} y \cdot \mu_{\text{agg}}(y) \, dy}{\int_{y_{\min}}^{y_{\max}} \mu_{\text{agg}}(y) \, dy} \approx \frac{\sum_{i=1}^M y_i \cdot \mu_{\text{agg}}(y_i)}{\sum_{i=1}^M \mu_{\text{agg}}(y_i)}$$

---

## 💡 Why Fuzzy Logic for HVAC & Energy Optimization?

1. **No Actuator Chattering**: Unlike bang-bang thermostats that rapidly turn compressors on and off when oscillating around 24°C, fuzzy logic smoothly modulates compressor power and fan speed.
2. **Coupled Multi-Variable Optimization**: Considers human heat emission, relative humidity (heat index effect), and grid electricity pricing simultaneously.
3. **Full Explainability**: Unlike deep neural networks, every decision is auditable and human-understandable.
