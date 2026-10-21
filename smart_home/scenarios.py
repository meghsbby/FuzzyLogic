"""
Preset real-world scenarios for the Smart Home Climate & Energy Optimizer.
"""

from typing import Dict, Any

PRESET_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "🌴 Hot Summer Afternoon (Empty House)": {
        "inputs": {"Temperature": 33.5, "Humidity": 60.0, "Occupancy": 0.0, "ElectricityTariff": 4.5},
        "description": "The house is empty during peak afternoon heat and expensive electricity tariff. The system applies energy setbacks.",
    },
    "🎉 Summer Evening Party (High Occupancy)": {
        "inputs": {"Temperature": 29.5, "Humidity": 80.0, "Occupancy": 8.0, "ElectricityTariff": 3.0},
        "description": "8 people indoors with high humidity and heat. System activates high AC cooling and turbo fan to maintain comfort.",
    },
    "❄️ Freezing Winter Morning": {
        "inputs": {"Temperature": 11.5, "Humidity": 35.0, "Occupancy": 2.0, "ElectricityTariff": 1.5},
        "description": "Chilly indoor temperatures early morning with cheap off-peak electricity. System powers heating actively.",
    },
    "🌸 Mild Spring Day (Ideal Comfort)": {
        "inputs": {"Temperature": 23.0, "Humidity": 48.0, "Occupancy": 1.0, "ElectricityTariff": 2.0},
        "description": "Near-perfect ambient conditions. System powers down compressors and runs quiet fan circulation for max EcoSaver score.",
    },
    "⚡ Extreme Heatwave during Peak Tariff": {
        "inputs": {"Temperature": 38.0, "Humidity": 72.0, "Occupancy": 4.0, "ElectricityTariff": 5.0},
        "description": "Dangerous outdoor/indoor heat coupled with maximum peak pricing tier. Balances life safety cooling with demand response.",
    },
    "🛋️ Movie Night with Family": {
        "inputs": {"Temperature": 26.5, "Humidity": 55.0, "Occupancy": 4.0, "ElectricityTariff": 3.0},
        "description": "4 people resting in living room. Moderate cooling and medium quiet airflow.",
    },
}
