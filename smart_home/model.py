"""
Smart Home Climate & Energy Optimizer - Fuzzy System Model.

Defines input variables (Temperature, Humidity, Occupancy, Electricity Tariff),
output variables (AC Cooling Power, Heater Power, Fan Speed, Eco Savings Index),
and a comprehensive 28+ rule Mamdani knowledge base.
"""

from typing import Dict, List
from fuzzy_engine import (
    LinguisticVariable,
    TriangularMF,
    TrapezoidalMF,
    Rule,
    FuzzyRule,
    MamdaniEngine,
)


def create_smart_climate_system() -> MamdaniEngine:
    """Builds and returns the configured Mamdani smart climate optimization engine."""
    engine = MamdaniEngine(name="SmartHomeClimateOptimizer", default_defuzz="centroid")

    # =========================================================================
    # 1. INPUT VARIABLES
    # =========================================================================

    # Temperature (°C) [10 to 40]
    temp = LinguisticVariable("Temperature", (10.0, 40.0), unit="°C", description="Indoor ambient temperature")
    temp.add_term("Freezing", TrapezoidalMF(10.0, 10.0, 12.0, 16.0))
    temp.add_term("Cold", TriangularMF(14.0, 18.0, 22.0))
    temp.add_term("Comfortable", TriangularMF(20.0, 23.5, 26.5))
    temp.add_term("Warm", TriangularMF(24.5, 28.0, 32.0))
    temp.add_term("Hot", TrapezoidalMF(30.0, 34.0, 40.0, 40.0))
    engine.add_input(temp)

    # Relative Humidity (%) [10 to 95]
    humidity = LinguisticVariable("Humidity", (10.0, 95.0), unit="%", description="Indoor relative humidity")
    humidity.add_term("Dry", TrapezoidalMF(10.0, 10.0, 25.0, 40.0))
    humidity.add_term("Optimal", TriangularMF(30.0, 50.0, 68.0))
    humidity.add_term("Humid", TrapezoidalMF(60.0, 75.0, 95.0, 95.0))
    engine.add_input(humidity)

    # Occupancy (people) [0 to 10]
    occupancy = LinguisticVariable("Occupancy", (0.0, 10.0), unit="people", description="Room head count")
    occupancy.add_term("Empty", TrapezoidalMF(0.0, 0.0, 0.2, 0.9))
    occupancy.add_term("Low", TriangularMF(0.5, 2.0, 3.5))
    occupancy.add_term("Medium", TriangularMF(2.5, 4.5, 6.5))
    occupancy.add_term("High", TrapezoidalMF(5.5, 7.5, 10.0, 10.0))
    engine.add_input(occupancy)

    # Electricity Tariff Tier [1 to 5] (1=Super Off-Peak, 3=Standard, 5=Peak Hours)
    tariff = LinguisticVariable("ElectricityTariff", (1.0, 5.0), unit="Tier", description="Current grid electricity cost tier")
    tariff.add_term("Cheap", TrapezoidalMF(1.0, 1.0, 1.8, 2.5))
    tariff.add_term("Normal", TriangularMF(2.0, 3.0, 4.0))
    tariff.add_term("Peak", TrapezoidalMF(3.5, 4.5, 5.0, 5.0))
    engine.add_input(tariff)

    # =========================================================================
    # 2. OUTPUT VARIABLES
    # =========================================================================

    # AC Cooling Power (%) [0 to 100]
    ac_power = LinguisticVariable("AC_Cooling", (0.0, 100.0), unit="%", description="Air conditioner compressor & cooling load")
    ac_power.add_term("Off", TrapezoidalMF(0.0, 0.0, 4.0, 15.0))
    ac_power.add_term("Low", TriangularMF(10.0, 25.0, 45.0))
    ac_power.add_term("Moderate", TriangularMF(35.0, 55.0, 75.0))
    ac_power.add_term("High", TriangularMF(65.0, 80.0, 95.0))
    ac_power.add_term("Max", TrapezoidalMF(85.0, 95.0, 100.0, 100.0))
    engine.add_output(ac_power)

    # Heater Power (%) [0 to 100]
    heater = LinguisticVariable("Heater", (0.0, 100.0), unit="%", description="Heating system power")
    heater.add_term("Off", TrapezoidalMF(0.0, 0.0, 4.0, 15.0))
    heater.add_term("Low", TriangularMF(10.0, 25.0, 45.0))
    heater.add_term("Moderate", TriangularMF(35.0, 55.0, 75.0))
    heater.add_term("High", TrapezoidalMF(65.0, 85.0, 100.0, 100.0))
    engine.add_output(heater)

    # Fan Speed (%) [0 to 100]
    fan = LinguisticVariable("Fan_Speed", (0.0, 100.0), unit="%", description="Circulation fan speed")
    fan.add_term("Off", TrapezoidalMF(0.0, 0.0, 4.0, 12.0))
    fan.add_term("Quiet", TriangularMF(10.0, 25.0, 42.0))
    fan.add_term("Medium", TriangularMF(35.0, 55.0, 75.0))
    fan.add_term("High", TriangularMF(65.0, 80.0, 95.0))
    fan.add_term("Turbo", TrapezoidalMF(85.0, 95.0, 100.0, 100.0))
    engine.add_output(fan)

    # Eco Savings Score (%) [0 to 100] (0 = High Energy Consumption, 100 = Maximum Green Saving)
    eco_score = LinguisticVariable("Eco_Index", (0.0, 100.0), unit="%", description="Energy efficiency and savings score")
    eco_score.add_term("Performance", TrapezoidalMF(0.0, 0.0, 15.0, 35.0))
    eco_score.add_term("Balanced", TriangularMF(30.0, 50.0, 70.0))
    eco_score.add_term("EcoSaver", TrapezoidalMF(65.0, 85.0, 100.0, 100.0))
    engine.add_output(eco_score)

    # =========================================================================
    # 3. RULE BASE (30 Domain-Expert Rules)
    # =========================================================================
    rules: List[FuzzyRule] = [
        # --- A. Unoccupied / Empty Home Conservation Rules ---
        Rule(
            antecedents=[("Occupancy", "Empty"), ("Temperature", "Comfortable")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Off"), ("Fan_Speed", "Off"), ("Eco_Index", "EcoSaver")],
            description="Empty home at comfortable temp -> Systems Off, EcoSaver",
        ),
        Rule(
            antecedents=[("Occupancy", "Empty"), ("Temperature", "Warm")],
            consequents=[("AC_Cooling", "Low"), ("Heater", "Off"), ("Fan_Speed", "Quiet"), ("Eco_Index", "EcoSaver")],
            description="Empty home warm -> Minimal cooling setback",
        ),
        Rule(
            antecedents=[("Occupancy", "Empty"), ("Temperature", "Hot")],
            consequents=[("AC_Cooling", "Moderate"), ("Heater", "Off"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            description="Empty home hot -> Moderate setback to prevent heat damage",
        ),
        Rule(
            antecedents=[("Occupancy", "Empty"), ("Temperature", "Cold")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Low"), ("Fan_Speed", "Off"), ("Eco_Index", "EcoSaver")],
            description="Empty home cold -> Minimal heater setback",
        ),
        Rule(
            antecedents=[("Occupancy", "Empty"), ("Temperature", "Freezing")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Moderate"), ("Fan_Speed", "Quiet"), ("Eco_Index", "Balanced")],
            description="Empty home freezing -> Pipe freeze protection",
        ),

        # --- B. Comfortable Temperature Base Rules ---
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Comfortable"), ("Humidity", "Optimal")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Off"), ("Fan_Speed", "Quiet"), ("Eco_Index", "EcoSaver")],
            description="Low occupancy, perfect climate -> Gentle air circulation",
        ),
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Comfortable"), ("Humidity", "Humid")],
            consequents=[("AC_Cooling", "Low"), ("Heater", "Off"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            description="Comfortable but humid -> Low AC dehumidification",
        ),
        Rule(
            antecedents=[("Occupancy", "Medium"), ("Temperature", "Comfortable"), ("Humidity", "Optimal")],
            consequents=[("AC_Cooling", "Low"), ("Heater", "Off"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            description="Medium occupancy comfortable -> Fresh air airflow",
        ),
        Rule(
            antecedents=[("Occupancy", "High"), ("Temperature", "Comfortable")],
            consequents=[("AC_Cooling", "Moderate"), ("Heater", "Off"), ("Fan_Speed", "High"), ("Eco_Index", "Balanced")],
            description="High occupancy comfortable -> Counter body heat dissipation",
        ),

        # --- C. Warm & Hot Cooling Operations ---
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Warm"), ("Humidity", "Optimal")],
            consequents=[("AC_Cooling", "Low"), ("Heater", "Off"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            description="Warm & optimal humidity -> Light cooling & breeze",
        ),
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Warm"), ("Humidity", "Humid")],
            consequents=[("AC_Cooling", "Moderate"), ("Heater", "Off"), ("Fan_Speed", "High"), ("Eco_Index", "Balanced")],
            description="Warm & humid -> Moderate cooling & high fan speed",
        ),
        Rule(
            antecedents=[("Occupancy", "Medium"), ("Temperature", "Warm")],
            consequents=[("AC_Cooling", "Moderate"), ("Heater", "Off"), ("Fan_Speed", "High"), ("Eco_Index", "Balanced")],
            description="Warm with family present -> Active cooling",
        ),
        Rule(
            antecedents=[("Occupancy", "High"), ("Temperature", "Warm")],
            consequents=[("AC_Cooling", "High"), ("Heater", "Off"), ("Fan_Speed", "High"), ("Eco_Index", "Performance")],
            description="Warm with crowd -> High cooling capacity",
        ),
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Hot"), ("Humidity", "Dry")],
            consequents=[("AC_Cooling", "Moderate"), ("Heater", "Off"), ("Fan_Speed", "High"), ("Eco_Index", "Balanced")],
            description="Hot & dry -> Moderate AC with air movement",
        ),
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Hot"), ("Humidity", "Humid")],
            consequents=[("AC_Cooling", "High"), ("Heater", "Off"), ("Fan_Speed", "Turbo"), ("Eco_Index", "Performance")],
            description="Hot & sticky humid -> Heavy AC cooling & Turbo fan",
        ),
        Rule(
            antecedents=[("Occupancy", "Medium"), ("Temperature", "Hot")],
            consequents=[("AC_Cooling", "High"), ("Heater", "Off"), ("Fan_Speed", "Turbo"), ("Eco_Index", "Performance")],
            description="Hot & occupied -> Fast cooling response",
        ),
        Rule(
            antecedents=[("Occupancy", "High"), ("Temperature", "Hot")],
            consequents=[("AC_Cooling", "Max"), ("Heater", "Off"), ("Fan_Speed", "Turbo"), ("Eco_Index", "Performance")],
            description="Scorching hot with gathering -> Maximum cooling power",
        ),

        # --- D. Cold & Freezing Heating Operations ---
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Cold"), ("Humidity", "Optimal")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Moderate"), ("Fan_Speed", "Quiet"), ("Eco_Index", "Balanced")],
            description="Cold & low occupancy -> Moderate gentle heating",
        ),
        Rule(
            antecedents=[("Occupancy", "Low"), ("Temperature", "Cold"), ("Humidity", "Dry")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Moderate"), ("Fan_Speed", "Quiet"), ("Eco_Index", "Balanced")],
            description="Cold & dry -> Warm up room comfortably",
        ),
        Rule(
            antecedents=[("Occupancy", "Medium"), ("Temperature", "Cold")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Low"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            description="Cold with medium crowd -> Low heating (body heat helps)",
        ),
        Rule(
            antecedents=[("Occupancy", "High"), ("Temperature", "Cold")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Low"), ("Fan_Speed", "High"), ("Eco_Index", "EcoSaver")],
            description="Cold with large crowd -> High body thermal gain, minimal heating",
        ),
        Rule(
            antecedents=[("Temperature", "Freezing"), ("Occupancy", "Low")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "High"), ("Fan_Speed", "Medium"), ("Eco_Index", "Performance")],
            description="Freezing temperature -> High power heating boost",
        ),
        Rule(
            antecedents=[("Temperature", "Freezing"), ("Occupancy", "Medium")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "High"), ("Fan_Speed", "Medium"), ("Eco_Index", "Performance")],
            description="Freezing with people -> Maintain warmth urgently",
        ),

        # --- E. Grid Electricity Tariff & Peak Demand Management ---
        Rule(
            antecedents=[("ElectricityTariff", "Peak"), ("Temperature", "Warm"), ("Occupancy", "Low")],
            consequents=[("AC_Cooling", "Low"), ("Fan_Speed", "High"), ("Eco_Index", "EcoSaver")],
            weight=1.0,
            description="Peak tariff hours -> Prioritize fan breeze over heavy AC",
        ),
        Rule(
            antecedents=[("ElectricityTariff", "Peak"), ("Temperature", "Cold"), ("Occupancy", "Low")],
            consequents=[("Heater", "Low"), ("Fan_Speed", "Quiet"), ("Eco_Index", "EcoSaver")],
            weight=1.0,
            description="Peak tariff hours -> Economical heating setback",
        ),
        Rule(
            antecedents=[("ElectricityTariff", "Peak"), ("Occupancy", "Empty")],
            consequents=[("AC_Cooling", "Off"), ("Heater", "Off"), ("Fan_Speed", "Off"), ("Eco_Index", "EcoSaver")],
            weight=1.0,
            description="Peak tariff with empty house -> Absolute power shedding",
        ),
        Rule(
            antecedents=[("ElectricityTariff", "Cheap"), ("Temperature", "Warm"), ("Occupancy", "Low")],
            consequents=[("AC_Cooling", "Moderate"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            weight=0.9,
            description="Cheap tariff -> Pre-cool room economically",
        ),
        Rule(
            antecedents=[("ElectricityTariff", "Cheap"), ("Temperature", "Cold"), ("Occupancy", "Low")],
            consequents=[("Heater", "Moderate"), ("Fan_Speed", "Medium"), ("Eco_Index", "Balanced")],
            weight=0.9,
            description="Cheap tariff -> Pre-heat room economically",
        ),
    ]

    engine.add_rules(rules)
    return engine
