"""
Smart Home Climate & Energy Optimizer - Command Line Interface (CLI).

Run directly from terminal:
    python run_cli.py --temp 31.5 --humidity 70 --occupancy 4 --tariff 3
    python run_cli.py --scenario "🎉 Summer Evening Party (High Occupancy)"
    python run_cli.py --benchmark
"""

import argparse
import sys
import time

# Ensure safe unicode encoding on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from smart_home import create_smart_climate_system, PRESET_SCENARIOS


def print_banner():
    banner = """
  +===================================================================+
  |    SMART HOME CLIMATE & ENERGY OPTIMIZER (FUZZY LOGIC ENGINE)     |
  |               Mamdani Fuzzy Inference System                      |
  +===================================================================+
    """
    print(banner)


def format_bar(val: float, max_val: float = 100.0, length: int = 25) -> str:
    filled = int(round((val / max_val) * length))
    filled = max(0, min(length, filled))
    return "#" * filled + "-" * (length - filled)


def run_single_inference(engine, inputs: dict, defuzz: str = "centroid"):
    result = engine.compute(inputs, defuzz_method=defuzz)

    print("\n" + "=" * 65)
    print(" 📥 INPUT CONDITIONS")
    print("=" * 65)
    for k, v in result.crisp_inputs.items():
        unit = engine.inputs[k].unit
        print(f"  • {k:<18}: {v:>6.1f} {unit}")

    print("\n" + "=" * 65)
    print(" 🧠 ACTIVE FUZZY RULES (MAMDANI FIRING STRENGTH)")
    print("=" * 65)
    active = result.active_rules
    if not active:
        print("  (No rules fired. Default neutral output applied.)")
    else:
        for r in active:
            bar = format_bar(r.firing_strength * 100, 100, length=12)
            print(f"  [{r.firing_strength:.2f}] {bar} Rule #{r.index}: {r.description}")

    print("\n" + "=" * 65)
    print(f" 🎯 DEFUZZIFIED CRISP OUTPUT DECISIONS ({defuzz.upper()})")
    print("=" * 65)
    for k, v in result.crisp_outputs.items():
        unit = engine.outputs[k].unit
        bar = format_bar(v, 100.0, length=20)
        print(f"  • {k:<15}: {v:>5.1f} {unit:<3} |{bar}|")
    print("=" * 65 + "\n")


def benchmark(engine, iterations: int = 5000):
    print(f"\n⚡ Running benchmark with {iterations} continuous inference cycles...")
    test_inputs = {"Temperature": 28.5, "Humidity": 65.0, "Occupancy": 3.0, "ElectricityTariff": 3.0}

    t0 = time.perf_counter()
    for _ in range(iterations):
        engine.compute(test_inputs)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    ops_sec = iterations / elapsed
    print(f"✅ Completed in {elapsed:.3f} seconds ({ops_sec:,.0f} inferences/sec)")
    print(f"   Average latency per decision: {(elapsed / iterations) * 1000:.3f} ms\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Home Climate Fuzzy Logic Optimizer")
    parser.add_argument("--temp", type=float, default=27.0, help="Indoor Temperature in °C (10 - 40)")
    parser.add_argument("--humidity", type=float, default=60.0, help="Indoor Humidity in % (10 - 95)")
    parser.add_argument("--occupancy", type=float, default=2.0, help="Occupancy in people (0 - 10)")
    parser.add_argument("--tariff", type=float, default=3.0, help="Grid Tariff Tier (1=Off-Peak, 5=Peak)")
    parser.add_argument("--defuzz", type=str, default="centroid", choices=["centroid", "bisector", "mom", "som", "lom"])
    parser.add_argument("--scenario", type=str, default=None, help="Preset scenario name")
    parser.add_argument("--list-scenarios", action="store_true", help="List all available preset scenarios")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark fuzzy inference engine speed")

    args = parser.parse_args()
    print_banner()

    engine = create_smart_climate_system()

    if args.list_scenarios:
        print("\nAvailable Preset Scenarios:")
        for name, data in PRESET_SCENARIOS.items():
            print(f"  • {name}")
            print(f"    {data['description']}")
            print(f"    Inputs: {data['inputs']}\n")
        return

    if args.benchmark:
        benchmark(engine)
        return

    if args.scenario:
        # Match scenario by prefix or exact
        matched_key = next((k for k in PRESET_SCENARIOS if args.scenario.lower() in k.lower()), None)
        if not matched_key:
            print(f"❌ Scenario '{args.scenario}' not found. Use --list-scenarios to view options.")
            sys.exit(1)
        scenario_data = PRESET_SCENARIOS[matched_key]
        print(f"📌 Running Scenario: {matched_key}")
        print(f"   {scenario_data['description']}")
        run_single_inference(engine, scenario_data["inputs"], defuzz=args.defuzz)
        return

    inputs = {
        "Temperature": args.temp,
        "Humidity": args.humidity,
        "Occupancy": args.occupancy,
        "ElectricityTariff": args.tariff,
    }
    run_single_inference(engine, inputs, defuzz=args.defuzz)


if __name__ == "__main__":
    main()
