"""
Smart Home Climate & Energy Optimizer module.
"""

from .model import create_smart_climate_system
from .scenarios import PRESET_SCENARIOS
from .visualizer import (
    plot_linguistic_variable,
    plot_defuzzification,
    plot_3d_control_surface,
)

__all__ = [
    "create_smart_climate_system",
    "PRESET_SCENARIOS",
    "plot_linguistic_variable",
    "plot_defuzzification",
    "plot_3d_control_surface",
]
