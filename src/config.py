"""
IV Batch Analyzer V5.0 - Professional Edition
Configuration Module

Handles configuration persistence and theme management.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ================= CONSTANTS =================

UNIT_MAP = {
    'Eff': '(%)', 'Voc': '(V)', 'Jsc': '(mA/cm²)',
    'FF': '(%)', 'Rs': '(Ωcm²)', 'Rsh': '(Ωcm²)'
}

COLUMN_MAPPING = {
    'Eff': ['eff', 'efficiency', 'pce', 'eta'],
    'Voc': ['voc', 'uoc', 'open_circuit_voltage'],
    'Jsc': ['jsc', 'isc', 'short_circuit_current', 'j_sc'],
    'FF': ['ff', 'fill_factor'],
    'Rs': ['rs', 'rs_light', 'series_resistance'],
    'Rsh': ['rsh', 'rsh_light', 'shunt_resistance', 'rp'],
    'ScanDir': ['scandirection', 'direction', 'scan_dir'],
    'CellName': ['cellname', 'device_id', 'sample_name', 'name', 'pixel'],
    'Voltage': ['voltage', 'v', 'bias'],  # For raw IV curves
    'Current': ['current', 'i', 'current_density', 'j']   # For raw IV curves
}

ANALYSIS_PARAMS = ['Eff', 'Voc', 'Jsc', 'FF', 'Rs', 'Rsh']
EFF_BINS = [0, 10, 15, 18, 20, 22, 24, 26, 30, 100]
EFF_BIN_LABELS = ['<10%', '10-15%', '15-18%', '18-20%', '20-22%', '22-24%', '24-26%', '26-30%', '>30%']


# ================= CONFIGURATION DATACLASSES =================

@dataclass(frozen=True)
class PlottingConfig:
    """Immutable configuration for plotting parameters."""
    colors: Tuple[str, ...] = (
        "#4DBBD5", "#E64B35", "#00A087", "#3C5488", "#F39B7F",
        "#8491B4", "#91D1C2", "#7E6148", "#B09C85"
    )
    font_family: Tuple[str, ...] = ("Arial", "Helvetica", "sans-serif")
    line_width: float = 2.0
    marker_size: int = 80
    base_figsize: Tuple[float, float] = (8, 6)
    dpi: int = 300
    show_points: bool = True


@dataclass(frozen=True)
class ThemeConfig:
    """Configuration for a plotting theme."""
    name: str
    style: str
    context: str
    palette: Any
    font_family: Tuple[str, ...]
    background_color: str
    text_color: str
    grid_color: str
    line_width: float = 2.0
    dpi: int = 300
    show_points: bool = True


# Define Themes
THEMES = {
    "Academic": ThemeConfig(
        name="Academic",
        style="ticks",
        context="paper",
        palette="deep",
        font_family=("Times New Roman", "serif"),
        background_color="white",
        text_color="black",
        grid_color="#e0e0e0",
        line_width=1.5,
        dpi=300
    ),
    "Dark": ThemeConfig(
        name="Dark",
        style="dark",
        context="talk",
        palette="bright",
        font_family=("Arial", "sans-serif"),
        background_color="#2b2b2b",
        text_color="#e0e0e0",
        grid_color="#555555",
        line_width=2.0,
        dpi=150
    ),
    "Presentation": ThemeConfig(
        name="Presentation",
        style="whitegrid",
        context="poster",
        palette="bold",
        font_family=("Arial", "sans-serif"),
        background_color="white",
        text_color="black",
        grid_color="#cccccc",
        line_width=3.0,
        dpi=200
    )
}





# Naming Patterns
NAMING_PATTERNS = {
    "Traceable": "IV_Report_{User}_{Timestamp}",
    "Chronological": "{YYYYMMDD}_{HHMM}_IV_Analysis",
    "Content-Centric": "Report_{Batch_Summary}_{RunID}",
    "Minimalist": "Analysis_{Timestamp_Short}"
}


@dataclass(frozen=True)
class AnalyzerConfig:
    """Immutable configuration for analysis logic."""
    default_initials: str = "USER"
    input_patterns: Tuple[str, ...] = (
        "*IVMeasurement*.csv", "*summary*.csv", "*result*.csv", "*data*.csv"
    )
    report_docx_name: str = "IV_Analysis_Report.docx"
    report_pptx_name: str = "IV_Analysis_Slides.pptx"
    excel_data_name: str = "IV_Processed_Data.xlsx"
    scan_direction: str = "Reverse"
    remove_duplicates: bool = True
    outlier_removal: bool = True
    outlier_removal: bool = True
    plot_theme: str = "Dark"
    naming_convention: str = "Traceable"
    enable_advanced_analysis: bool = False
    ff_threshold_for_fitting: float = 40.0
    
    # Granular Control
    champion_criteria: str = "Max Eff"  # "Max Eff" or "Max FF"
    resistance_method: str = "Slope"    # "Slope" or "Fitting"
    output_formats: Tuple[str, ...] = ("png",)  # "png", "svg", "pdf"
    report_types: Tuple[str, ...] = ("excel", "word", "pptx")
    selected_plots: Tuple[str, ...] = (
        "box", "hist", "trend", "yield", "jv_curve", "voc_jsc", 
        "combo_drivers", "resistance", "model_fitting", "hysteresis", "anomalies"
    )
    
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "Eff_Min": 0.1, "Voc_Min": 0.1, "Jsc_Min": 0.1, "FF_Min": 10.0, "FF_Max": 90.0
    })
    control_keywords: Tuple[str, ...] = ("Ref", "Ctrl", "Control", "Std", "Baseline")
    plotting: PlottingConfig = field(default_factory=PlottingConfig)


# ================= CONFIG MANAGER =================

class ConfigManager:
    """Manages configuration persistence to JSON file."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to config.json. If None, uses default location.
        """
        if config_path is None:
            self.config_path = Path.home() / ".iv_analyzer" / "config.json"
        else:
            self.config_path = Path(config_path)
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Returns:
            Dictionary with user preferences. Returns defaults if file doesn't exist or is corrupted.
        """
        defaults = {
            "last_folder": str(Path.cwd()),
            "scan_direction": "Reverse",
            "thresholds": {
                "Eff_Min": 0.1,
                "Voc_Min": 0.1,
                "Jsc_Min": 0.1,
                "FF_Min": 10.0,
                "FF_Max": 90.0
            },
            "remove_duplicates": True,
            "outlier_removal": True,
            "plot_theme": "Dark",
            "naming_convention": "Traceable",
            "enable_advanced_analysis": False,
            "ff_threshold_for_fitting": 40.0,
            "theme": "dark",
            "window_geometry": "900x700"
        }
        
        if not self.config_path.exists():
            logger.info("No config file found, using defaults")
            return defaults
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Merge with defaults (user config takes precedence)
            merged = defaults.copy()
            merged.update(user_config)
            
            logger.info(f"Loaded config from {self.config_path}")
            return merged
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return defaults
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            config: Dictionary with user preferences
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved config to {self.config_path}")
            
        except IOError as e:
            logger.error(f"Failed to save config: {e}")
    
    def get_analyzer_config(self, user_config: Dict[str, Any]) -> AnalyzerConfig:
        """
        Convert user config dictionary to AnalyzerConfig dataclass.
        
        Args:
            user_config: Dictionary with user preferences
            
        Returns:
            AnalyzerConfig instance
        """
        return AnalyzerConfig(
            scan_direction=user_config.get("scan_direction", "Reverse"),
            remove_duplicates=user_config.get("remove_duplicates", True),
            outlier_removal=user_config.get("outlier_removal", True),

            plot_theme=user_config.get("plot_theme", "Dark"),
            naming_convention=user_config.get("naming_convention", "Traceable"),
            enable_advanced_analysis=user_config.get("enable_advanced_analysis", False),
            ff_threshold_for_fitting=user_config.get("ff_threshold_for_fitting", 40.0),
            champion_criteria=user_config.get("champion_criteria", "Max Eff"),
            resistance_method=user_config.get("resistance_method", "Slope"),
            output_formats=tuple(user_config.get("output_formats", ["png"])),
            report_types=tuple(user_config.get("report_types", ["excel", "word", "pptx"])),
            selected_plots=tuple(user_config.get("selected_plots", [
                "box", "hist", "trend", "yield", "jv_curve", "voc_jsc", 
                "combo_drivers", "resistance", "model_fitting", "hysteresis", "anomalies"
            ])),
            thresholds=user_config.get("thresholds", {
                "Eff_Min": 0.1, "Voc_Min": 0.1, "Jsc_Min": 0.1,
                "FF_Min": 10.0, "FF_Max": 90.0
            })
        )
