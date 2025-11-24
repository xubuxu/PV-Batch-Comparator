"""
IV Batch Analyzer V5.0 - Professional Edition
Package Initialization
"""

from .config import (
    AnalyzerConfig,
    PlottingConfig,
    ConfigManager,
    UNIT_MAP,
    COLUMN_MAPPING,
    ANALYSIS_PARAMS
)

from .data_loader import IVDataLoader, CSVParseError
from .statistics import IVStatistics

__version__ = "5.0.0"
__all__ = [
    'AnalyzerConfig',
    'PlottingConfig',
    'ConfigManager',
    'IVDataLoader',
    'IVStatistics',
    'CSVParseError',
    'UNIT_MAP',
    'COLUMN_MAPPING',
    'ANALYSIS_PARAMS',
]
