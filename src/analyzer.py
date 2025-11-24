"""
IV Batch Analyzer V5.0 - Professional Edition
Core Analyzer Module

Main controller that orchestrates all components.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union, Optional

from .config import AnalyzerConfig
from .data_loader import IVDataLoader
from .statistics import IVStatistics
from .visualizer import IVVisualizer
from .reporter import IVReportGenerator

logger = logging.getLogger(__name__)


# ================= MAIN CONTROLLER =================

class IVBatchAnalyzer:
    """Facade class that orchestrates the modular components."""

    def __init__(self, root_dir: Union[str, Path], config: Optional[AnalyzerConfig] = None, theme: str = "dark"):
        """
        Initialize IVBatchAnalyzer.
        
        Args:
            root_dir: Root directory to analyze
            config: AnalyzerConfig instance
            theme: UI theme ("dark" or "light")
        """
        self.root_dir = Path(root_dir)
        self.config = config or AnalyzerConfig()
        self.theme = theme
        self.stop_requested = False
        
        # Progress callback for GUI
        self.progress_callback = None

        self.loader = IVDataLoader(self.config)
        self.stats = IVStatistics(self.config)
        self.reporter = IVReportGenerator(self.config)

    def set_progress_callback(self, callback) -> None:
        """
        Set callback function for progress updates.
        
        Args:
            callback: Function(percent, message) to call for progress updates
        """
        self.progress_callback = callback
    
    def _update_progress(self, percent: float, message: str) -> None:
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def run(self) -> Optional[Path]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Path to output directory, or None if failed/cancelled
        """
        try:
            # 1. Load Data
            self._update_progress(10, "Loading data files...")
            self.loader.stop_requested = self.stop_requested
            raw_df, batch_map_df, user_initials = self.loader.load_data(self.root_dir)

            if self.stop_requested:
                logger.info("Analysis cancelled by user.")
                return None

            if raw_df.empty:
                logger.error("No data found or operation cancelled.")
                return None

            # 2. Create Output Directory
            self._update_progress(20, "Creating output directory...")
            output_dir = self.reporter.create_output_dir(self.root_dir, raw_df, user_initials)

            # 3. Process Statistics
            if self.stop_requested:
                return None
            
            self._update_progress(30, "Cleaning and processing data...")
            clean_df, batch_order = self.stats.clean_data(raw_df)

            if clean_df.empty:
                logger.warning("No data remained after cleaning.")
                return None

            self._update_progress(45, "Computing statistics...")
            group_colors = self.stats.assign_colors(batch_order)
            stats_df, champion_df, top_cells_df, yield_df, comparisons = \
                self.stats.compute_statistics(clean_df, batch_order)

            # 4. Visualize
            if self.stop_requested:
                return None
            
            self._update_progress(60, "Generating plots...")
            visualizer = IVVisualizer(self.config, output_dir, theme=self.theme)
            img_paths = visualizer.visualize(clean_df, stats_df, champion_df, batch_order, group_colors)

            # 5. Generate Reports
            if self.stop_requested:
                return None
            
            self._update_progress(80, "Exporting reports...")
            # Extract data folder name for PPT title
            data_folder_name = self.root_dir.name
            self.reporter.export_reports(
                clean_df, stats_df, champion_df, top_cells_df, yield_df,
                batch_map_df, comparisons, img_paths, user_initials, data_folder_name
            )

            self._update_progress(100, "Analysis complete!")
            logger.info("=== Analysis Complete ===")
            return output_dir
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return None
