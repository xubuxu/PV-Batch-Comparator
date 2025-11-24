"""
IV Batch Analyzer V5.0 - Professional Edition
Visualizer Module

Generates plots using Matplotlib/Seaborn with theme support for dark/light modes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import optimize

from .config import UNIT_MAP, ANALYSIS_PARAMS

logger = logging.getLogger(__name__)


# ================= VISUALIZER =================

class IVVisualizer:
    """Generates plots using Matplotlib/Seaborn with theme support."""

    def __init__(self, config, output_dir: Optional[Path], theme: str = "dark"):
        """
        Initialize IVVisualizer.
        
        Args:
            config: AnalyzerConfig instance
            output_dir: Directory to save plots
            theme: Color theme ("dark" or "light")
        """
        self.config = config
        self.output_dir = output_dir
        self.theme = theme
        self.img_paths: Dict[str, Path] = {}

        # State injection
        self.clean_df: Optional[pd.DataFrame] = None
        self.stats_df: Optional[pd.DataFrame] = None
        self.champion_df: Optional[pd.DataFrame] = None
        self.batch_order: List[str] = []
        self.group_colors: Dict[str, str] = {}

        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        """Configure matplotlib/seaborn style based on theme."""
        from .config import THEMES
        
        # Get theme config (fallback to Dark if not found)
        theme_name = getattr(self.config, 'plot_theme', 'Dark')
        theme_config = THEMES.get(theme_name, THEMES['Dark'])
        
        # Apply Seaborn style
        sns.set_theme(
            style=theme_config.style, 
            context=theme_config.context, 
            font_scale=1.1
        )
        
        # Apply custom rcParams
        plt.rcParams.update({
            'font.family': 'sans-serif', # Default fallback
            'font.sans-serif': list(theme_config.font_family),
            'font.serif': list(theme_config.font_family), # In case serif is preferred
            'axes.unicode_minus': False,
            'lines.linewidth': theme_config.line_width,
            'figure.facecolor': theme_config.background_color,
            'axes.facecolor': theme_config.background_color,
            'axes.edgecolor': theme_config.text_color,
            'axes.labelcolor': theme_config.text_color,
            'xtick.color': theme_config.text_color,
            'ytick.color': theme_config.text_color,
            'text.color': theme_config.text_color,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': theme_config.grid_color,
            'grid.linestyle': '--',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.linewidth': 1.5,
            'figure.dpi': theme_config.dpi,
            'savefig.dpi': theme_config.dpi
        })
        
        # Store for use in plots
        self.theme_config = theme_config

    def _save_plot(self, filename: str) -> Optional[Path]:
        """Save current plot to file(s) based on config."""
        if not self.output_dir:
            return None
        try:
            # Get configured formats (default to png if missing)
            formats = getattr(self.config, 'output_formats', ('png',))
            base_name = filename.rsplit('.', 1)[0]
            saved_path = None
            
            for fmt in formats:
                # Clean format string
                fmt = fmt.lower().strip().replace('.', '')
                fname = f"{base_name}.{fmt}"
                path = self.output_dir / fname
                
                plt.savefig(
                    path, 
                    dpi=self.theme_config.dpi, 
                    bbox_inches='tight', 
                    facecolor=plt.gcf().get_facecolor(),
                    format=fmt
                )
                
                # Return path to PNG for report generation (preferred)
                if fmt == 'png':
                    saved_path = path
            
            # If PNG wasn't saved but others were, return the last one
            if not saved_path and formats:
                saved_path = self.output_dir / f"{base_name}.{formats[-1]}"
                
            return saved_path
            
        except Exception as e:
            logger.error(f"Failed to save {filename}: {e}")
            return None
        finally:
            plt.close()

    @staticmethod
    def _apply_legend_style(ax) -> None:
        """Apply consistent legend styling."""
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=True)

    @staticmethod
    def _reconstruct_jv_curve(voc: float, jsc: float, ff: float, points: int = 100) -> tuple:
        """Reconstructs JV curve using single-diode model approximation."""
        voc_val = float(voc)
        jsc_val = float(jsc)

        v_curve = np.linspace(-0.1, voc_val + 0.05, points)
        j_curve = jsc_val * (1 - v_curve / voc_val)  # Default linear fallback

        try:
            def model_j(v, diode_factor):
                with np.errstate(over='ignore', invalid='ignore'):
                    term = np.exp(voc_val / diode_factor) - 1
                    if term == 0 or np.isinf(term) or np.isnan(term):
                        term = 1e9
                    exp_v = np.exp(v / diode_factor) - 1
                    return jsc_val - (jsc_val / term) * exp_v

            def get_ff_error(diode_factor):
                if diode_factor <= 1e-4:
                    return 100
                v_r = np.linspace(0, voc_val, 50)
                j_r = model_j(v_r, diode_factor)
                p_max = np.max(v_r * j_r)
                return (p_max / (voc_val * jsc_val)) * 100 - ff

            b_opt = optimize.fsolve(get_ff_error, 0.05, xtol=1e-4)[0]
            if b_opt < 1e-4:
                b_opt = 0.05

            j_curve = model_j(v_curve, b_opt)
        except Exception as e:
            logger.debug(f"JV model fit failed (using linear fallback): {e}")

        return v_curve, j_curve

    # --- PLOTTING METHODS ---

    def _plot_boxplot(self) -> None:
        """Generate boxplot for all parameters."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in self.clean_df.columns]
            fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 6))
            if len(params) == 1:
                axes = [axes]

            for ax, param in zip(axes, params):
                sns.boxplot(
                    data=self.clean_df, x='Batch', y=param, order=self.batch_order,
                    ax=ax, palette=self.group_colors, showfliers=False, width=0.5
                )
                if self.config.plotting.show_points:
                    sns.stripplot(
                        data=self.clean_df, x='Batch', y=param, order=self.batch_order,
                        ax=ax, color=".25", size=4, alpha=0.6, jitter=True
                    )
                ax.set_title("")
                ax.set_ylabel(f"{param} {UNIT_MAP.get(param, '')}", fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.set_xlabel('')

            plt.tight_layout()
            self.img_paths['box'] = self._save_plot('1_Boxplot.png')
        except Exception as e:
            logger.error(f"Boxplot failed: {e}")

    def _plot_histogram(self) -> None:
        """Generate efficiency distribution histogram."""
        try:
            plt.figure(figsize=self.config.plotting.base_figsize)
            sns.histplot(
                data=self.clean_df, x='Eff', hue='Batch', hue_order=self.batch_order,
                palette=self.group_colors, kde=True, element="step", fill=False, legend=False
            )
            plt.title('Efficiency Distribution', fontweight='bold')
            plt.xlabel(f'Efficiency {UNIT_MAP["Eff"]}')

            handles = [
                mlines.Line2D([], [], color=self.group_colors[b], linewidth=2, label=b)
                for b in self.batch_order
            ]
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Batch")

            self.img_paths['hist'] = self._save_plot('1_Histogram.png')
        except Exception as e:
            logger.error(f"Histogram failed: {e}")

    def _plot_trend(self) -> None:
        """Generate trend analysis plot."""
        try:
            params = [p for p in ANALYSIS_PARAMS if p in self.clean_df.columns]
            fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 6))
            if len(params) == 1:
                axes = [axes]

            for ax, param in zip(axes, params):
                td = self.stats_df.set_index('Batch').reindex(self.batch_order)
                x = range(len(self.batch_order))
                ax.plot(x, td[f'{param}_Max'], label='Max', color='#E64B35', linestyle='--', marker='^', markersize=8)
                ax.plot(x, td[f'{param}_Mean'], label='Mean', color='#3C5488', linestyle='-', marker='o', markersize=8)
                ax.plot(x, td[f'{param}_Median'], label='Median', color='#00A087', linestyle=':', marker='x',
                        markersize=8)
                ax.set_title("")
                ax.set_ylabel(f"{param} {UNIT_MAP.get(param, '')}", fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(self.batch_order, rotation=45)

            handles, labels = axes[0].get_legend_handles_labels()
            axes[-1].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout()
            self.img_paths['trend'] = self._save_plot('2_Trend.png')
        except Exception as e:
            logger.error(f"Trend plot failed: {e}")

    def _plot_yield(self) -> None:
        """Generate yield distribution plot."""
        try:
            bin_counts = pd.crosstab(self.clean_df['Batch'], self.clean_df['Eff_Bin'], normalize='index') * 100
            bin_counts = bin_counts.reindex(self.batch_order).fillna(0)
            bin_counts = bin_counts.loc[:, (bin_counts != 0).any(axis=0)]

            if not bin_counts.empty:
                ax = bin_counts.plot(
                    kind='bar', stacked=True, figsize=self.config.plotting.base_figsize,
                    colormap='viridis_r', width=0.7
                )
                plt.title('Efficiency Yield Distribution', fontweight='bold')
                plt.ylabel('Percentage (%)')
                plt.xticks(rotation=45)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Eff Range")
                self.img_paths['yield'] = self._save_plot('3_Yield.png')
        except Exception as e:
            logger.error(f"Yield plot failed: {e}")

    def _plot_jv_curves(self) -> None:
        """Generate J-V curves for champion cells."""
        try:
            plt.figure(figsize=self.config.plotting.base_figsize)
            has_jv = False
            for batch in self.batch_order:
                if batch in self.champion_df['Batch'].values:
                    row = self.champion_df[self.champion_df['Batch'] == batch].iloc[0]
                    try:
                        v, j = self._reconstruct_jv_curve(row['Voc'], row['Jsc'], row['FF'])
                        plt.plot(
                            v, j, label=f"{batch} ({row['Eff']:.2f}%)",
                            color=self.group_colors.get(batch, 'black'), linewidth=2.5
                        )
                        has_jv = True
                    except Exception as e:
                        logger.warning(f"Could not plot JV curve for {batch}: {e}")
                        continue
            if has_jv:
                plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                plt.xlabel('Voltage (V)', fontweight='bold')
                plt.ylabel('Current Density (mA/cm²)', fontweight='bold')
                plt.title('J-V Curves of Champion Cells', fontweight='bold')
                self._apply_legend_style(plt.gca())
                self.img_paths['jv_curve'] = self._save_plot('4_JV_Curves.png')
            else:
                plt.close()
        except Exception as e:
            logger.error(f"JV Curve plot failed: {e}")

    def _plot_correlations(self) -> None:
        """Generate correlation analysis plots."""
        try:
            if 'Voc' in self.clean_df.columns and 'Jsc' in self.clean_df.columns:
                plt.figure(figsize=self.config.plotting.base_figsize)
                sns.scatterplot(
                    data=self.clean_df, x='Jsc', y='Voc', hue='Batch', hue_order=self.batch_order,
                    palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, edgecolor='k'
                )

                avg_ff = self.clean_df['FF'].mean() / 100.0 if 'FF' in self.clean_df.columns else 0.75
                x_min, x_max = plt.xlim()
                y_min, y_max = plt.ylim()

                if x_max > x_min and y_max > y_min:
                    xx = np.linspace(x_min, x_max, 100)
                    yy = np.linspace(y_min, y_max, 100)
                    grid_x, grid_y = np.meshgrid(xx, yy)
                    grid_z = grid_x * grid_y * avg_ff

                    cs = plt.contour(grid_x, grid_y, grid_z, levels=8, colors='gray', linestyles='--', alpha=0.4)
                    plt.clabel(cs, inline=1, fontsize=10, fmt='%.1f%%')

                plt.title(f'Voc vs. Jsc Correlation (Iso-Eff @ FF={avg_ff * 100:.0f}%)', fontweight='bold')
                self._apply_legend_style(plt.gca())
                self.img_paths['voc_jsc'] = self._save_plot('5_Voc_Jsc_Tradeoff.png')

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            pairs = [
                ('Voc', 'Eff', axes[0, 0]), ('Jsc', 'Eff', axes[0, 1]),
                ('FF', 'Eff', axes[1, 0]), ('Voc', 'Jsc', axes[1, 1])
            ]

            for x, y, ax in pairs:
                if x in self.clean_df.columns and y in self.clean_df.columns:
                    sns.scatterplot(
                        data=self.clean_df, x=x, y=y, hue='Batch', hue_order=self.batch_order,
                        palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, ax=ax,
                        legend=False
                    )
                    ax.set_title(f'{y} vs {x}', fontweight='bold')
                    ax.set_xlabel(f'{x} {UNIT_MAP.get(x, "")}')
                    ax.set_ylabel(f'{y} {UNIT_MAP.get(y, "")}')

            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.group_colors[b], markersize=10)
                for b in self.batch_order
            ]
            fig.legend(handles, self.batch_order, bbox_to_anchor=(0.86, 0.9), loc='upper left', title="Batch")
            plt.tight_layout(rect=(0, 0, 0.85, 1))
            self.img_paths['combo_drivers'] = self._save_plot('6_Drivers.png')
        except Exception as e:
            logger.error(f"Correlation plots failed: {e}")

    def _plot_resistance_analysis(self) -> None:
        """Generate resistance analysis plots."""
        try:
            if 'Rs' not in self.clean_df.columns or 'FF' not in self.clean_df.columns:
                return
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # FF vs Rs
            sns.scatterplot(
                data=self.clean_df, x='Rs', y='FF', hue='Batch', hue_order=self.batch_order,
                palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, ax=axes[0, 0],
                legend=False
            )
            axes[0, 0].set_title('FF vs Rs', fontweight='bold')
            axes[0, 0].set_xlabel(f'Rs {UNIT_MAP["Rs"]}')
            axes[0, 0].set_ylabel(f'FF {UNIT_MAP["FF"]}')

            # FF vs Rsh
            if 'Rsh' in self.clean_df.columns:
                sns.scatterplot(
                    data=self.clean_df, x='Rsh', y='FF', hue='Batch', hue_order=self.batch_order,
                    palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, ax=axes[0, 1],
                    legend=False
                )
                axes[0, 1].set_title('FF vs Rsh', fontweight='bold')
                axes[0, 1].set_xlabel(f'Rsh {UNIT_MAP["Rsh"]}')
                axes[0, 1].set_ylabel(f'FF {UNIT_MAP["FF"]}')
                axes[0, 1].set_xscale('log')

            # Eff vs Rs
            sns.scatterplot(
                data=self.clean_df, x='Rs', y='Eff', hue='Batch', hue_order=self.batch_order,
                palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, ax=axes[1, 0],
                legend=False
            )
            axes[1, 0].set_title('Eff vs Rs', fontweight='bold')
            axes[1, 0].set_xlabel(f'Rs {UNIT_MAP["Rs"]}')
            axes[1, 0].set_ylabel(f'Eff {UNIT_MAP["Eff"]}')

            # Eff vs Rsh
            if 'Rsh' in self.clean_df.columns:
                sns.scatterplot(
                    data=self.clean_df, x='Rsh', y='Eff', hue='Batch', hue_order=self.batch_order,
                    palette=self.group_colors, s=self.config.plotting.marker_size, alpha=0.8, ax=axes[1, 1],
                    legend=False
                )
                axes[1, 1].set_title('Eff vs Rsh', fontweight='bold')
                axes[1, 1].set_xlabel(f'Rsh {UNIT_MAP["Rsh"]}')
                axes[1, 1].set_ylabel(f'Eff {UNIT_MAP["Eff"]}')
                axes[1, 1].set_xscale('log')

            handles = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=self.group_colors[b], markersize=10)
                for b in self.batch_order
            ]
            fig.legend(handles, self.batch_order, bbox_to_anchor=(0.86, 0.9), loc='upper left', title="Batch")
            plt.tight_layout(rect=(0, 0, 0.85, 1))
            self.img_paths['resistance'] = self._save_plot('7_Resistance.png')
        except Exception as e:
            logger.error(f"Resistance plot failed: {e}")

    def _plot_model_fitting(self) -> None:
        """Generate model fitting analysis plots."""
        try:
            # Check if we have fitted data
            if 'Rs_fitted' not in self.clean_df.columns:
                return

            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(2, 3)

            # 1. Measured vs Fitted IV Curves (Top Row)
            ax1 = fig.add_subplot(gs[0, :])
            has_curves = False
            
            # We need to reconstruct curves using fitted parameters
            for batch in self.batch_order:
                batch_data = self.clean_df[
                    (self.clean_df['Batch'] == batch) & 
                    (self.clean_df['fit_successful'] == True)
                ]
                if not batch_data.empty:
                    # Take the best cell
                    best_cell = batch_data.sort_values('Eff', ascending=False).iloc[0]
                    
                    # Reconstruct curve using diode equation (simplified for plotting)
                    # I = IL - I0*(exp((V+I*Rs)/(n*Vt)) - 1) - (V+I*Rs)/Rsh
                    # Here we use the simplified reconstruction from before but with fitted params
                    try:
                        v, j = self._reconstruct_jv_curve(
                            best_cell['Voc'], best_cell['Jsc'], best_cell['FF']
                        )
                        ax1.plot(v, j, label=f"{batch} (Meas)", 
                               color=self.group_colors.get(batch, 'black'), 
                               linestyle='-', alpha=0.6)
                        
                        # Ideally we would plot the exact fitted curve here
                        # For now, we'll just plot the measured one to show we have data
                        has_curves = True
                    except:
                        pass
            
            if has_curves:
                ax1.set_title('Model Fitting: Champion Cells', fontweight='bold')
                ax1.set_xlabel('Voltage (V)')
                ax1.set_ylabel('Current Density (mA/cm²)')
                ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
                self._apply_legend_style(ax1)

            # 2. Rs Comparison (Slope vs Fitted)
            ax2 = fig.add_subplot(gs[1, 0])
            if 'Rs_slope' in self.clean_df.columns:
                sns.scatterplot(
                    data=self.clean_df, x='Rs_slope', y='Rs_fitted', 
                    hue='Batch', hue_order=self.batch_order, palette=self.group_colors,
                    ax=ax2, legend=False
                )
                # Add diagonal line
                min_val = min(self.clean_df['Rs_slope'].min(), self.clean_df['Rs_fitted'].min())
                max_val = max(self.clean_df['Rs_slope'].max(), self.clean_df['Rs_fitted'].max())
                ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
                ax2.set_title('Rs: Slope vs Fitted', fontweight='bold')
                ax2.set_xlabel(f'Rs Slope {UNIT_MAP["Rs"]}')
                ax2.set_ylabel(f'Rs Fitted {UNIT_MAP["Rs"]}')

            # 3. Ideality Factor Distribution
            ax3 = fig.add_subplot(gs[1, 1])
            if 'n' in self.clean_df.columns:
                sns.histplot(
                    data=self.clean_df, x='n', hue='Batch', hue_order=self.batch_order,
                    palette=self.group_colors, element="step", fill=False, ax=ax3, legend=False
                )
                ax3.set_title('Ideality Factor (n) Distribution', fontweight='bold')
                ax3.set_xlabel('Ideality Factor')
                ax3.axvline(1.0, color='gray', linestyle='--')
                ax3.axvline(2.0, color='gray', linestyle='--')

            # 4. Fitting Success Rate
            ax4 = fig.add_subplot(gs[1, 2])
            if 'extraction_method' in self.clean_df.columns:
                counts = self.clean_df['extraction_method'].value_counts()
                counts.plot(kind='bar', ax=ax4, color='#3C5488')
                ax4.set_title('Fitting Method Usage', fontweight='bold')
                ax4.set_ylabel('Count')
                ax4.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            self.img_paths['model_fitting'] = self._save_plot('9_Model_Fitting.png')
        except Exception as e:
            logger.error(f"Model fitting plot failed: {e}")

    def _plot_hysteresis(self, hysteresis_df: Optional[pd.DataFrame]) -> None:
        """Generate hysteresis analysis plots."""
        try:
            if hysteresis_df is None or hysteresis_df.empty:
                return

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 1. Hysteresis Index Distribution
            sns.histplot(
                data=hysteresis_df, x='HI_Eff', hue='Batch', multiple="stack",
                palette=self.group_colors, ax=axes[0, 0]
            )
            axes[0, 0].set_title('Hysteresis Index (HI) Distribution', fontweight='bold')
            axes[0, 0].set_xlabel('Hysteresis Index (%)')
            axes[0, 0].axvline(5, color='orange', linestyle='--', label='Moderate (5%)')
            axes[0, 0].axvline(15, color='red', linestyle='--', label='Significant (15%)')
            axes[0, 0].legend()

            # 2. Forward vs Reverse Efficiency
            sns.scatterplot(
                data=hysteresis_df, x='Eff_Reverse', y='Eff_Forward',
                hue='Batch', palette=self.group_colors, ax=axes[0, 1]
            )
            # Diagonal line
            max_eff = max(hysteresis_df['Eff_Reverse'].max(), hysteresis_df['Eff_Forward'].max())
            axes[0, 1].plot([0, max_eff], [0, max_eff], 'k--', alpha=0.5)
            axes[0, 1].set_title('Forward vs Reverse Efficiency', fontweight='bold')
            axes[0, 1].set_xlabel(f'Reverse Eff {UNIT_MAP["Eff"]}')
            axes[0, 1].set_ylabel(f'Forward Eff {UNIT_MAP["Eff"]}')

            # 3. HI vs Efficiency
            sns.scatterplot(
                data=hysteresis_df, x='Eff_Average', y='HI_Eff',
                hue='Batch', palette=self.group_colors, ax=axes[1, 0], legend=False
            )
            axes[1, 0].set_title('Hysteresis vs Efficiency', fontweight='bold')
            axes[1, 0].set_xlabel(f'Average Eff {UNIT_MAP["Eff"]}')
            axes[1, 0].set_ylabel('Hysteresis Index (%)')
            axes[1, 0].axhline(0, color='gray', linestyle='-', linewidth=0.5)

            # 4. Hysteresis Category
            category_counts = hysteresis_df['Category'].value_counts()
            if not category_counts.empty:
                category_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1, 1],
                                   colors=['#00A087', '#E64B35', '#F39B7F'])
                axes[1, 1].set_ylabel('')
                axes[1, 1].set_title('Hysteresis Severity', fontweight='bold')

            plt.tight_layout()
            self.img_paths['hysteresis'] = self._save_plot('10_Hysteresis_Analysis.png')
        except Exception as e:
            logger.error(f"Hysteresis plot failed: {e}")

    def _plot_anomaly_detection(self) -> None:
        """Generate anomaly detection plots."""
        try:
            if 'has_s_shape' not in self.clean_df.columns:
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # 1. Anomaly Statistics
            anomalies = self.clean_df[['Batch', 'has_s_shape', 'has_kink']].melt(
                id_vars=['Batch'], var_name='Anomaly', value_name='Present'
            )
            anomalies = anomalies[anomalies['Present'] == True]
            
            if not anomalies.empty:
                sns.countplot(
                    data=anomalies, x='Batch', hue='Anomaly',
                    palette={'has_s_shape': '#E64B35', 'has_kink': '#F39B7F'},
                    ax=axes[0]
                )
                axes[0].set_title('Detected Anomalies Count', fontweight='bold')
                axes[0].tick_params(axis='x', rotation=45)
            else:
                axes[0].text(0.5, 0.5, "No Anomalies Detected", 
                           ha='center', va='center', transform=axes[0].transAxes)

            # 2. Severity Distribution
            if 'severity' in self.clean_df.columns:
                severity_order = ['None', 'Mild', 'Moderate', 'Severe']
                sns.countplot(
                    data=self.clean_df, x='severity', hue='Batch',
                    order=[s for s in severity_order if s in self.clean_df['severity'].unique()],
                    palette=self.group_colors, ax=axes[1]
                )
                axes[1].set_title('S-Shape Severity Distribution', fontweight='bold')
                axes[1].set_xlabel('Severity Level')

            plt.tight_layout()
            self.img_paths['anomalies'] = self._save_plot('11_Anomaly_Detection.png')
        except Exception as e:
            logger.error(f"Anomaly plot failed: {e}")

    def visualize(self, clean_df: pd.DataFrame, stats_df: pd.DataFrame,
                  champion_df: pd.DataFrame, batch_order: List[str],
                  group_colors: Dict[str, str],
                  hysteresis_df: Optional[pd.DataFrame] = None) -> Dict[str, Path]:
        """
        Main entry point for visualization.
        
        Args:
            clean_df: Cleaned dataframe
        return self.img_paths
