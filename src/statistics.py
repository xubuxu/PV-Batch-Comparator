"""
IV Batch Analyzer V5.0 - Professional Edition  
Statistics Module

Handles data cleaning, statistical analysis, yield calculation, and hysteresis analysis.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats

from .config import ANALYSIS_PARAMS, EFF_BINS, EFF_BIN_LABELS
from .data_loader import IVDataLoader

logger = logging.getLogger(__name__)


# ================= STATISTICS ENGINE =================

class IVStatistics:
    """Handles data cleaning, statistical analysis, yield calculation, and hysteresis analysis."""

    def __init__(self, config):
        """
        Initialize IVStatistics.
        
        Args:
            config: AnalyzerConfig instance
        """
        self.config = config

    def clean_data(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Applies filters and outlier removal.
        
        Args:
            raw_df: Raw dataframe from loader
            
        Returns:
            Tuple of (cleaned_df, batch_order)
        """
        if raw_df.empty:
            return pd.DataFrame(), []

        logger.info("Cleaning and processing data...")
        df = raw_df.copy()
        thresh = self.config.thresholds

        # 1. Filter by Scan Direction (Vectorized string op)
        if 'ScanDir' in df.columns:
            pref = self.config.scan_direction
            if pref in ['Reverse', 'Forward']:
                mask = df['ScanDir'].astype(str).str.contains(pref[0], case=False, na=False)
                df = df[mask]

        # 2. Physical Thresholds (Vectorized boolean mask)
        mask_phys = (
                (df['Eff'] > thresh['Eff_Min']) &
                (df['Voc'] > thresh['Voc_Min']) &
                (df['Jsc'] > thresh['Jsc_Min'])
        )
        if 'FF' in df.columns:
            mask_phys &= (df['FF'] > thresh['FF_Min']) & (df['FF'] < thresh['FF_Max'])

        df = df[mask_phys].copy()

        # 3. Remove Duplicates (Keep best Eff per cell)
        if self.config.remove_duplicates and 'CellName' in df.columns:
            df = df.sort_values('Eff', ascending=False).drop_duplicates(subset=['Batch', 'CellName'])

        # 4. Outlier Removal (IQR)
        cleaned_groups = []
        unique_batches = sorted(df['Batch'].unique(), key=IVDataLoader.natural_keys)

        for batch in unique_batches:
            group = df[df['Batch'] == batch]
            if self.config.outlier_removal and len(group) >= 5:
                q1 = group['Eff'].quantile(0.25)
                q3 = group['Eff'].quantile(0.75)
                iqr = q3 - q1
                group = group[(group['Eff'] >= q1 - 1.5 * iqr) & (group['Eff'] <= q3 + 3.0 * iqr)]
            cleaned_groups.append(group)

        if not cleaned_groups:
            logger.warning("All data removed during cleaning.")
            return pd.DataFrame(), []

        clean_df = pd.concat(cleaned_groups, ignore_index=True)
        clean_df['Eff_Bin'] = pd.cut(clean_df['Eff'], bins=EFF_BINS, labels=EFF_BIN_LABELS)

        return clean_df, unique_batches

    def assign_colors(self, batch_order: List[str]) -> Dict[str, str]:
        """
        Assigns consistent colors to batches.
        
        Args:
            batch_order: List of batch names in order
            
        Returns:
            Dictionary mapping batch names to colors
        """
        palette = self.config.plotting.colors
        return {batch: palette[i % len(palette)] for i, batch in enumerate(batch_order)}

    def compute_statistics(self, clean_df: pd.DataFrame, batch_order: List[str]) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Calculates statistics using vectorized Pandas aggregation.
        
        Args:
            clean_df: Cleaned dataframe
            batch_order: List of batch names in order
            
        Returns:
            Tuple of (stats_df, champion_df, top_cells_df, yield_df, comparisons)
        """
        # --- OPTIMIZATION: Vectorized Aggregation ---
        available_params = [p for p in ANALYSIS_PARAMS if p in clean_df.columns]

        # Define aggregation dictionary
        agg_funcs = {param: ['mean', 'median', 'max', 'std'] for param in available_params}
        agg_funcs['Batch'] = ['count']  # Count samples

        # Perform aggregation
        grouped = clean_df.groupby('Batch')
        stats_raw = grouped.agg(agg_funcs)

        # Flatten MultiIndex columns (e.g., ('Eff', 'mean') -> 'Eff_Mean')
        stats_list = []

        # Identify Control Batch
        control_batch = next(
            (b for b in batch_order if any(k.lower() in b.lower() for k in self.config.control_keywords)),
            batch_order[0] if batch_order else None
        )
        comparisons = {'Control': control_batch, 'Results': {}}
        control_data = clean_df[clean_df['Batch'] == control_batch]

        # Reconstruct stats dataframe in correct order and perform T-Tests
        for batch in batch_order:
            if batch not in stats_raw.index:
                continue

            # Extract aggregated stats
            row = {'Batch': batch, 'Count': stats_raw.loc[batch, ('Batch', 'count')]}
            for param in available_params:
                row[f'{param}_Mean'] = stats_raw.loc[batch, (param, 'mean')]
                row[f'{param}_Median'] = stats_raw.loc[batch, (param, 'median')]
                row[f'{param}_Max'] = stats_raw.loc[batch, (param, 'max')]
                row[f'{param}_Std'] = stats_raw.loc[batch, (param, 'std')]

                # T-Test Logic (Cannot be easily vectorized as it compares groups)
                if batch != control_batch and not control_data.empty:
                    df_b = clean_df[clean_df['Batch'] == batch]
                    if len(df_b) > 1 and len(control_data) > 1:
                        try:
                            a = df_b[param].dropna()
                            b = control_data[param].dropna()
                            t, p = stats.ttest_ind(a, b, equal_var=False)
                            diff = row[f'{param}_Mean'] - control_data[param].mean()
                            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

                            if batch not in comparisons['Results']:
                                comparisons['Results'][batch] = {}
                            comparisons['Results'][batch][param] = {
                                'p': p, 'sig': sig, 'diff': diff,
                                'dir': "Increase" if diff > 0 else "Decrease"
                            }
                        except Exception as e:
                            logger.debug(f"T-Test skipped for {batch} vs {control_batch} on {param}: {e}")

            stats_list.append(row)

        stats_df = pd.DataFrame(stats_list)

        # Champion Selection (Vectorized)
        criteria = getattr(self.config, 'champion_criteria', 'Max Eff')
        sort_col = 'FF' if criteria == 'Max FF' else 'Eff'
        
        champion_df = clean_df.sort_values(sort_col, ascending=False).groupby('Batch', as_index=False).first()
        champion_df = champion_df.set_index('Batch').reindex(batch_order).reset_index()

        # Top 10 Cells
        top_cells_df = clean_df.sort_values('Eff', ascending=False).head(10)

        # Yield Calculation
        yield_raw = pd.crosstab(clean_df['Batch'], clean_df['Eff_Bin'], normalize='index') * 100
        yield_df = yield_raw.reindex(batch_order).fillna(0).reset_index()

        return stats_df, champion_df, top_cells_df, yield_df, comparisons

    def calculate_hysteresis_metrics(self, raw_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Calculate hysteresis metrics for cells with both forward and reverse scans.
        
        Important for perovskite solar cells analysis.
        
        Hysteresis Index (HI):
            HI = (PCE_reverse - PCE_forward) / PCE_reverse × 100%
        
        Categories:
            - Negligible: HI < 5%
            - Moderate: 5% <= HI < 15%
            - Significant: HI >= 15%
        
        Args:
            raw_df: Raw dataframe before filtering (must contain ScanDir column)
            
        Returns:
            DataFrame with hysteresis metrics or None if data unavailable
        """
        if 'ScanDir' not in raw_df.columns or 'CellName' not in raw_df.columns or raw_df.empty:
            logger.info("Hysteresis analysis skipped: Missing scan direction or cell name data")
            return None
        
        logger.info("Calculating hysteresis metrics...")
        
        # Separate forward and reverse scans
        df = raw_df.copy()
        df['ScanDir'] = df['ScanDir'].astype(str).str.strip().str.upper()
        
        forward_mask = df['ScanDir'].str.contains('F', case=False, na=False)
        reverse_mask = df['ScanDir'].str.contains('R', case=False, na=False)
        
        df_forward = df[forward_mask].copy()
        df_reverse = df[reverse_mask].copy()
        
        if df_forward.empty or df_reverse.empty:
            logger.warning("Hysteresis analysis skipped: Missing forward or reverse scans")
            return None
        
        # Find cells with both scans
        cells_forward = set(df_forward['CellName'].unique())
        cells_reverse = set(df_reverse['CellName'].unique())
        cells_both = cells_forward & cells_reverse
        
        if not cells_both:
            logger.warning("Hysteresis analysis skipped: No cells with both scan directions")
            return None
        
        logger.info(f"Found {len(cells_both)} cells with both forward and reverse scans")
        
        # Build hysteresis dataframe
        hysteresis_records = []
        
        for cell in cells_both:
            try:
                # Get forward and reverse data (keep best if duplicates)
                fwd = df_forward[df_forward['CellName'] == cell].sort_values('Eff', ascending=False).iloc[0]
                rev = df_reverse[df_reverse['CellName'] == cell].sort_values('Eff', ascending=False).iloc[0]
                
                # Calculate metrics
                params = ['Eff', 'Voc', 'Jsc', 'FF']
                record = {
                    'CellName': cell,
                    'Batch': fwd.get('Batch', 'Unknown')
                }
                
                for param in params:
                    if param in fwd and param in rev:
                        val_fwd = fwd[param]
                        val_rev = rev[param]
                        record[f'{param}_Forward'] = val_fwd
                        record[f'{param}_Reverse'] = val_rev
                        
                        # Hysteresis Index
                        if val_rev != 0:
                            hi = ((val_rev - val_fwd) / val_rev) * 100
                            record[f'HI_{param}'] = hi
                        else:
                            record[f'HI_{param}'] = np.nan
                
                # Overall hysteresis category based on Eff
                hi_eff = record.get('HI_Eff', np.nan)
                if pd.isna(hi_eff):
                    category = 'Unknown'
                elif abs(hi_eff) < 5:
                    category = 'Negligible'
                elif abs(hi_eff) < 15:
                    category = 'Moderate'
                else:
                    category = 'Significant'
                
                record['Category'] = category
                record['Eff_Average'] = (record['Eff_Forward'] + record['Eff_Reverse']) / 2
                
                hysteresis_records.append(record)
                
            except Exception as e:
                logger.debug(f"Hysteresis calculation failed for cell {cell}: {e}")
                continue
        
        if not hysteresis_records:
            logger.warning("No hysteresis metrics could be calculated")
            return None
        
        hysteresis_df = pd.DataFrame(hysteresis_records)
        logger.info(f"✓ Hysteresis analysis complete: {len(hysteresis_df)} cells analyzed")
        
        return hysteresis_df
