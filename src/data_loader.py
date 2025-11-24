"""
IV Batch Analyzer V5.0 - Professional Edition
Data Loader Module

Handles file scanning, CSV parsing with auto-detection, and initial data cleaning.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

import pandas as pd

from .config import COLUMN_MAPPING, ANALYSIS_PARAMS

logger = logging.getLogger(__name__)


# ================= CUSTOM EXCEPTIONS =================

class CSVParseError(Exception):
    """Raised when CSV file cannot be parsed."""
    pass


# ================= DATA LOADER =================

class IVDataLoader:
    """Handles file scanning, reading with smart CSV parsing, and initial cleaning."""

    def __init__(self, config):
        """
        Initialize IVDataLoader.
        
        Args:
            config: AnalyzerConfig instance
        """
        self.config = config
        self.stop_requested = False

    @staticmethod
    def natural_keys(text: str) -> List[Union[int, str]]:
        """
        Helper for natural sorting (e.g., Batch_2 < Batch_10).
        
        Args:
            text: Text to convert to natural sort key
            
        Returns:
            List of integers and strings for natural sorting
        """
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

    @staticmethod
    def _detect_delimiter(file_path: Path) -> str:
        """
        Auto-detect CSV delimiter using csv.Sniffer.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected delimiter character
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = f.read(4096)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                logger.debug(f"Detected delimiter '{delimiter}' in {file_path.name}")
                return delimiter
        except Exception as e:
            logger.debug(f"Delimiter detection failed: {e}. Using comma.")
            return ','

    @staticmethod
    def _find_header_row(file_path: Path, delimiter: str, encoding: str = 'utf-8') -> int:
        """
        Find the row containing column headers (e.g., "Voltage", "Eff", "Current").
        
        Args:
            file_path: Path to CSV file
            delimiter: CSV delimiter
            encoding: File encoding
            
        Returns:
            Row index (0-based) of header row, or 0 if not found
        """
        header_keywords = ['voltage', 'current', 'v', 'i', 'efficiency', 'eff', 'voc', 'jsc', 'ff']
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for idx, line in enumerate(f):
                    if idx > 50:  # Don't search beyond first 50 rows
                        break
                    
                    line_lower = line.lower()
                    # Check if line contains at least 2 header keywords
                    keyword_count = sum(1 for kw in header_keywords if kw in line_lower)
                    
                    if keyword_count >= 2:
                        logger.debug(f"Found header row at line {idx} in {file_path.name}")
                        return idx
            
            logger.debug(f"No clear header row found in {file_path.name}, using row 0")
            return 0
            
        except Exception as e:
            logger.debug(f"Header detection failed: {e}")
            return 0

    def _read_csv_smart(self, file_path: Path) -> pd.DataFrame:
        """
        Read CSV with smart parsing: auto-detect delimiter, encoding, and header row.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame
            
        Raises:
            CSVParseError: If file cannot be parsed
        """
        # Step 1: Detect delimiter
        delimiter = self._detect_delimiter(file_path)
        
        # Step 2: Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin1']
        df_temp = None
        encoding_used = None
        
        for encoding in encodings:
            try:
                # Detect header row
                header_row = self._find_header_row(file_path, delimiter, encoding)
                
                # Read CSV
                df_temp = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    skiprows=header_row,
                    on_bad_lines='skip'
                )
                
                encoding_used = encoding
                logger.debug(f"Successfully read {file_path.name} with encoding '{encoding}'")
                break
                
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.debug(f"Failed to read with {encoding}: {e}")
                continue
        
        if df_temp is None or df_temp.empty:
            raise CSVParseError(f"Could not parse {file_path.name} with any encoding")
        
        return df_temp

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
        """
        Maps variable CSV headers to standard internal names.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            Dictionary mapping standard names to original column names
        """
        df.columns = [str(c).strip() for c in df.columns]
        lower_cols = {c.lower(): c for c in df.columns}
        found_cols = {}

        for std_name, possible_names in COLUMN_MAPPING.items():
            for name in possible_names:
                if name in lower_cols:
                    found_cols[std_name] = lower_cols[name]
                    break
                # Fallback: partial match
                for col_lower, col_orig in lower_cols.items():
                    if name in col_lower and std_name not in found_cols:
                        found_cols[std_name] = col_orig
                        break
        return found_cols

    def _parse_batch_label(self, folder_name: str) -> str:
        """
        Extracts a readable batch name from folder string.
        
        Args:
            folder_name: Folder name to parse
            
        Returns:
            Batch label string
        """
        match = re.search(r'(?:RUN|R)[-_]?(\d+)', folder_name, re.IGNORECASE)
        if match:
            return f"R{int(match.group(1))}"

        for kw in self.config.control_keywords:
            if kw.lower() in folder_name.lower():
                return "Control"

        digits = re.findall(r'\d+', folder_name)
        if digits:
            return f"Batch_{digits[-1]}"
        return folder_name[:10]

    def _detect_user_initials(self, raw_df: pd.DataFrame) -> str:
        """
        Heuristic to guess user initials from filenames.
        
        Args:
            raw_df: Raw dataframe with SortKey column
            
        Returns:
            User initials string
        """
        candidates = []
        if not raw_df.empty and 'SortKey' in raw_df.columns:
            for fname in raw_df['SortKey'].unique():
                match = re.search(r'-(?!RUN)([A-Z]{2,4})\d+', str(fname))
                if match:
                    candidates.append(match.group(1))

        if candidates:
            return Counter(candidates).most_common(1)[0][0]
        return self.config.default_initials

    def load_data(self, root_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        Scans directory and returns raw data, batch mapping, and user initials.
        
        Args:
            root_dir: Root directory to scan
            
        Returns:
            Tuple of (raw_df, batch_map_df, user_initials)
        """
        root_path = Path(root_dir)
        logger.info(f"Scanning directory: {root_path.resolve()}")

        data_list = []
        batch_map_list = []
        target_files = []

        # 1. Scan files
        for dirpath, _, files in os.walk(root_path):
            if self.stop_requested:
                break

            matched_file = next(
                (f for f in files if any(fnmatch.fnmatch(f, p) for p in self.config.input_patterns)),
                None
            )

            if matched_file:
                target_files.append((Path(dirpath), matched_file))

        # Sort by folder name naturally
        target_files.sort(key=lambda x: self.natural_keys(str(x[0])))
        seen_labels: Dict[str, int] = {}

        # 2. Read files with smart CSV parsing
        for dirpath, filename in target_files:
            if self.stop_requested:
                break

            file_path = dirpath / filename
            try:
                # Use smart CSV reading
                try:
                    df_temp = self._read_csv_smart(file_path)
                except CSVParseError as e:
                    logger.error(f"CSV parse error: {e}")
                    continue
                except PermissionError:
                    logger.error(f"Permission denied (File open?): {file_path}")
                    continue

                cols_map = self._normalize_columns(df_temp)
                if 'Eff' not in cols_map:
                    logger.warning(f"No Eff column found in {file_path.name}, skipping")
                    continue

                # Rename and subset
                rename_dict = {v: k for k, v in cols_map.items()}
                keep_cols = list(cols_map.values())
                if 'CellName' in cols_map:
                    keep_cols.append(cols_map['CellName'])

                df_subset = df_temp[keep_cols].rename(columns=rename_dict)
                df_subset = df_subset.loc[:, ~df_subset.columns.duplicated()]

                # Coerce numeric types
                for col in ANALYSIS_PARAMS:
                    if col in df_subset.columns:
                        df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')

                df_subset.dropna(subset=['Eff'], inplace=True)

                # Generate Batch Label
                short_label = self._parse_batch_label(dirpath.name)
                if short_label in seen_labels:
                    seen_labels[short_label] += 1
                    short_label = f"{short_label}-{seen_labels[short_label]}"
                else:
                    seen_labels[short_label] = 0

                df_subset['Batch'] = short_label
                df_subset['SortKey'] = str(dirpath)

                data_list.append(df_subset)
                batch_map_list.append({'Batch': short_label, 'Folder': dirpath.name})
                logger.info(f"✓ Loaded: {short_label} ({len(df_subset)} cells)")

            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")

        if not data_list:
            return pd.DataFrame(), pd.DataFrame(), self.config.default_initials

        raw_df = pd.concat(data_list, ignore_index=True)
        batch_map_df = pd.DataFrame(batch_map_list)
        user_initials = self._detect_user_initials(raw_df)

        return raw_df, batch_map_df, user_initials
