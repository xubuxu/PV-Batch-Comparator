"""
IV Batch Analyzer V5.0 - Professional Edition
Reporter Module

Generates Excel, Word, and PowerPoint reports.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL

try:
    from pptx import Presentation
    from pptx.util import Inches as PptInches, Pt as PptPt
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    HAS_PPTX = True
except ImportError:
    HAS_PPTX = False
    Presentation = None
    PptInches = None
    PptPt = None
    PP_ALIGN = None
    MSO_ANCHOR = None

logger = logging.getLogger(__name__)


# ================= CUSTOM EXCEPTIONS =================

class ReportGenerationError(Exception):
    """Raised when report generation fails."""
    pass


# ================= REPORT GENERATOR =================

class IVReportGenerator:
    """Generates Excel, Word, and PowerPoint reports."""

    def __init__(self, config):
        """
        Initialize IVReportGenerator.
        
        Args:
            config: AnalyzerConfig instance
        """
        self.config = config
        self.output_dir: Optional[Path] = None

    def create_output_dir(self, root_dir: Path, raw_df: pd.DataFrame, user_initials: str) -> Path:
        """
        Creates output directory with timestamp and run info.
        
        Args:
            root_dir: Root directory for analysis
            raw_df: Raw dataframe
            user_initials: User initials
            
        Returns:
            Path to created output directory
        """
        run_nums = []
        if not raw_df.empty and 'Batch' in raw_df.columns:
            for batch in raw_df['Batch'].unique():
                match = re.search(r'\d+', str(batch))
                if match:
                    run_nums.append(int(match.group(0)))

        suffix = datetime.now().strftime("%Y%m%d_%H%M")
        if run_nums:
            min_r, max_r = min(run_nums), max(run_nums)
            suffix = f"RUN{min_r}" if min_r == max_r else f"RUN{min_r}-{max_r}"

        folder_name = f"IV_Report_{user_initials}_{suffix}"
        self.output_dir = root_dir / folder_name
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Output directory created: {self.output_dir.name}")
        return self.output_dir

    def export_reports(self, clean_df: pd.DataFrame, stats_df: pd.DataFrame,
                       champion_df: pd.DataFrame, top_cells_df: pd.DataFrame,
                       yield_df: pd.DataFrame, batch_map_df: pd.DataFrame,
                       comparisons: Dict[str, Any], img_paths: Dict[str, Path],
                       user_initials: str, data_folder_name: str = "") -> None:
        """
        Generate all reports (Excel, Word, PowerPoint).
        
        Args:
            clean_df: Cleaned data
            stats_df: Statistics
            champion_df: Champion cells
            top_cells_df: Top 10 cells
            yield_df: Yield distribution
            batch_map_df: Batch mapping
            comparisons: Statistical comparisons
            img_paths: Dictionary of plot paths
            user_initials: User initials
            data_folder_name: Name of the data folder for PPT title
        """
        if not self.output_dir:
            raise ReportGenerationError("Output directory not created")
            
        logger.info("Exporting reports...")

        try:
            # --- 1. Excel Export ---
            self._export_excel(clean_df, stats_df, champion_df, top_cells_df, yield_df)
            
            # --- 2. Word Export ---
            self._export_word(stats_df, champion_df, top_cells_df, yield_df, 
                            batch_map_df, comparisons, img_paths, user_initials)
            
            # --- 3. PPT Export ---
            if HAS_PPTX:
                self._export_powerpoint(stats_df, champion_df, top_cells_df, yield_df,
                                      batch_map_df, comparisons, img_paths, user_initials, data_folder_name)
            
            logger.info(f"✓ Reports saved to: {self.output_dir}")
            
        except Exception as e:
            raise ReportGenerationError(f"Report generation failed: {e}") from e

    def _export_excel(self, clean_df: pd.DataFrame, stats_df: pd.DataFrame,
                     champion_df: pd.DataFrame, top_cells_df: pd.DataFrame,
                     yield_df: pd.DataFrame) -> None:
        """Export data to Excel file."""
        path = self.output_dir / self.config.excel_data_name
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            clean_df.to_excel(writer, sheet_name='Cleaned_Data', index=False)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            if not champion_df.empty:
                champion_df.to_excel(writer, sheet_name='Champions', index=False)
            top_cells_df.to_excel(writer, sheet_name='Top_10_Cells', index=False)
            yield_df.to_excel(writer, sheet_name='Yield_Table', index=False)

            for sheet in writer.sheets.values():
                for col in sheet.columns:
                    sheet.column_dimensions[col[0].column_letter].width = 15

    def _export_word(self, stats_df: pd.DataFrame, champion_df: pd.DataFrame,
                    top_cells_df: pd.DataFrame, yield_df: pd.DataFrame,
                    batch_map_df: pd.DataFrame, comparisons: Dict[str, Any],
                    img_paths: Dict[str, Path], user_initials: str) -> None:
        """Export report to Word document."""
        doc = Document()
        doc.add_heading(f'IV Analysis Report ({user_initials})', 0)
        doc.add_heading('1. Executive Summary', level=1)

        if comparisons.get('Results'):
            ctrl = comparisons['Control']
            for batch, res in comparisons['Results'].items():
                p = doc.add_paragraph()
                run = p.add_run(f"▶ {batch} vs {ctrl}: ")
                run.bold = True
                run.font.color.rgb = RGBColor(0, 51, 102)
                if 'Eff' in res:
                    e = res['Eff']
                    p.add_run(f"Eff {e['dir']} by {abs(e['diff']):.2f}% (p={e['p']:.3f}, {e['sig']}).")
        else:
            doc.add_paragraph("Single batch analysis.")

        doc.add_heading('2. Data Tables', level=1)
        self._add_word_table(doc, batch_map_df, ['Batch', 'Folder'], "2.1 Batch & Folder Reference")

        cols = ['Batch', 'Count', 'Eff_Mean', 'Eff_Max', 'Voc_Mean', 'FF_Mean']
        cols = [c for c in cols if c in stats_df.columns]
        self._add_word_table(doc, stats_df, cols, "2.2 Statistical Summary of PV Parameters")

        champ_cols = [c for c in ['Batch', 'CellName', 'Eff', 'Voc', 'Jsc', 'FF', 'Rs'] if c in champion_df.columns]
        self._add_word_table(doc, champion_df, champ_cols, "2.3 Champion Cell Parameters")
        self._add_word_table(doc, top_cells_df, champ_cols, "2.4 Top 10 Highest Efficiency Cells")

        yield_cols = [c for c in yield_df.columns if c != 'Batch']
        valid_yield_cols = ['Batch'] + [c for c in yield_cols if yield_df[c].sum() > 0]
        self._add_word_table(doc, yield_df, valid_yield_cols, "2.5 Efficiency Yield Distribution (%)")

        doc.add_heading('3. Visual Analysis', level=1)
        plot_order = [
            ('jv_curve', 'J-V Curves of Champion Cells'),
            ('voc_jsc', 'Voc vs. Jsc Correlation Analysis'),
            ('box', 'Distribution of Electrical Parameters'),
            ('hist', 'Efficiency Histogram'),
            ('trend', 'Batch Trend Analysis (Max/Mean/Median)'),
            ('yield', 'Efficiency Yield Stack'),
            ('combo_drivers', 'Key Efficiency Drivers (Correlations)'),
            ('resistance', 'Parasitic Resistance Analysis (Rs & Rsh)')
        ]
        for key, title in plot_order:
            if key in img_paths and img_paths[key]:
                doc.add_heading(title, level=2)
                doc.add_picture(str(img_paths[key]), width=Inches(6.5))
        
        doc.save(self.output_dir / self.config.report_docx_name)

    def _export_powerpoint(self, stats_df: pd.DataFrame, champion_df: pd.DataFrame,
                          top_cells_df: pd.DataFrame, yield_df: pd.DataFrame,
                          batch_map_df: pd.DataFrame, comparisons: Dict[str, Any],
                          img_paths: Dict[str, Path], user_initials: str, data_folder_name: str = "") -> None:
        """Export report to PowerPoint presentation."""
        prs = Presentation()
        prs.slide_width = PptInches(13.33)
        prs.slide_height = PptInches(7.5)

        # Title Slide
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        # Use data folder name if provided, otherwise fallback to generic title
        if data_folder_name:
            slide.shapes.title.text = f"IV Analysis Report - {data_folder_name}"
        else:
            slide.shapes.title.text = "IV Measurement Analysis"
        slide.placeholders[1].text = f"Operator: {user_initials}\nDate: {datetime.now().strftime('%Y-%m-%d')}"

        # Summary Slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = "Executive Summary"
        tf = slide.shapes.placeholders[1].text_frame
        slide.shapes.title.text_frame.paragraphs[0].font.size = PptPt(24)

        if comparisons.get('Results'):
            for batch, res in comparisons['Results'].items():
                p = tf.add_paragraph()
                p.text = f"{batch} vs {comparisons['Control']}:"
                p.font.bold = True
                p.font.size = PptPt(16)

                if 'Eff' in res:
                    p2 = tf.add_paragraph()
                    p2.level = 1
                    p2.text = f"Efficiency {res['Eff']['dir']} {abs(res['Eff']['diff']):.2f}% ({res['Eff']['sig']})"
                    p2.font.size = PptPt(16)

        cols = ['Batch', 'Count', 'Eff_Mean', 'Eff_Max', 'Voc_Mean', 'FF_Mean']
        cols = [c for c in cols if c in stats_df.columns]
        champ_cols = [c for c in ['Batch', 'CellName', 'Eff', 'Voc', 'Jsc', 'FF', 'Rs'] if c in champion_df.columns]
        yield_cols = [c for c in yield_df.columns if c != 'Batch']
        valid_yield_cols = ['Batch'] + [c for c in yield_cols if yield_df[c].sum() > 0]

        self._add_pptx_table(prs, batch_map_df, ['Batch', 'Folder'], "Batch & Folder Reference")
        self._add_pptx_table(prs, stats_df, cols, "Statistical Summary of PV Parameters")
        self._add_pptx_table(prs, champion_df, champ_cols, "Champion Cell Parameters")
        self._add_pptx_table(prs, top_cells_df, champ_cols, "Top 10 Highest Efficiency Cells")
        self._add_pptx_table(prs, yield_df, valid_yield_cols, "Efficiency Yield Distribution (%)")

        plot_order = [
            ('jv_curve', 'J-V Curves of Champion Cells'),
            ('voc_jsc', 'Voc vs. Jsc Correlation Analysis'),
            ('box', 'Distribution of Electrical Parameters'),
            ('hist', 'Efficiency Histogram'),
            ('trend', 'Batch Trend Analysis (Max/Mean/Median)'),
            ('yield', 'Efficiency Yield Stack'),
            ('combo_drivers', 'Key Efficiency Drivers (Correlations)'),
            ('resistance', 'Parasitic Resistance Analysis (Rs & Rsh)')
        ]
        
        for key, title in plot_order:
            if key in img_paths and img_paths[key]:
                slide = prs.slides.add_slide(prs.slide_layouts[5])
                slide.shapes.title.text = title
                slide.shapes.title.text_frame.paragraphs[0].font.size = PptPt(24)
                slide.shapes.add_picture(str(img_paths[key]), PptInches(2.5), PptInches(1.2),
                                        height=PptInches(5.8))

        prs.save(self.output_dir / self.config.report_pptx_name)

    # --- TABLE GENERATION METHODS ---

    @staticmethod
    def _add_word_table(doc: Document, df: pd.DataFrame, columns: list, title: str) -> None:
        """Add formatted table to Word document."""
        doc.add_heading(title, level=2)
        if df.empty or not columns:
            return

        table = doc.add_table(rows=1, cols=len(columns))
        table.style = 'Table Grid'
        hdr_cells = table.rows[0].cells

        for i, col in enumerate(columns):
            header_text = str(col).replace('_', ' (') + ')' if '_' in str(col) else str(col)
            hdr_cells[i].text = header_text
            p = hdr_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.runs[0]
            run.font.bold = True
            run.font.size = Pt(9)
            hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        for _, row in df.iterrows():
            row_cells = table.add_row().cells
            for i, col in enumerate(columns):
                val = row.get(col, "")
                if isinstance(val, (int, float)):
                    txt = f"{val:.0f}" if 'Count' in col else f"{val:.2f}"
                else:
                    txt = str(val)

                row_cells[i].text = txt
                p = row_cells[i].paragraphs[0]
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.runs[0].font.size = Pt(9)
                row_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        doc.add_paragraph()

    @staticmethod
    def _add_pptx_table(prs: Presentation, df: pd.DataFrame, columns: list, title: str) -> None:
        """Add formatted table to PowerPoint presentation."""
        if df.empty or not columns:
            return
        max_rows = 10
        total_rows = len(df)
        num_slides = math.ceil(total_rows / max_rows)

        for i in range(num_slides):
            start = i * max_rows
            end = min((i + 1) * max_rows, total_rows)
            chunk = df.iloc[start:end]

            slide = prs.slides.add_slide(prs.slide_layouts[5])
            slide.shapes.title.text = f"{title} ({i + 1}/{num_slides})" if num_slides > 1 else title
            slide.shapes.title.text_frame.paragraphs[0].font.size = PptPt(24)

            rows, cols = len(chunk) + 1, len(columns)
            left, top, width, height = PptInches(0.5), PptInches(1.5), PptInches(12.0), PptInches(0.5 * rows)

            table = slide.shapes.add_table(rows, cols, left, top, width, height).table

            for c_idx, col_name in enumerate(columns):
                cell = table.cell(0, c_idx)
                header_text = str(col_name).replace('_', ' (') + ')' if '_' in str(col_name) else str(col_name)
                cell.text = header_text
                p = cell.text_frame.paragraphs[0]
                p.font.bold = True
                p.font.size = PptPt(12)
                p.alignment = PP_ALIGN.CENTER
                cell.vertical_anchor = MSO_ANCHOR.MIDDLE

            for r_idx in range(len(chunk)):
                row_data = chunk.iloc[r_idx]
                for c_idx, col_name in enumerate(columns):
                    cell = table.cell(r_idx + 1, c_idx)
                    val = row_data.get(col_name, "")
                    if isinstance(val, (int, float)):
                        txt = f"{val:.0f}" if 'Count' in col_name else f"{val:.2f}"
                    else:
                        txt = str(val)

                    cell.text = txt
                    p = cell.text_frame.paragraphs[0]
                    p.font.size = PptPt(11)
                    p.alignment = PP_ALIGN.CENTER
                    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
