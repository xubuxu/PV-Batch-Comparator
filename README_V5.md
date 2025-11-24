# IV Batch Analyzer V5.0 - Professional Edition

🎉 **Modern, Professional-Grade Solar Cell IV Analysis Tool**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ New in V5.0

### 🎨 Modern UI with CustomTkinter
- **Dark Mode by Default** with light theme toggle
- **3-Tab Interface**: Dashboard, Live Log, and Plot Preview
- **Real-time Progress Tracking** with visual progress bar
- **Thread-Safe Logging** with queue-based updates
- **Embedded Plot Preview** using matplotlib integration

### 🚀 Enhanced Robustness
- **Smart CSV Parsing**: Auto-detects delimiters (comma, semicolon, tab)
- **Header Row Detection**: Automatically skips metadata from lab equipment files  
- **Config Persistence**: Saves your settings to `config.json` across sessions
- **Stop/Cancel Functionality**: Safely interrupt analysis at any time
- **Improved Error Handling**: Specific exceptions with actionable guidance

### 📊 Preserved Features
- All existing calculation logic (statistics, outlier removal, T-tests)
- Complete report generation (Excel, Word, PowerPoint)
- Advanced visualizations (8 plot types)
- Batch processing with natural sorting

### 🔬 Advanced Analysis Features (Optional)
- **Single-Diode Model Fitting**: Two-step approach for Rs/Rsh extraction
  - Slope-based method (robust, always available)
  - Lambert W fitting (high accuracy, quality-gated)
- **Hysteresis Analysis**: Critical for perovskite solar cells
  - Hysteresis Index calculation
  - Forward/Reverse scan comparison
- **S-Shape Detection**: Automatic anomaly identification in IV curves

> **⚠️ Important Note on Physics Modeling**
>
> This tool uses Single Diode Model approximation for parameter extraction, optimized for batch trend analysis. For precise physics modeling of tandem devices (e.g., sub-cell recombination breakdown), specialized simulation software is recommended.

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `customtkinter>=5.0.0` - Modern GUI framework
- `pandas>=1.5.0` - Data processing
- `matplotlib>=3.6.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualizations
- `scipy>=1.9.0` - Scientific computing
- `python-docx>=0.8.11` - Word reports
- `python-pptx>=0.6.21` - PowerPoint reports
- `openpyxl>=3.0.0` - Excel export

## 🚀 Quick Start

### Option 1: GUI Mode (Recommended)
```bash
python main.py
```

### Option 2: Programmatic Use
```python
from src.analyzer import IVBatchAnalyzer
from src.config import AnalyzerConfig

# Create analyzer
analyzer = IVBatchAnalyzer(
    root_dir="/path/to/data",
    config=AnalyzerConfig(
        scan_direction="Reverse",
        outlier_removal=True
    ),
    theme="dark"
)

# Run analysis
output_dir = analyzer.run()
print(f"Results saved to: {output_dir}")
```

## 📁 Project Structure

```
IV_Batch_Analyzer/
├── main.py                  # GUI application entry point
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration & constants
│   ├── data_loader.py       # Smart CSV parsing
│   ├── statistics.py        # Data cleaning & analysis
│   ├── visualizer.py        # Plot generation
│   ├── reporter.py          # Excel/Word/PPT export
│   └── analyzer.py          # Main controller
├── requirements.txt         # Python dependencies
├── iv_analyzer.py          # Legacy V4.2 (backup)
└── README.md               # This file
```

## 🎯 Usage Guide

### Dashboard Tab
1. **Select Data Folder**: Choose root directory containing your IV measurement CSV files
2. **Configure Parameters**:
   - Scan Direction: Reverse/Forward/All
   - Minimum thresholds for Eff, Voc, Jsc
   - FF range (10-90% default)
   - Toggle duplicate/outlier removal
3. **Run Analysis**: Click the green "▶ Run Analysis" button
4. **Monitor Progress**: Watch the progress bar and status messages
5. **Open Results**: Click "📁 Open Output Folder" when complete

### Live Log Tab
- View real-time analysis progress
- Color-coded log levels (INFO, WARNING, ERROR)
- Auto-scrolling with timestamps
- Clear log button

### Preview Tab
- Select from generated plots using dropdown
- Zoom/pan controls via matplotlib toolbar
- Instant preview without opening files

### Theme Toggle
- Click "🌙 Toggle Theme" to switch between dark and light modes
- Preference saved automatically

## 📊 Input File Format

**Supported Formats:**
- CSV files with common delimiters (comma, semicolon, tab)
- Automatic header row detection
- Flexible column naming (see `COLUMN_MAPPING` in `src/config.py`)

**Required Columns** (case-insensitive):
- Efficiency: `Eff`, `Efficiency`, `PCE`, `Eta`
- Open Circuit Voltage: `Voc`, `Uoc`
- Short Circuit Current: `Jsc`, `Isc`
- Fill Factor: `FF`

**Optional Columns:**
- Series Resistance: `Rs`
- Shunt Resistance: `Rsh`, `Rp`
- Scan Direction: `ScanDirection`, `Direction`
- Cell Name: `CellName`, `Device_ID`, `Sample_Name`

## 📈 Output Reports

Analysis generates a timestamped folder with:

### 📊 Excel Workbook (`IV_Processed_Data.xlsx`)
- Cleaned Data
- Statistics Summary
- Champion Cells
- Top 10 Cells
- Yield Distribution

### 📄 Word Report (`IV_Analysis_Report.docx`)
- Executive Summary with T-test results
- Data Tables
- All 8 plots with captions

### 📽️ PowerPoint Slides (`IV_Analysis_Slides.pptx`)
- Professional presentation-ready slides
- Title and summary slides
- Data tables (paginated for large datasets)
- Full-resolution plots

### 🖼️ Plot Files (PNG, 300 DPI)
1. `1_Boxplot.png` - Parameter distributions
2. `1_Histogram.png` - Efficiency histogram
3. `2_Trend.png` - Batch trend analysis
4. `3_Yield.png` - Efficiency yield distribution
5. `4_JV_Curves.png` - J-V curves of champion cells
6. `5_Voc_Jsc_Tradeoff.png` - Voc vs Jsc correlation
7. `6_Drivers.png` - Efficiency driver correlations
8. `7_Resistance.png` - Parasitic resistance analysis

## 🛠️ Configuration

Settings are automatically saved to `~/.iv_analyzer/config.json`:

```json
{
  "last_folder": "/path/to/data",
  "scan_direction": "Reverse",
  "thresholds": {
    "Eff_Min": 0.1,
    "Voc_Min": 0.1,
    "Jsc_Min": 0.1,
    "FF_Min": 10.0,
    "FF_Max": 90.0
  },
  "remove_duplicates": true,
  "outlier_removal": true,
  "theme": "dark",
  "window_geometry": "900x700"
}
```

## 🔧 Troubleshooting

**CSV Parsing Issues:**
- Ensure files contain required columns
- Check for metadata rows (automatically skipped if detected)
- Verify encoding (UTF-8, GBK, Latin1 supported)

**GUI Not Opening:**
```bash
# Test dependencies
python -c "import customtkinter; print('OK')"

# Check Python version
python --version  # Requires 3.8+
```

**Analysis Errors:**
- Check log tab for detailed error messages
- Verify data folder contains CSV files matching patterns
- Ensure sufficient permissions to read files and write output

## 📜 License

MIT License - Free to use and modify

## 🙏 Acknowledgments

Built with:
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - Modern UI framework
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization
- [python-docx](https://python-docx.readthedocs.io/) & [python-pptx](https://python-pptx.read thedocs.io/) - Report generation

## 📞 Support

For issues or questions, please review the log output in the Live Log tab for detailed error messages.

---

**IV Batch Analyzer V5.0 - Professional Edition**  
*Making solar cell analysis beautiful, fast, and reliable* ☀️
