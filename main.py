"""
IV Batch Analyzer V5.0 - Professional Edition
Main Application with Modern GUI

CustomTkinter-based GUI with tabbed interface, progress tracking, and theme support.
"""

import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, scrolledtext, messagebox
from typing import Optional

import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for preview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import our modules
from src.config import ConfigManager, AnalyzerConfig, THEMES, NAMING_PATTERNS
from src.analyzer import IVBatchAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ================= THREAD-SAFE LOGGING HANDLER =================

class QueueHandler(logging.Handler):
    """Thread-safe logging handler that uses a queue."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


# ================= MAIN GUI CLASS =================

class IVAnalyzerGUI:
    """Main GUI Application Class."""

    def __init__(self):
        # Initialize Config
        self.config_manager = ConfigManager()
        self.user_config = self.config_manager.load_config()
        
        # Setup Window
        ctk.set_appearance_mode(self.user_config.get("theme", "Dark"))
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("IV Batch Analyzer V5.0 - Professional Edition")
        self.root.geometry(self.user_config.get("window_geometry", "1100x800"))
        
        # State Variables
        self.root_dir_var = ctk.StringVar(value=self.user_config.get("last_folder", ""))
        self.progress_var = ctk.DoubleVar(value=0.0)
        self.status_var = ctk.StringVar(value="Ready")
        self.scan_dir_var = ctk.StringVar(value=self.user_config.get("scan_direction", "Reverse"))
        self.remove_dupes_var = ctk.BooleanVar(value=self.user_config.get("remove_duplicates", True))
        self.outlier_var = ctk.BooleanVar(value=self.user_config.get("outlier_removal", True))
        
        # Advanced Analysis Variables
        self.enable_advanced_var = ctk.BooleanVar(value=self.user_config.get("enable_advanced_analysis", False))
        self.ff_threshold_var = ctk.DoubleVar(value=self.user_config.get("ff_threshold_for_fitting", 40.0))
        
        # Plotting Theme
        self.plot_theme_var = ctk.StringVar(value=self.user_config.get("plot_theme", "Dark"))
        
        # Naming Convention
        self.naming_var = ctk.StringVar(value=self.user_config.get("naming_convention", "Traceable"))
        
        # Granular Control Variables
        self.champion_criteria_var = ctk.StringVar(value=self.user_config.get("champion_criteria", "Max Eff"))
        self.resistance_method_var = ctk.StringVar(value=self.user_config.get("resistance_method", "Slope"))
        
        # Output Formats
        formats = self.user_config.get("output_formats", ["png"])
        self.fmt_png_var = ctk.BooleanVar(value="png" in formats)
        self.fmt_svg_var = ctk.BooleanVar(value="svg" in formats)
        self.fmt_pdf_var = ctk.BooleanVar(value="pdf" in formats)
        
        # Report Types
        reports = self.user_config.get("report_types", ["excel", "word", "pptx"])
        self.rpt_excel_var = ctk.BooleanVar(value="excel" in reports)
        self.rpt_word_var = ctk.BooleanVar(value="word" in reports)
        self.rpt_pptx_var = ctk.BooleanVar(value="pptx" in reports)
        
        # Plot Selection Variables
        selected_plots = self.user_config.get("selected_plots", [
            "box", "hist", "trend", "yield", "jv_curve", "voc_jsc", 
            "combo_drivers", "resistance", "model_fitting", "hysteresis", "anomalies"
        ])
        self.plot_vars = {
            "box": ctk.BooleanVar(value="box" in selected_plots),
            "hist": ctk.BooleanVar(value="hist" in selected_plots),
            "trend": ctk.BooleanVar(value="trend" in selected_plots),
            "yield": ctk.BooleanVar(value="yield" in selected_plots),
            "jv_curve": ctk.BooleanVar(value="jv_curve" in selected_plots),
            "voc_jsc": ctk.BooleanVar(value="voc_jsc" in selected_plots),
            "combo_drivers": ctk.BooleanVar(value="combo_drivers" in selected_plots),
            "resistance": ctk.BooleanVar(value="resistance" in selected_plots),
            "model_fitting": ctk.BooleanVar(value="model_fitting" in selected_plots),
            "hysteresis": ctk.BooleanVar(value="hysteresis" in selected_plots),
            "anomalies": ctk.BooleanVar(value="anomalies" in selected_plots)
        }
        
        # Threading & Logging
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        logging.getLogger().addHandler(self.queue_handler)
        self.analyzer_thread = None
        self.stop_event = threading.Event()
        
        # Build UI
        self._create_widgets()
        
        # Start log poller
        self.root.after(100, self._poll_log_queue)

    def _create_widgets(self):
        """Create the main GUI layout."""
        # Main Container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="IV Batch Analyzer Professional", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Tabview
        self.tabview = ctk.CTkTabview(main_frame)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_dashboard = self.tabview.add("📊 Dashboard")
        self.tab_advanced = self.tabview.add("⚙️ Advanced")
        self.tab_log = self.tabview.add("📝 Live Log")
        self.tab_preview = self.tabview.add("🖼️ Preview")
        
        # Setup each tab
        self._setup_dashboard_tab()
        self._setup_advanced_tab()
        self._setup_log_tab()
        self._setup_preview_tab()
        
        # Status Bar
        self._create_status_bar(main_frame)

    def _setup_dashboard_tab(self):
        """Setup Dashboard tab."""
        # 1. Folder Selection
        folder_frame = ctk.CTkFrame(self.tab_dashboard)
        folder_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(folder_frame, text="Data Folder:").pack(side="left", padx=10)
        ctk.CTkEntry(folder_frame, textvariable=self.root_dir_var, width=400).pack(side="left", padx=10)
        ctk.CTkButton(folder_frame, text="Browse", command=self._browse_folder).pack(side="left", padx=10)
        
        # 2. Basic Settings
        settings_frame = ctk.CTkFrame(self.tab_dashboard)
        settings_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Scan Direction:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(settings_frame, values=["Reverse", "Forward"], variable=self.scan_dir_var).grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkCheckBox(settings_frame, text="Remove Duplicates", variable=self.remove_dupes_var).grid(row=0, column=2, padx=10, pady=10)
        ctk.CTkCheckBox(settings_frame, text="Outlier Removal (IQR)", variable=self.outlier_var).grid(row=0, column=3, padx=10, pady=10)
        
        # 3. Theme & Naming
        style_frame = ctk.CTkFrame(self.tab_dashboard)
        style_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(style_frame, text="Plotting Theme:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(style_frame, values=list(THEMES.keys()), variable=self.plot_theme_var).grid(row=0, column=1, padx=10, pady=10)
        
        ctk.CTkLabel(style_frame, text="Naming:").grid(row=0, column=2, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(style_frame, values=list(NAMING_PATTERNS.keys()), variable=self.naming_var).grid(row=0, column=3, padx=10, pady=10)
        
        # 4. Advanced Toggle
        adv_frame = ctk.CTkFrame(self.tab_dashboard)
        adv_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkCheckBox(adv_frame, text="Enable Advanced Analysis (Physics & Hysteresis)", 
                       variable=self.enable_advanced_var, command=self._toggle_advanced_options).pack(side="left", padx=10, pady=10)
        
        self.ff_threshold_entry = ctk.CTkEntry(adv_frame, textvariable=self.ff_threshold_var, width=60)
        self.ff_label = ctk.CTkLabel(adv_frame, text="Min FF for Fitting (%):")
        
        if self.enable_advanced_var.get():
            self.ff_label.pack(side="left", padx=5)
            self.ff_threshold_entry.pack(side="left", padx=5)
            
        # 5. Action Buttons
        btn_frame = ctk.CTkFrame(self.tab_dashboard, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        self.run_btn = ctk.CTkButton(
            btn_frame, text="🚀 Run Analysis", 
            command=self._start_analysis_thread,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=40, width=200
        )
        self.run_btn.pack(side="left", padx=20)
        
        self.stop_btn = ctk.CTkButton(
            btn_frame, text="🛑 Stop", 
            command=self._stop_analysis,
            fg_color="#E64B35", hover_color="#C0392B",
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=20)

    def _setup_advanced_tab(self):
        """Setup Advanced Settings tab."""
        # 1. Processing Options
        proc_frame = ctk.CTkFrame(self.tab_advanced)
        proc_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(proc_frame, text="Processing Options", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        ctk.CTkLabel(proc_frame, text="Champion Criteria:").pack(side="left", padx=10, pady=10)
        ctk.CTkComboBox(proc_frame, values=["Max Eff", "Max FF"], variable=self.champion_criteria_var).pack(side="left", padx=10)
        
        ctk.CTkLabel(proc_frame, text="Resistance Method:").pack(side="left", padx=10, pady=10)
        ctk.CTkComboBox(proc_frame, values=["Slope", "Fitting"], variable=self.resistance_method_var).pack(side="left", padx=10)
        
        # 2. Output Control
        out_frame = ctk.CTkFrame(self.tab_advanced)
        out_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(out_frame, text="Output Control", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # Formats
        fmt_frame = ctk.CTkFrame(out_frame)
        fmt_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(fmt_frame, text="Image Formats:").pack(side="left", padx=10)
        ctk.CTkCheckBox(fmt_frame, text="PNG", variable=self.fmt_png_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(fmt_frame, text="SVG", variable=self.fmt_svg_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(fmt_frame, text="PDF", variable=self.fmt_pdf_var).pack(side="left", padx=5)
        
        # Reports
        rpt_frame = ctk.CTkFrame(out_frame)
        rpt_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(rpt_frame, text="Report Types:").pack(side="left", padx=10)
        ctk.CTkCheckBox(rpt_frame, text="Excel", variable=self.rpt_excel_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(rpt_frame, text="Word", variable=self.rpt_word_var).pack(side="left", padx=5)
        ctk.CTkCheckBox(rpt_frame, text="PowerPoint", variable=self.rpt_pptx_var).pack(side="left", padx=5)
        
        # 3. Plot Selection
        plot_frame = ctk.CTkFrame(self.tab_advanced)
        plot_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        ctk.CTkLabel(plot_frame, text="Plot Selection", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=5)
        
        # Grid layout for plots
        grid_frame = ctk.CTkFrame(plot_frame)
        grid_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        plots = [
            ("Boxplots", "box"), ("Histograms", "hist"), ("Trend Analysis", "trend"),
            ("Yield Analysis", "yield"), ("JV Curves", "jv_curve"), ("Voc-Jsc Tradeoff", "voc_jsc"),
            ("Driver Analysis", "combo_drivers"), ("Resistance Analysis", "resistance"),
            ("Model Fitting", "model_fitting"), ("Hysteresis", "hysteresis"), ("Anomalies", "anomalies")
        ]
        
        for i, (label, key) in enumerate(plots):
            row = i // 3
            col = i % 3
            ctk.CTkCheckBox(grid_frame, text=label, variable=self.plot_vars[key]).grid(row=row, column=col, padx=10, pady=10, sticky="w")

    def _setup_log_tab(self):
        """Setup Log tab."""
        self.log_text = scrolledtext.ScrolledText(self.tab_log, state='disabled', height=20)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure tags for coloring
        self.log_text.tag_config('INFO', foreground='white')
        self.log_text.tag_config('WARNING', foreground='orange')
        self.log_text.tag_config('ERROR', foreground='red')

    def _setup_preview_tab(self):
        """Setup Preview tab."""
        self.preview_frame = ctk.CTkFrame(self.tab_preview)
        self.preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Dropdown to select plot
        self.preview_var = ctk.StringVar(value="Select Plot")
        self.preview_combo = ctk.CTkComboBox(
            self.tab_preview, 
            variable=self.preview_var,
            values=["Run analysis first..."],
            command=self._update_preview
        )
        self.preview_combo.pack(pady=10)
        
        # Canvas placeholder
        self.canvas = None

    def _create_status_bar(self, parent):
        """Create status bar with progress."""
        status_frame = ctk.CTkFrame(parent, height=30)
        status_frame.pack(fill="x", side="bottom", padx=10, pady=5)
        
        self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var)
        self.status_label.pack(side="left", padx=10)
        
        self.progress_bar = ctk.CTkProgressBar(status_frame, variable=self.progress_var)
        self.progress_bar.pack(side="right", padx=10, fill="x", expand=True)

    def _toggle_advanced_options(self):
        """Show/hide advanced options."""
        if self.enable_advanced_var.get():
            self.ff_label.pack(side="left", padx=5)
            self.ff_threshold_entry.pack(side="left", padx=5)
        else:
            self.ff_label.pack_forget()
            self.ff_threshold_entry.pack_forget()

    def _browse_folder(self):
        """Open folder dialog."""
        folder = filedialog.askdirectory()
        if folder:
            self.root_dir_var.set(folder)

    def _poll_log_queue(self):
        """Poll log queue and update text widget."""
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.log_text.configure(state='normal')
            
            tag = 'INFO'
            if 'WARNING' in msg: tag = 'WARNING'
            if 'ERROR' in msg: tag = 'ERROR'
            
            self.log_text.insert('end', msg + '\n', tag)
            self.log_text.see('end')
            self.log_text.configure(state='disabled')
        
        self.root.after(100, self._poll_log_queue)

    def _start_analysis_thread(self):
        """Start analysis in a separate thread."""
        root_dir = self.root_dir_var.get()
        if not root_dir:
            messagebox.showwarning("Input Error", "Please select a data folder.")
            return
        
        self._save_current_config()
        
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress_var.set(0)
        self.status_var.set("Starting analysis...")
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, 'end')
        self.log_text.configure(state='disabled')
        
        self.stop_event.clear()
        
        self.analyzer_thread = threading.Thread(target=self._run_analysis, args=(root_dir,), daemon=True)
        self.analyzer_thread.start()

    def _run_analysis(self, root_dir):
        """Run the analysis logic."""
        try:
            # Create Config
            config = AnalyzerConfig(
                scan_direction=self.scan_dir_var.get(),
                remove_duplicates=self.remove_dupes_var.get(),
                outlier_removal=self.outlier_var.get(),
                plot_theme=self.plot_theme_var.get(),
                naming_convention=self.naming_var.get(),
                enable_advanced_analysis=self.enable_advanced_var.get(),
                ff_threshold_for_fitting=self.ff_threshold_var.get(),
                champion_criteria=self.champion_criteria_var.get(),
                resistance_method=self.resistance_method_var.get(),
                output_formats=tuple([f for f, v in [("png", self.fmt_png_var), ("svg", self.fmt_svg_var), ("pdf", self.fmt_pdf_var)] if v.get()]),
                report_types=tuple([r for r, v in [("excel", self.rpt_excel_var), ("word", self.rpt_word_var), ("pptx", self.rpt_pptx_var)] if v.get()]),
                selected_plots=tuple([k for k, v in self.plot_vars.items() if v.get()])
            )
            
            analyzer = IVBatchAnalyzer(root_dir, config, theme=self.user_config.get("theme", "dark"))
            
            # Hook up progress
            def progress_callback(percent, message):
                self.root.after(0, lambda: self.progress_var.set(percent / 100))
                self.root.after(0, lambda: self.status_var.set(message))
            
            analyzer.set_progress_callback(progress_callback)
            
            # Run
            output_dir = analyzer.run()
            
            if output_dir:
                self.root.after(0, lambda: self._on_analysis_complete(output_dir))
            else:
                self.root.after(0, lambda: self._on_analysis_failed())
                
        except Exception as e:
            logger.error(f"Thread error: {e}", exc_info=True)
            self.root.after(0, lambda: self._on_analysis_failed())

    def _stop_analysis(self):
        """Request stop."""
        # Note: In a real implementation, we'd pass the stop signal to the analyzer
        # For now, we just update UI
        self.status_var.set("Stopping...")
        # The analyzer class needs to check a flag periodically
        # We can implement this by passing a stop_event to the analyzer if we modify it
        # For now, we'll just disable the button
        self.stop_btn.configure(state="disabled")

    def _on_analysis_complete(self, output_dir):
        """Handle completion."""
        self.status_var.set("Analysis Complete!")
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_var.set(1.0)
        
        messagebox.showinfo("Success", f"Analysis complete!\nOutput: {output_dir}")
        
        # Update preview dropdown
        self._load_preview_images(output_dir)
        
        # Open folder
        if os.name == 'nt':
            os.startfile(output_dir)

    def _on_analysis_failed(self):
        """Handle failure."""
        self.status_var.set("Analysis Failed or Stopped.")
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress_var.set(0)

    def _load_preview_images(self, output_dir):
        """Load generated images into preview dropdown."""
        self.preview_images = {}
        try:
            for file in output_dir.glob("*.png"):
                self.preview_images[file.stem] = file
            
            if self.preview_images:
                self.preview_combo.configure(values=list(self.preview_images.keys()))
                self.preview_combo.set(list(self.preview_images.keys())[0])
                self._update_preview(self.preview_combo.get())
        except Exception as e:
            logger.error(f"Failed to load previews: {e}")

    def _update_preview(self, choice):
        """Update the matplotlib canvas with selected image."""
        if not hasattr(self, 'preview_images') or choice not in self.preview_images:
            return
            
        path = self.preview_images[choice]
        
        # Clear old canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            
        # Load image using matplotlib to display in Tkinter
        try:
            img = matplotlib.image.imread(path)
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis('off')
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.preview_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            logger.error(f"Preview failed: {e}")

    def _save_current_config(self):
        """Save current UI state to config."""
        self.user_config.update({
            "last_folder": self.root_dir_var.get(),
            "scan_direction": self.scan_dir_var.get(),
            "remove_duplicates": self.remove_dupes_var.get(),
            "outlier_removal": self.outlier_var.get(),
            "plot_theme": self.plot_theme_var.get(),
            "naming_convention": self.naming_var.get(),
            "enable_advanced_analysis": self.enable_advanced_var.get(),
            "ff_threshold_for_fitting": self.ff_threshold_var.get(),
            "champion_criteria": self.champion_criteria_var.get(),
            "resistance_method": self.resistance_method_var.get(),
            "output_formats": [f for f, v in [("png", self.fmt_png_var), ("svg", self.fmt_svg_var), ("pdf", self.fmt_pdf_var)] if v.get()],
            "report_types": [r for r, v in [("excel", self.rpt_excel_var), ("word", self.rpt_word_var), ("pptx", self.rpt_pptx_var)] if v.get()],
            "selected_plots": [k for k, v in self.plot_vars.items() if v.get()]
        })
        self.config_manager.save_config(self.user_config)

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


# ================= MAIN ENTRY POINT =================

def main():
    """Main entry point."""
    app = IVAnalyzerGUI()
    app.run()


if __name__ == "__main__":
    main()
