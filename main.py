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
from tkinter import filedialog, scrolledtext
from typing import Optional

import customtkinter as ctk
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for preview
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import our modules
from src.config import ConfigManager, AnalyzerConfig
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


# ================= MAIN GUI APPLICATION =================

class IVAnalyzerGUI:
    """Main GUI application with modern CustomTkinter interface."""
    
    def __init__(self):
        """Initialize the GUI application."""
        # Config manager
        self.config_manager = ConfigManager()
        self.user_config = self.config_manager.load_config()
        
        # Main window
        ctk.set_appearance_mode(self.user_config.get("theme", "dark"))
        ctk.set_default_color_theme("blue")
        
        self.root = ctk.CTk()
        self.root.title("IV Batch Analyzer V5.0 - Professional Edition")
        self.root.geometry(self.user_config.get("window_geometry", "900x700"))
        
        # Variables
        self.folder_path = ctk.StringVar(value=self.user_config.get("last_folder", str(Path.cwd())))
        self.scan_dir_var = ctk.StringVar(value=self.user_config.get("scan_direction", "Reverse"))
        self.eff_min_var = ctk.DoubleVar(value=self.user_config.get("thresholds", {}).get("Eff_Min", 0.1))
        self.voc_min_var = ctk.DoubleVar(value=self.user_config.get("thresholds", {}).get("Voc_Min", 0.1))
        self.jsc_min_var = ctk.DoubleVar(value=self.user_config.get("thresholds", {}).get("Jsc_Min", 0.1))
        self.ff_min_var = ctk.DoubleVar(value=self.user_config.get("thresholds", {}).get("FF_Min", 10.0))
        self.ff_max_var = ctk.DoubleVar(value=self.user_config.get("thresholds", {}).get("FF_Max", 90.0))
        self.remove_duplicates_var = ctk.BooleanVar(value=self.user_config.get("remove_duplicates", True))
        self.remove_outliers_var = ctk.BooleanVar(value=self.user_config.get("outlier_removal", True))
        
        # State
        self.analyzer_instance: Optional[IVBatchAnalyzer] = None
        self.output_folder: Optional[Path] = None
        self.is_running = False
        self.log_queue = queue.Queue()
        self.plot_paths = {}
        
        # Setup UI
        self._setup_ui()
        self._setup_logging()
        
        # Start log queue polling
        self.root.after(100, self._poll_log_queue)
        
        # Save config on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_ui(self):
        """Setup the complete UI with tabs."""
        # Create tabview
        self.tabview = ctk.CTkTabview(self.root)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add tabs
        self.tab_dashboard = self.tabview.add("📊 Dashboard")
        self.tab_log = self.tabview.add("📝 Live Log")
        self.tab_preview = self.tabview.add("🖼️ Preview")
        
        # Setup each tab
        self._setup_dashboard_tab()
        self._setup_log_tab()
        self._setup_preview_tab()
    
    def _setup_dashboard_tab(self):
        """Setup Dashboard tab with configuration and controls."""
        # Folder Selection Section
        folder_frame = ctk.CTkFrame(self.tab_dashboard)
        folder_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            folder_frame, 
            text="📂 Select Data Folder", 
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=(10, 5))
        
        folder_entry_frame = ctk.CTkFrame(folder_frame)
        folder_entry_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkEntry(
            folder_entry_frame,
            textvariable=self.folder_path,
            width=500,
            height=40,
            font=ctk.CTkFont(size=13)
        ).pack(side="left", padx=(0, 10), expand=True, fill="x")
        
        ctk.CTkButton(
            folder_entry_frame,
            text="Browse",
            command=self.select_folder,
            width=100,
            height=40
        ).pack(side="right")
        
        # Configuration Panel
        config_frame = ctk.CTkFrame(self.tab_dashboard)
        config_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(
            config_frame,
            text="⚙️ Analysis Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=(15, 10))
        
        # Grid for config options
        options_frame = ctk.CTkFrame(config_frame)
        options_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Row 0: Scan Direction
        ctk.CTkLabel(options_frame, text="Scan Direction:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkComboBox(
            options_frame,
            values=["Reverse", "Forward", "All"],
            variable=self.scan_dir_var,
            width=150
        ).grid(row=0, column=1, padx=10, pady=10, sticky="w")
        
        # Row 1-2: Thresholds
        ctk.CTkLabel(options_frame, text="Min Efficiency (%):").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(options_frame, textvariable=self.eff_min_var, width=100).grid(row=1, column=1, padx=10, pady=10, sticky="w")
        
        ctk.CTkLabel(options_frame, text="Min Voc (V):").grid(row=1, column=2, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(options_frame, textvariable=self.voc_min_var, width=100).grid(row=1, column=3, padx=10, pady=10, sticky="w")
        
        ctk.CTkLabel(options_frame, text="Min Jsc (mA/cm²):").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(options_frame, textvariable=self.jsc_min_var, width=100).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        
        ctk.CTkLabel(options_frame, text="FF Range (%):").grid(row=2, column=2, padx=10, pady=10, sticky="w")
        ff_range_frame = ctk.CTkFrame(options_frame)
        ff_range_frame.grid(row=2, column=3, padx=10, pady=10, sticky="w")
        ctk.CTkEntry(ff_range_frame, textvariable=self.ff_min_var, width=60).pack(side="left")
        ctk.CTkLabel(ff_range_frame, text=" - ").pack(side="left")
        ctk.CTkEntry(ff_range_frame, textvariable=self.ff_max_var, width=60).pack(side="left")
        
        # Row 3: Checkboxes
        checkbox_frame = ctk.CTkFrame(options_frame)
        checkbox_frame.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="w")
        
        ctk.CTkCheckBox(checkbox_frame, text="Remove Duplicates", variable=self.remove_duplicates_var).pack(side="left", padx=10)
        ctk.CTkCheckBox(checkbox_frame, text="Remove Outliers", variable=self.remove_outliers_var).pack(side="left", padx=10)
        
        # Action Buttons
        button_frame = ctk.CTkFrame(self.tab_dashboard)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        # Run Analysis Button (large, centered)
        self.run_button = ctk.CTkButton(
            button_frame,
            text="▶ Run Analysis",
            command=self.start_analysis,
            width=300,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="#14a44d",
            hover_color="#0d7a3a"
        )
        self.run_button.pack(pady=10)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(button_frame, width=400)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(button_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.progress_label.pack()
        
        # Stop and Open Folder buttons
        action_buttons_frame = ctk.CTkFrame(button_frame)
        action_buttons_frame.pack(pady=10)
        
        self.stop_button = ctk.CTkButton(
            action_buttons_frame,
            text="⏹ Stop",
            command=self.stop_analysis,
            width=120,
            fg_color="#dc3545",
            hover_color="#b02a37",
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=5)
        
        self.open_folder_button = ctk.CTkButton(
            action_buttons_frame,
            text="📁 Open Output Folder",
            command=self.open_output_folder,
            width=200,
            state="disabled"
        )
        self.open_folder_button.pack(side="left", padx=5)
        
        # Theme toggle
        self.theme_button = ctk.CTkButton(
            action_buttons_frame,
            text="🌙 Toggle Theme",
            command=self.toggle_theme,
            width=150
        )
        self.theme_button.pack(side="left", padx=5)
    
    def _setup_log_tab(self):
        """Setup Live Log tab."""
        log_frame = ctk.CTkFrame(self.tab_log)
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            state='disabled',
            font=("Consolas", 10),
            bg="#1e1e1e" if self.user_config.get("theme", "dark") == "dark" else "white",
            fg="#e0e0e0" if self.user_config.get("theme", "dark") == "dark" else "black"
        )
        self.log_text.pack(fill="both", expand=True)
        
        # Log controls
        log_controls = ctk.CTkFrame(self.tab_log)
        log_controls.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(
            log_controls,
            text="Clear Log",
            command=self.clear_log,
            width=100
        ).pack(side="left", padx=5)
    
    def _setup_preview_tab(self):
        """Setup Preview tab with matplotlib figure embedding."""
        preview_frame = ctk.CTkFrame(self.tab_preview)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Plot selector
        selector_frame = ctk.CTkFrame(preview_frame)
        selector_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(selector_frame, text="Select Plot:").pack(side="left", padx=10)
        
        self.plot_selector = ctk.CTkComboBox(
            selector_frame,
            values=["No plots available"],
            command=self.update_preview,
            width=300
        )
        self.plot_selector.pack(side="left", padx=10)
        
        ctk.CTkButton(
            selector_frame,
            text="Refresh",
            command=lambda: self.update_preview(self.plot_selector.get()),
            width=100
        ).pack(side="left", padx=10)
        
        # Canvas for matplotlib figure
        self.preview_canvas_frame = ctk.CTkFrame(preview_frame)
        self.preview_canvas_frame.pack(fill="both", expand=True)
        
        # Placeholder
        self.preview_label = ctk.CTkLabel(
            self.preview_canvas_frame,
            text="Preview will appear here after analysis completes",
            font=ctk.CTkFont(size=14)
        )
        self.preview_label.pack(expand=True)
        
        self.canvas = None
        self.toolbar = None
    
    def _setup_logging(self):
        """Setup logging with queue handler."""
        queue_handler = QueueHandler(self.log_queue)
        queue_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(queue_handler)
    
    def _poll_log_queue(self):
        """Poll the log queue and update UI."""
        while True:
            try:
                message = self.log_queue.get_nowait()
                self.log_text.configure(state='normal')
                self.log_text.insert("end", message + '\n')
                self.log_text.see("end")
                self.log_text.configure(state='disabled')
            except queue.Empty:
                break
        
        self.root.after(100, self._poll_log_queue)
    
    # === EVENT HANDLERS ===
    
    def select_folder(self):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(initialdir=self.folder_path.get())
        if folder:
            self.folder_path.set(folder)
    
    def start_analysis(self):
        """Start analysis in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.open_folder_button.configure(state="disabled")
        self.progress_bar.set(0)
        
        threading.Thread(target=self._run_analysis, daemon=True).start()
    
    def stop_analysis(self):
        """Stop the running analysis."""
        if self.analyzer_instance:
            self.analyzer_instance.stop_requested = True
            logger.info("Stop requested by user...")
    
    def _run_analysis(self):
        """Run analysis (in background thread)."""
        try:
            # Build config
            config = AnalyzerConfig(
                scan_direction=self.scan_dir_var.get(),
                remove_duplicates=self.remove_duplicates_var.get(),
                outlier_removal=self.remove_outliers_var.get(),
                thresholds={
                    "Eff_Min": self.eff_min_var.get(),
                    "Voc_Min": self.voc_min_var.get(),
                    "Jsc_Min": self.jsc_min_var.get(),
                    "FF_Min": self.ff_min_var.get(),
                    "FF_Max": self.ff_max_var.get()
                }
            )
            
            theme = self.user_config.get("theme", "dark")
            self.analyzer_instance = IVBatchAnalyzer(self.folder_path.get(), config, theme)
            self.analyzer_instance.set_progress_callback(self.update_progress)
            
            output_dir = self.analyzer_instance.run()
            
            if output_dir:
                self.output_folder = output_dir
                self.root.after(0, self._on_analysis_complete)
                
                # Save config
                self._save_current_config()
            
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            self.root.after(0, lambda: self._show_message("Error", f"Analysis failed: {e}"))
        
        finally:
            self.root.after(0, self._reset_ui_after_analysis)
    
    def _on_analysis_complete(self):
        """Called when analysis completes successfully."""
        self.open_folder_button.configure(state="normal")
        self._show_message("Success", "Analysis complete!")
        
        # Update preview with available plots
        if self.output_folder:
            self.plot_paths = {
                "Boxplot": self.output_folder / "1_Boxplot.png",
                "Histogram": self.output_folder / "1_Histogram.png",
                "Trend": self.output_folder / "2_Trend.png",
                "Yield": self.output_folder / "3_Yield.png",
                "JV Curves": self.output_folder / "4_JV_Curves.png",
                "Voc vs Jsc": self.output_folder / "5_Voc_Jsc_Tradeoff.png",
                "Drivers": self.output_folder / "6_Drivers.png",
                "Resistance": self.output_folder / "7_Resistance.png"
            }
            # Filter to only existing files
            self.plot_paths = {k: v for k, v in self.plot_paths.items() if v.exists()}
            
            if self.plot_paths:
                self.plot_selector.configure(values=list(self.plot_paths.keys()))
                self.plot_selector.set(list(self.plot_paths.keys())[0])
                self.update_preview(list(self.plot_paths.keys())[0])
    
    def _reset_ui_after_analysis(self):
        """Reset UI state after analysis."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def update_progress(self, percent: float, message: str):
        """Update progress bar and label (called from worker thread)."""
        self.root.after(0, lambda: self._update_progress_ui(percent, message))
    
    def _update_progress_ui(self, percent: float, message: str):
        """Update progress UI (must be called on main thread)."""
        self.progress_bar.set(percent / 100.0)
        self.progress_label.configure(text=message)
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        if self.output_folder and self.output_folder.exists():
            if sys.platform == "win32":
                os.startfile(self.output_folder)
            elif sys.platform == "darwin":
                subprocess.run(["open", str(self.output_folder)])
            else:
                subprocess.run(["xdg-open", str(self.output_folder)])
    
    def toggle_theme(self):
        """Toggle between dark and light themes."""
        current = ctk.get_appearance_mode()
        new_theme = "light" if current == "Dark" else "dark"
        ctk.set_appearance_mode(new_theme)
        
        # Update log text background
        bg = "#1e1e1e" if new_theme == "dark" else "white"
        fg = "#e0e0e0" if new_theme == "dark" else "black"
        self.log_text.config(bg=bg, fg=fg)
        
        # Save theme preference
        self.user_config["theme"] = new_theme
        self.config_manager.save_config(self.user_config)
        
        logger.info(f"Theme changed to: {new_theme}")
    
    def update_preview(self, plot_name: str):
        """Update the preview with selected plot."""
        if plot_name not in self.plot_paths:
            return
        
        plot_path = self.plot_paths[plot_name]
        
        # Clear previous canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar:
            self.toolbar.destroy()
        if self.preview_label:
            self.preview_label.destroy()
            self.preview_label = None
        
        # Create new figure
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Load and display image
        from PIL import Image
        img = Image.open(plot_path)
        ax.imshow(img)
        ax.axis('off')
        
        # Embed in canvas
        self.canvas = FigureCanvasTkAgg(fig, master=self.preview_canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.preview_canvas_frame)
        self.toolbar.update()
    
    def clear_log(self):
        """Clear the log text area."""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, "end")
        self.log_text.configure(state='disabled')
    
    def _save_current_config(self):
        """Save current configuration."""
        self.user_config.update({
            "last_folder": self.folder_path.get(),
            "scan_direction": self.scan_dir_var.get(),
            "thresholds": {
                "Eff_Min": self.eff_min_var.get(),
                "Voc_Min": self.voc_min_var.get(),
                "Jsc_Min": self.jsc_min_var.get(),
                "FF_Min": self.ff_min_var.get(),
                "FF_Max": self.ff_max_var.get()
            },
            "remove_duplicates": self.remove_duplicates_var.get(),
            "outlier_removal": self.remove_outliers_var.get()
        })
        self.config_manager.save_config(self.user_config)
    
    def _on_closing(self):
        """Handle window closing."""
        # Save window geometry
        self.user_config["window_geometry"] = self.root.geometry()
        self.config_manager.save_config(self.user_config)
        self.root.destroy()
    
    def _show_message(self, title: str, message: str):
        """Show message dialog (must be called on main thread)."""
        from tkinter import messagebox
        messagebox.showinfo(title, message)
    
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
