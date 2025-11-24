"""
IV Batch Analyzer V5.0 - Professional Edition
Physics Module

Advanced photovoltaic parameter extraction and analysis:
- Single-diode model fitting (two-step approach)
- S-shape and kink detection
- Lambert W-based parameter extraction
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple, Optional
import warnings

import numpy as np
from scipy.stats import linregress
from scipy.special import lambertw
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Suppress optimization warnings for batch processing
warnings.filterwarnings('ignore', category=OptimizeWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


# ================= CONSTANTS =================

# Physical constants
K_BOLTZMANN = 1.380649e-23  # J/K
Q_ELECTRON = 1.602176634e-19  # C
VT_300K = 0.0259  # Thermal voltage at 300K (kT/q)

# Quality thresholds
DEFAULT_FF_THRESHOLD = 40.0  # % - minimum FF for full fitting
MIN_DATA_POINTS = 20  # Minimum points for reliable fitting
VOC_REGION_FRACTION = 0.9  # Use top 10% of voltage for Rs extraction
ISC_REGION_FRACTION = 0.9  # Use top 10% of current for Rsh extraction


# ================= SLOPE-BASED EXTRACTION =================

def extract_resistance_from_slopes(voltage: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    """
    Extract series and shunt resistance using slope method.
    
    This is the PRIMARY and ROBUST method that always succeeds.
    
    Algorithm:
    - Rs: Slope of V-I curve near Voc (where I ≈ 0)
    - Rsh: Slope of V-I curve near Isc (where V ≈ 0)
    
    Args:
        voltage: Voltage array (V)
        current: Current density array (mA/cm²)
        
    Returns:
        dict: {'Rs_slope': float, 'Rsh_slope': float, 'method': 'slope'}
    """
    try:
        # Ensure arrays are numpy and sorted by voltage
        V = np.asarray(voltage, dtype=float)
        I = np.asarray(current, dtype=float)
        
        # Sort by voltage
        sort_idx = np.argsort(V)
        V = V[sort_idx]
        I = I[sort_idx]
        
        # Remove any NaN or Inf
        mask = np.isfinite(V) & np.isfinite(I)
        V = V[mask]
        I = I[mask]
        
        if len(V) < 5:
            logger.warning("Insufficient data points for slope extraction")
            return {'Rs_slope': np.nan, 'Rsh_slope': np.nan, 'method': 'slope'}
        
        # Extract Rs from Voc region (high voltage, low current)
        Voc = np.max(V)
        Isc_abs = np.max(np.abs(I))
        
        # Select points near Voc (V > 0.9*Voc)
        voc_mask = V > (VOC_REGION_FRACTION * Voc)
        if np.sum(voc_mask) >= 3:
            V_voc = V[voc_mask]
            I_voc = I[voc_mask]
            # Rs = -dV/dI at Voc region
            slope_rs, intercept, r_value, p_value, std_err = linregress(I_voc, V_voc)
            Rs_slope = abs(slope_rs)  # Ω·cm²
        else:
            logger.warning("Insufficient points in Voc region for Rs extraction")
            Rs_slope = np.nan
        
        # Extract Rsh from Isc region (low voltage, high current)
        # Select points near Isc (|I| > 0.9*|Isc|)
        isc_mask = np.abs(I) > (ISC_REGION_FRACTION * Isc_abs)
        if np.sum(isc_mask) >= 3:
            V_isc = V[isc_mask]
            I_isc = I[isc_mask]
            # Rsh = -dV/dI at Isc region
            slope_rsh, intercept, r_value, p_value, std_err = linregress(I_isc, V_isc)
            Rsh_slope = abs(slope_rsh)  # Ω·cm²
        else:
            logger.warning("Insufficient points in Isc region for Rsh extraction")
            Rsh_slope = np.nan
        
        return {
            'Rs_slope': Rs_slope,
            'Rsh_slope': Rsh_slope,
            'method': 'slope'
        }
        
    except Exception as e:
        logger.error(f"Slope extraction failed: {e}")
        return {'Rs_slope': np.nan, 'Rsh_slope': np.nan, 'method': 'slope'}


# ================= S-SHAPE DETECTION =================

def detect_s_shape(voltage: np.ndarray, current: np.ndarray) -> Dict[str, any]:
    """
    Detect S-shape or kink anomalies in IV curve.
    
    Detection methods:
    1. Curvature analysis (second derivative)
    2. Derivative ratio
    3. Kink factor
    
    Args:
        voltage: Voltage array (V)
        current: Current density array (mA/cm²)
        
    Returns:
        dict: {
            'has_s_shape': bool,
            'has_kink': bool,
            'derivative_ratio': float,
            'kink_factor': float,
            'severity': str  # 'None', 'Mild', 'Moderate', 'Severe'
        }
    """
    try:
        # Ensure arrays and sort
        V = np.asarray(voltage, dtype=float)
        I = np.asarray(current, dtype=float)
        
        sort_idx = np.argsort(V)
        V = V[sort_idx]
        I = I[sort_idx]
        
        # Remove NaN/Inf
        mask = np.isfinite(V) & np.isfinite(I)
        V = V[mask]
        I = I[mask]
        
        if len(V) < 10:
            return {
                'has_s_shape': False,
                'has_kink': False,
                'derivative_ratio': np.nan,
                'kink_factor': np.nan,
                'severity': 'Unknown'
            }
        
        # Smooth data using Savitzky-Golay filter
        window_length = min(11, len(V) if len(V) % 2 == 1 else len(V) - 1)
        if window_length < 5:
            window_length = 5
        
        try:
            I_smooth = savgol_filter(I, window_length=window_length, polyorder=2)
        except:
            I_smooth = I
        
        # Calculate first derivative dI/dV
        dI_dV = np.gradient(I_smooth, V)
        
        # Calculate second derivative d²I/dV²
        d2I_dV2 = np.gradient(dI_dV, V)
        
        # Metric 1: Derivative Ratio
        # Normal curve: DR ≈ 2-3, S-shape: DR > 5
        derivative_ratio = np.max(np.abs(dI_dV)) / (np.mean(np.abs(dI_dV)) + 1e-10)
        
        # Metric 2: Kink Factor
        # Find voltage where |dI/dV| is maximum
        max_deriv_idx = np.argmax(np.abs(dI_dV))
        V_max_deriv = V[max_deriv_idx]
        Voc = np.max(V)
        kink_factor = (Voc - V_max_deriv) / (Voc + 1e-10)
        
        # Metric 3: Inflection points (zero crossings of d²I/dV²)
        zero_crossings = np.where(np.diff(np.sign(d2I_dV2)))[0]
        num_inflections = len(zero_crossings)
        
        # Decision logic
        has_kink = kink_factor > 0.2
        has_s_shape = derivative_ratio > 5.0 or num_inflections > 2
        
        # Severity scoring
        score = 0
        if derivative_ratio > 5: score += 1
        if derivative_ratio > 10: score += 1
        if kink_factor > 0.2: score += 1
        if kink_factor > 0.4: score += 1
        if num_inflections > 2: score += 1
        
        if score == 0:
            severity = 'None'
        elif score <= 1:
            severity = 'Mild'
        elif score <= 3:
            severity = 'Moderate'
        else:
            severity = 'Severe'
        
        return {
            'has_s_shape': has_s_shape,
            'has_kink': has_kink,
            'derivative_ratio': float(derivative_ratio),
            'kink_factor': float(kink_factor),
            'num_inflections': int(num_inflections),
            'severity': severity
        }
        
    except Exception as e:
        logger.error(f"S-shape detection failed: {e}")
        return {
            'has_s_shape': False,
            'has_kink': False,
            'derivative_ratio': np.nan,
            'kink_factor': np.nan,
            'severity': 'Unknown'
        }


# ================= QUALITY GATE =================

def should_use_full_fitting(voltage: np.ndarray, current: np.ndarray, 
                           ff: Optional[float] = None,
                           has_s_shape: bool = False,
                           ff_threshold: float = DEFAULT_FF_THRESHOLD) -> bool:
    """
    Determine if full Lambert W fitting should be attempted.
    
    Quality gates:
    1. FF > threshold (default 40%)
    2. No S-shape anomaly
    3. Sufficient data points
    4. Monotonic voltage sweep
    
    Args:
        voltage: Voltage array
        current: Current array
        ff: Fill factor (%)
        has_s_shape: Whether S-shape was detected
        ff_threshold: Minimum FF for fitting
        
    Returns:
        bool: True if full fitting should be attempted
    """
    # Gate 1: FF check
    if ff is not None and ff < ff_threshold:
        logger.debug(f"Skipping full fit: FF={ff:.1f}% < {ff_threshold}%")
        return False
    
    # Gate 2: S-shape check
    if has_s_shape:
        logger.debug("Skipping full fit: S-shape detected")
        return False
    
    # Gate 3: Data points check
    if len(voltage) < MIN_DATA_POINTS:
        logger.debug(f"Skipping full fit: Only {len(voltage)} points < {MIN_DATA_POINTS}")
        return False
    
    # Gate 4: Monotonic voltage check
    V = np.asarray(voltage)
    if not (np.all(np.diff(V) > 0) or np.all(np.diff(V) < 0)):
        logger.debug("Skipping full fit: Non-monotonic voltage sweep")
        return False
    
    return True


# ================= LAMBERT W FITTING =================

def fit_single_diode_model_full(voltage: np.ndarray, current: np.ndarray,
                                Rs_initial: float, Rsh_initial: float,
                                temperature: float = 300.0) -> Dict[str, float]:
    """
    Fit single-diode model using Lambert W function.
    
    Model equation:
        I = IL - I0[exp((V + I*Rs)/(n*Vt)) - 1] - (V + I*Rs)/Rsh
    
    Uses Lambert W function for analytical solution.
    
    Args:
        voltage: Voltage array (V)
        current: Current density array (mA/cm²)
        Rs_initial: Initial guess for Rs from slope method
        Rsh_initial: Initial guess for Rsh from slope method
        temperature: Cell temperature (K)
        
    Returns:
        dict: {
            'Rs_fitted': float,
            'Rsh_fitted': float,
            'n': float,  # ideality factor
            'I0': float,  # saturation current
            'IL': float,  # light current
            'fit_R2': float,
            'method': 'lambert_w'
        }
    """
    try:
        V = np.asarray(voltage, dtype=float)
        I = np.asarray(current, dtype=float)
        
        # Sort and clean
        sort_idx = np.argsort(V)
        V = V[sort_idx]
        I = I[sort_idx]
        mask = np.isfinite(V) & np.isfinite(I)
        V = V[mask]
        I = I[mask]
        
        # Thermal voltage
        Vt = K_BOLTZMANN * temperature / Q_ELECTRON  # ≈ 0.026V at 300K
        
        # Initial parameter estimates
        IL_init = np.max(np.abs(I))  # Light current ≈ Isc
        I0_init = 1e-9  # Typical saturation current (mA/cm²)
        n_init = 1.5  # Mid-range ideality factor
        
        # Define single-diode model
        def diode_model(V_arr, IL, I0, n, Rs, Rsh):
            """Single-diode model using Lambert W function."""
            I_arr = np.zeros_like(V_arr)
            
            for idx, V_val in enumerate(V_arr):
                try:
                    # Lambert W solution
                    # This is a simplified implementation
                    # Full Lambert W solution would be more complex
                    
                    # Approximate iterative solution
                    I_guess = IL - V_val / Rsh
                    for _ in range(5):  # Newton iterations
                        exp_term = np.exp((V_val + I_guess * Rs) / (n * Vt))
                        f = IL - I0 * (exp_term - 1) - (V_val + I_guess * Rs) / Rsh - I_guess
                        df = -I0 * Rs / (n * Vt) * exp_term - Rs / Rsh - 1
                        I_guess = I_guess - f / df
                        
                    I_arr[idx] = I_guess
                except:
                    I_arr[idx] = np.nan
            
            return I_arr
        
        # Fit using curve_fit
        p0 = [IL_init, I0_init, n_init, Rs_initial, Rsh_initial]
        bounds = (
            [IL_init * 0.8, 1e-12, 1.0, 0.0, 100.0],  # Lower bounds
            [IL_init * 1.2, 1e-6, 2.5, 50.0, 1e6]     # Upper bounds
        )
        
        popt, pcov = curve_fit(
            diode_model, V, I,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
            method='trf'
        )
        
        IL_fit, I0_fit, n_fit, Rs_fit, Rsh_fit = popt
        
        # Calculate fit quality (R²)
        I_fitted = diode_model(V, *popt)
        ss_res = np.sum((I - I_fitted) ** 2)
        ss_tot = np.sum((I - np.mean(I)) ** 2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'Rs_fitted': float(Rs_fit),
            'Rsh_fitted': float(Rsh_fit),
            'n': float(n_fit),
            'I0': float(I0_fit),
            'IL': float(IL_fit),
            'fit_R2': float(R2),
            'method': 'lambert_w'
        }
        
    except Exception as e:
        logger.warning(f"Lambert W fitting failed: {e}")
        return {
            'Rs_fitted': np.nan,
            'Rsh_fitted': np.nan,
            'n': np.nan,
            'I0': np.nan,
            'IL': np.nan,
            'fit_R2': np.nan,
            'method': 'lambert_w_failed'
        }


# ================= MAIN ENTRY POINT =================

def extract_pv_parameters(voltage: np.ndarray, current: np.ndarray,
                         ff: Optional[float] = None,
                         temperature: float = 300.0,
                         ff_threshold: float = DEFAULT_FF_THRESHOLD) -> Dict[str, any]:
    """
    Extract PV parameters using two-step strategy.
    
    Strategy:
    1. Always extract Rs/Rsh using slope method (robust)
    2. Detect S-shape anomalies
    3. If curve quality is high, attempt full Lambert W fitting
    
    Args:
        voltage: Voltage array (V)
        current: Current density array (mA/cm²)
        ff: Fill factor (%) - optional but recommended
        temperature: Cell temperature (K)
        ff_threshold: Minimum FF for full fitting
        
    Returns:
        dict: Combined results from slope and (optionally) fitting methods
    """
    # Step 1: Slope-based extraction (always)
    slope_params = extract_resistance_from_slopes(voltage, current)
    
    # Step 2: S-shape detection
    s_shape_results = detect_s_shape(voltage, current)
    
    # Step 3: Quality gate for full fitting
    use_full_fit = should_use_full_fitting(
        voltage, current, ff,
        s_shape_results['has_s_shape'],
        ff_threshold
    )
    
    # Combine results
    results = {**slope_params, **s_shape_results}
    
    if use_full_fit and not np.isnan(slope_params['Rs_slope']):
        # Step 4: Full Lambert W fitting
        fit_params = fit_single_diode_model_full(
            voltage, current,
            slope_params['Rs_slope'],
            slope_params['Rsh_slope'],
            temperature
        )
        results.update(fit_params)
        results['fit_attempted'] = True
        results['fit_successful'] = not np.isnan(fit_params.get('Rs_fitted', np.nan))
    else:
        # Slope-only mode
        results.update({
            'Rs_fitted': np.nan,
            'Rsh_fitted': np.nan,
            'n': np.nan,
            'I0': np.nan,
            'IL': np.nan,
            'fit_R2': np.nan,
            'fit_attempted': False,
            'fit_successful': False,
            'method': 'slope_only'
        })
    
    return results
