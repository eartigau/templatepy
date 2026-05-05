"""
Scattered points dance free,
Gaussian arms embrace the noise—
Outliers fade away.

Gaussian-weighted Savitzky-Golay filter for irregular grids.
Optimized with Numba JIT compilation and parallel processing.

Author: Claude (Anthropic)
License: MIT
"""

import numpy as np
import math
from typing import Optional, Union, Literal

# Numba JIT compilation for performance
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ============================================================================
# Public API
# ============================================================================

def gaussian_savgol(
    x: np.ndarray,
    y: np.ndarray,
    x_out: np.ndarray,
    window_fwhm: float,
    polyorder: int = 3,
    deriv: int = 0,
    sigma_clip: Optional[float] = 5.0,
    clip_iter: int = 3,
    yerr: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Gaussian-weighted Savitzky-Golay filter for irregular grids.
    
    Performs local polynomial regression with Gaussian spatial weights,
    supporting interpolation to arbitrary output grid and optional
    sigma-clipping for outlier rejection.
    
    Parameters
    ----------
    x : array_like
        Input x coordinates (1D, need not be sorted).
    y : array_like
        Input y values. May contain NaN (automatically masked).
    x_out : array_like
        Output x coordinates for interpolation/smoothing.
    window_fwhm : float
        Full Width at Half Maximum of Gaussian weighting function,
        in same units as x. Points within ~3×FWHM contribute.
    polyorder : int, optional
        Polynomial order for local fitting (default: 3).
        Higher orders follow sharper features but amplify noise.
    deriv : int, optional
        Derivative order to compute (default: 0 = smoothing only).
        Use 1 for first derivative, 2 for second, etc.
    sigma_clip : float or None, optional
        Sigma threshold for outlier rejection (default: 5.0).
        Points with |residual| > sigma_clip × MAD are rejected.
        Set to None to disable outlier rejection.
    clip_iter : int, optional
        Number of sigma-clipping iterations (default: 3).
    yerr : array_like or None, optional
        Error bars (1-sigma uncertainties) on y values.
        If provided, weights are multiplied by 1/yerr².
        Points with yerr <= 0 or NaN are masked.
        
    Returns
    -------
    y_out : ndarray
        Filtered/interpolated values at x_out positions.
        NaN where insufficient data or >50% NaN in window.
        
    Examples
    --------
    >>> # Basic smoothing
    >>> x = np.sort(np.random.uniform(0, 10, 1000))
    >>> y = np.sin(x) + 0.1 * np.random.randn(1000)
    >>> x_out = np.linspace(0, 10, 500)
    >>> y_smooth = gaussian_savgol(x, y, x_out, window_fwhm=0.5)
    
    >>> # With error bars
    >>> yerr = 0.1 + 0.05 * np.random.rand(1000)  # heteroscedastic errors
    >>> y_weighted = gaussian_savgol(x, y, x_out, window_fwhm=0.5, yerr=yerr)
    
    >>> # Compute derivative
    >>> dy = gaussian_savgol(x, y, x_out, window_fwhm=0.5, deriv=1)
    
    >>> # Without outlier rejection (faster)
    >>> y_fast = gaussian_savgol(x, y, x_out, window_fwhm=0.5, sigma_clip=None)
    
    Notes
    -----
    - Requires Numba for optimal performance (~100x faster than pure NumPy).
    - Falls back to NumPy implementation if Numba unavailable.
    - Window radius is 1.5 × FWHM, capturing >99% of Gaussian weight.
    - For uniform grids, consider scipy.signal.savgol_filter instead.
    - When yerr is provided, total weight = spatial_weight / yerr².
    
    Performance
    -----------
    With Numba (400k input → 10k output, window ~600 points):
        ~0.5 seconds with sigma-clipping
        ~0.1 seconds without sigma-clipping
    """
    # Input validation and conversion
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_out = np.asarray(x_out, dtype=np.float64)
    
    if x.ndim != 1 or y.ndim != 1 or x_out.ndim != 1:
        raise ValueError("x, y, and x_out must be 1D arrays")
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length ({len(x)} != {len(y)})")
    if window_fwhm <= 0:
        raise ValueError("window_fwhm must be positive")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    if deriv < 0:
        raise ValueError("deriv must be non-negative")
    
    # Handle yerr
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)
        if yerr.ndim != 1 or len(yerr) != len(y):
            raise ValueError(f"yerr must be 1D with same length as y ({len(yerr)} != {len(y)})")
        # Convert to inverse variance weights, mask invalid
        with np.errstate(divide='ignore', invalid='ignore'):
            yerr_weights = np.where((yerr > 0) & np.isfinite(yerr), 1.0 / (yerr * yerr), 0.0)
    else:
        yerr_weights = np.ones(len(y), dtype=np.float64)
    
    # Compute Gaussian sigma and window radius
    sigma_x = window_fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
    radius = 1.5 * window_fwhm  # ~99.7% of Gaussian weight
    
    # Sort input by x for efficient binary search
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx].copy()
    y_sorted = y[sort_idx].copy()
    yerr_w_sorted = yerr_weights[sort_idx].copy()
    
    # Choose implementation
    reject = sigma_clip is not None
    sigma_clip_val = sigma_clip if reject else 0.0
    
    if HAS_NUMBA:
        return _process_numba(
            x_sorted, y_sorted, yerr_w_sorted, x_out, sigma_x, radius,
            polyorder, deriv, reject, sigma_clip_val, clip_iter
        )
    else:
        return _process_numpy(
            x_sorted, y_sorted, yerr_w_sorted, x_out, sigma_x, radius,
            polyorder, deriv, reject, sigma_clip_val, clip_iter
        )


def gaussian_savgol_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_out: np.ndarray,
    window_fwhm: float,
    polyorder: int = 3,
    deriv: int = 0,
    sigma_clip: Optional[float] = 5.0,
    clip_iter: int = 3,
    yerr: Optional[np.ndarray] = None,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply Gaussian-weighted Savitzky-Golay filter along one axis of 2D array.
    
    Parameters
    ----------
    x : array_like
        Input x coordinates (1D).
    y : array_like
        Input y values (2D). Filter applied along specified axis.
    x_out : array_like
        Output x coordinates.
    window_fwhm : float
        FWHM of Gaussian weights.
    polyorder : int, optional
        Polynomial order (default: 3).
    deriv : int, optional
        Derivative order (default: 0).
    sigma_clip : float or None, optional
        Sigma threshold for outlier rejection (default: 5.0).
    clip_iter : int, optional
        Number of clipping iterations (default: 3).
    yerr : array_like or None, optional
        Error bars on y values (same shape as y, or 1D if constant across rows).
    axis : int, optional
        Axis along which to filter (default: -1).
        
    Returns
    -------
    y_out : ndarray
        Filtered array with shape adjusted for x_out length along axis.
    """
    y = np.asarray(y, dtype=np.float64)
    
    if y.ndim == 1:
        return gaussian_savgol(x, y, x_out, window_fwhm, polyorder, 
                               deriv, sigma_clip, clip_iter, yerr)
    
    if yerr is not None:
        yerr = np.asarray(yerr, dtype=np.float64)
        if yerr.ndim == 1:
            # Same errors for all rows
            return np.apply_along_axis(
                lambda row: gaussian_savgol(x, row, x_out, window_fwhm, polyorder,
                                            deriv, sigma_clip, clip_iter, yerr),
                axis, y
            )
        else:
            # Different errors for each row - need to iterate manually
            if axis == -1 or axis == y.ndim - 1:
                result = np.empty((y.shape[0], len(x_out)))
                for i in range(y.shape[0]):
                    result[i] = gaussian_savgol(x, y[i], x_out, window_fwhm, polyorder,
                                                deriv, sigma_clip, clip_iter, yerr[i])
                return result
            else:
                result = np.empty((len(x_out), y.shape[1]))
                for i in range(y.shape[1]):
                    result[:, i] = gaussian_savgol(x, y[:, i], x_out, window_fwhm, polyorder,
                                                   deriv, sigma_clip, clip_iter, yerr[:, i])
                return result
    else:
        return np.apply_along_axis(
            lambda row: gaussian_savgol(x, row, x_out, window_fwhm, polyorder,
                                        deriv, sigma_clip, clip_iter, None),
            axis, y
        )


# ============================================================================
# Numba-optimized implementation
# ============================================================================

if HAS_NUMBA:
    
    @njit(cache=True)
    def _solve_weighted_poly(dx, y, weights, order, dx_scale):
        """
        Solve weighted least squares for polynomial coefficients.
        Direct implementation avoiding numpy.linalg overhead.
        
        dx is normalized by dx_scale to improve numerical conditioning.
        Coefficients are returned in the ORIGINAL (unnormalized) basis.
        """
        n = len(dx)
        m = order + 1
        
        # Build normal equations: (V'WV) c = V'Wy
        # Using normalized dx for numerical stability
        VtWV = np.zeros((m, m))
        VtWy = np.zeros(m)
        
        for i in range(n):
            w = weights[i]
            if w <= 0:
                continue
            
            # Normalized coordinate
            u = dx[i] / dx_scale
            
            # Powers of u
            ui_powers = np.empty(m)
            ui_powers[0] = 1.0
            for j in range(1, m):
                ui_powers[j] = ui_powers[j-1] * u
            
            # Accumulate
            for j in range(m):
                VtWy[j] += w * ui_powers[j] * y[i]
                for k in range(m):
                    VtWV[j, k] += w * ui_powers[j] * ui_powers[k]
        
        # Gaussian elimination with partial pivoting
        aug = np.zeros((m, m + 1))
        for i in range(m):
            for j in range(m):
                aug[i, j] = VtWV[i, j]
            aug[i, m] = VtWy[i]
        
        for i in range(m):
            # Pivot
            max_row = i
            max_val = abs(aug[i, i])
            for k in range(i + 1, m):
                if abs(aug[k, i]) > max_val:
                    max_val = abs(aug[k, i])
                    max_row = k
            
            if max_row != i:
                for j in range(m + 1):
                    aug[i, j], aug[max_row, j] = aug[max_row, j], aug[i, j]
            
            if abs(aug[i, i]) < 1e-14:
                continue
            
            # Eliminate
            for k in range(i + 1, m):
                factor = aug[k, i] / aug[i, i]
                for j in range(i, m + 1):
                    aug[k, j] -= factor * aug[i, j]
        
        # Back substitution (coefficients in normalized basis)
        coeffs_norm = np.zeros(m)
        for i in range(m - 1, -1, -1):
            if abs(aug[i, i]) < 1e-14:
                coeffs_norm[i] = 0.0
            else:
                coeffs_norm[i] = aug[i, m]
                for j in range(i + 1, m):
                    coeffs_norm[i] -= aug[i, j] * coeffs_norm[j]
                coeffs_norm[i] /= aug[i, i]
        
        # Convert back to original basis
        # If p(u) = sum_j c_j u^j where u = x/s
        # Then p(x) = sum_j c_j (x/s)^j = sum_j (c_j / s^j) x^j
        # So coeff_orig[j] = coeff_norm[j] / s^j
        coeffs = np.empty(m)
        scale_power = 1.0
        for j in range(m):
            coeffs[j] = coeffs_norm[j] / scale_power
            scale_power *= dx_scale
        
        return coeffs
    
    
    @njit(cache=True)
    def _weighted_mad(residuals, weights):
        """Compute weighted Median Absolute Deviation."""
        n = len(residuals)
        abs_res = np.abs(residuals)
        
        # Sort by absolute residual
        idx = np.argsort(abs_res)
        sorted_res = abs_res[idx]
        sorted_w = weights[idx]
        
        # Find weighted median
        cumsum = 0.0
        total = 0.0
        for i in range(n):
            total += sorted_w[i]
        
        cutoff = 0.5 * total
        for i in range(n):
            cumsum += sorted_w[i]
            if cumsum >= cutoff:
                return sorted_res[i]
        
        return sorted_res[n - 1]
    
    
    @njit(cache=True)
    def _fit_point(dx, y, spatial_weights, order, deriv, 
                   reject, sigma_clip, n_iter):
        """Fit single output point with optional sigma-clipping."""
        n = len(dx)
        eff_order = min(order, n - 1)
        
        # Scale factor for numerical conditioning
        # Use max absolute dx (window half-width)
        dx_scale = 0.0
        for i in range(n):
            if abs(dx[i]) > dx_scale:
                dx_scale = abs(dx[i])
        if dx_scale < 1e-15:
            dx_scale = 1.0
        
        if not reject or n <= order + 2:
            # Simple weighted fit
            coeffs = _solve_weighted_poly(dx, y, spatial_weights, eff_order, dx_scale)
        else:
            # Iterative sigma-clipping
            weights = spatial_weights.copy()
            
            for iteration in range(n_iter):
                coeffs = _solve_weighted_poly(dx, y, weights, eff_order, dx_scale)
                
                # Compute residuals
                residuals = np.empty(n)
                for i in range(n):
                    pred = 0.0
                    xi = 1.0
                    for j in range(eff_order + 1):
                        pred += coeffs[j] * xi
                        xi *= dx[i]
                    residuals[i] = y[i] - pred
                
                # MAD-based sigma estimate
                sigma = 1.4826 * _weighted_mad(residuals, weights)
                sigma = max(sigma, 1e-12)
                
                # Clip outliers
                n_good = 0
                for i in range(n):
                    if abs(residuals[i]) > sigma_clip * sigma:
                        weights[i] = 0.0
                    else:
                        weights[i] = spatial_weights[i]
                        n_good += 1
                
                if n_good <= order:
                    weights = spatial_weights.copy()
                    break
            
            # Final fit
            coeffs = _solve_weighted_poly(dx, y, weights, eff_order, dx_scale)
        
        # Return derivative coefficient × factorial
        if deriv > eff_order:
            return 0.0
        
        factorial = 1.0
        for i in range(1, deriv + 1):
            factorial *= i
        
        return coeffs[deriv] * factorial
    
    
    @njit(parallel=True, cache=True)
    def _process_numba(x_sorted, y_sorted, yerr_w_sorted, x_out, sigma_x, radius,
                       polyorder, deriv, reject, sigma_clip, n_iter):
        """Process all output points in parallel."""
        n_out = len(x_out)
        n_in = len(x_sorted)
        y_out = np.empty(n_out)
        
        for i in prange(n_out):
            x0 = x_out[i]
            
            # Binary search for window bounds
            left = np.searchsorted(x_sorted, x0 - radius)
            right = np.searchsorted(x_sorted, x0 + radius)
            
            if right <= left:
                y_out[i] = np.nan
                continue
            
            # Count valid (non-NaN) points with positive yerr weight
            n_valid = 0
            n_nan = 0
            for j in range(left, right):
                if np.isnan(y_sorted[j]) or yerr_w_sorted[j] <= 0:
                    n_nan += 1
                else:
                    n_valid += 1
            
            # Check validity
            if n_valid <= polyorder:
                y_out[i] = np.nan
                continue
            
            if n_nan > n_valid:  # >50% NaN
                y_out[i] = np.nan
                continue
            
            # Extract valid points
            dx = np.empty(n_valid)
            y_local = np.empty(n_valid)
            weights = np.empty(n_valid)
            
            k = 0
            for j in range(left, right):
                if not np.isnan(y_sorted[j]) and yerr_w_sorted[j] > 0:
                    d = x_sorted[j] - x0
                    dx[k] = d
                    y_local[k] = y_sorted[j]
                    # Combined weight: spatial gaussian × inverse variance
                    spatial_w = np.exp(-d * d / (2.0 * sigma_x * sigma_x))
                    weights[k] = spatial_w * yerr_w_sorted[j]
                    k += 1
            
            # Fit
            y_out[i] = _fit_point(dx, y_local, weights, polyorder, deriv,
                                  reject, sigma_clip, n_iter)
        
        return y_out


# ============================================================================
# Pure NumPy fallback
# ============================================================================

def _process_numpy(x_sorted, y_sorted, yerr_w_sorted, x_out, sigma_x, radius,
                   polyorder, deriv, reject, sigma_clip, n_iter):
    """Pure NumPy implementation (fallback when Numba unavailable)."""
    n_out = len(x_out)
    y_out = np.empty(n_out)
    
    for i in range(n_out):
        x0 = x_out[i]
        
        left = np.searchsorted(x_sorted, x0 - radius)
        right = np.searchsorted(x_sorted, x0 + radius)
        
        if right <= left:
            y_out[i] = np.nan
            continue
        
        x_local = x_sorted[left:right]
        y_local = y_sorted[left:right]
        yerr_w_local = yerr_w_sorted[left:right]
        
        # Valid = not NaN and positive yerr weight
        valid = ~np.isnan(y_local) & (yerr_w_local > 0)
        n_valid = np.sum(valid)
        n_nan = len(y_local) - n_valid
        
        if n_valid <= polyorder or n_nan > n_valid:
            y_out[i] = np.nan
            continue
        
        dx = x_local[valid] - x0
        y_val = y_local[valid]
        yerr_w = yerr_w_local[valid]
        
        # Combined weights: spatial × inverse variance
        spatial_w = np.exp(-dx**2 / (2 * sigma_x**2))
        weights = spatial_w * yerr_w
        
        y_out[i] = _fit_numpy(dx, y_val, weights, polyorder, deriv,
                              reject, sigma_clip, n_iter)
    
    return y_out


def _fit_numpy(dx, y, spatial_weights, polyorder, deriv, reject, sigma_clip, n_iter):
    """NumPy single-point fit with optional sigma-clipping."""
    n = len(dx)
    order = min(polyorder, n - 1)
    
    # Scale dx for numerical conditioning
    dx_scale = np.max(np.abs(dx))
    if dx_scale < 1e-15:
        dx_scale = 1.0
    dx_norm = dx / dx_scale
    
    V = np.column_stack([dx_norm**j for j in range(order + 1)])
    weights = spatial_weights.copy()
    
    if reject and n > polyorder + 2:
        for _ in range(n_iter):
            W = np.diag(weights)
            VtW = V.T @ W
            try:
                coeffs_norm = np.linalg.solve(VtW @ V, VtW @ y)
            except np.linalg.LinAlgError:
                coeffs_norm = np.linalg.lstsq(V * weights[:, None], y * weights, rcond=None)[0]
            
            # Convert to original basis for residual computation
            coeffs = coeffs_norm / (dx_scale ** np.arange(order + 1))
            
            residuals = y - np.column_stack([dx**j for j in range(order + 1)]) @ coeffs
            abs_res = np.abs(residuals)
            
            # Weighted MAD
            idx = np.argsort(abs_res)
            cumsum = np.cumsum(weights[idx])
            med_idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
            sigma = 1.4826 * abs_res[idx[min(med_idx, len(abs_res)-1)]]
            sigma = max(sigma, 1e-12)
            
            outliers = abs_res > sigma_clip * sigma
            weights = np.where(outliers, 0.0, spatial_weights)
            
            if np.sum(weights > 0) <= polyorder:
                weights = spatial_weights.copy()
                break
    
    W = np.diag(weights)
    VtW = V.T @ W
    try:
        coeffs_norm = np.linalg.solve(VtW @ V, VtW @ y)
    except np.linalg.LinAlgError:
        coeffs_norm = np.linalg.lstsq(V * weights[:, None], y * weights, rcond=None)[0]
    
    # Convert back to original basis
    coeffs = coeffs_norm / (dx_scale ** np.arange(order + 1))
    
    if deriv > order:
        return 0.0
    
    return coeffs[deriv] * math.factorial(deriv)


# ============================================================================
# Module info
# ============================================================================

__version__ = "1.0.0"
__all__ = ["gaussian_savgol", "gaussian_savgol_2d", "HAS_NUMBA"]


if __name__ == "__main__":
    # Quick test
    print(f"gaussian_savgol v{__version__}")
    print(f"Numba acceleration: {'enabled' if HAS_NUMBA else 'disabled'}")
    
    # Demo
    np.random.seed(42)
    n = 10000
    x = np.sort(np.random.uniform(0, 10, n))
    y_true = np.sin(x) + 0.3 * np.sin(5 * x)
    
    # Heteroscedastic noise: some points have larger errors
    yerr = 0.05 + 0.15 * np.random.rand(n)  # errors between 0.05 and 0.2
    y = y_true + yerr * np.random.randn(n)
    
    # Add outliers
    outliers = np.random.choice(n, n // 1000, replace=False)
    y[outliers] += np.random.choice([-1, 1], len(outliers)) * np.random.uniform(1, 3, len(outliers))
    
    x_out = np.linspace(0, 10, 1000)
    y_true_out = np.sin(x_out) + 0.3 * np.sin(5 * x_out)
    
    import time
    
    # Without yerr
    t0 = time.perf_counter()
    y_no_yerr = gaussian_savgol(x, y, x_out, window_fwhm=0.3, sigma_clip=5.0)
    t1 = time.perf_counter()
    
    # With yerr
    t2 = time.perf_counter()
    y_with_yerr = gaussian_savgol(x, y, x_out, window_fwhm=0.3, sigma_clip=5.0, yerr=yerr)
    t3 = time.perf_counter()
    
    err_no_yerr = np.sqrt(np.nanmean((y_no_yerr - y_true_out)**2))
    err_with_yerr = np.sqrt(np.nanmean((y_with_yerr - y_true_out)**2))
    
    print(f"\nTest: {n:,} -> {len(x_out):,} points, {len(outliers)} outliers")
    print(f"  Heteroscedastic errors: yerr in [{yerr.min():.2f}, {yerr.max():.2f}]")
    print(f"\n  Without yerr: {(t1-t0)*1000:.1f} ms, RMS = {err_no_yerr:.4f}")
    print(f"  With yerr:    {(t3-t2)*1000:.1f} ms, RMS = {err_with_yerr:.4f}")
    print(f"  Improvement:  {100*(err_no_yerr - err_with_yerr)/err_no_yerr:.1f}%")
