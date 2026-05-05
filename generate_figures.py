"""
generate_figures.py
-------------------
Regenerate the diagnostic figures used in the templatepy documentation.
Run after building a template to produce PNG files in figures/.

Usage:
    python generate_figures.py <template_csv> [--output_dir figures]

The script loads a pre-built template CSV and generates:
  fig_spectrum.png             — full template spectrum with odd/even orders
  fig_order_zoom.png           — zoom on a single échelle order + derivative
  fig_derivatives.png          — d0, d1, d2 spectral derivatives
  fig_odd_even_residuals.png   — odd vs even order residuals
  fig_snr.png                  — per-pixel S/N proxy (flux / eflux)

Can also be called programmatically:
    from generate_figures import run_all_figures
    run_all_figures(tbl, fig_dir='figures')
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from astropy.table import Table

# ── Style ───────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
DARK_BG   = "#0d1117"
SURFACE   = "#161b22"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
MUTED     = "#8b949e"
TEXT      = "#e6edf3"

RC = {
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  TEXT,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
    "text.color":       TEXT,
    "grid.color":       "#21262d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
}
plt.rcParams.update(RC)

# ── Helpers ──────────────────────────────────────────────────────────────────
def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {path}")


def wavelength_range(tbl, w0, w1):
    """Return boolean mask for wavelength range [w0, w1] nm."""
    return (tbl["wavelength"] >= w0) & (tbl["wavelength"] <= w1)


def auto_ylim(*arrays, margin=0.08):
    """Return (ylo, yhi) that tightly bracket all finite values with strict min/max."""
    combined = np.concatenate([np.asarray(a).ravel() for a in arrays])
    finite = combined[np.isfinite(combined)]
    if len(finite) == 0:
        return 0, 1
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    span = hi - lo if hi != lo else 1.0
    return lo - margin * span, hi + margin * span


# ── Figure 1 — Full spectrum overview ────────────────────────────────────────
def fig_spectrum_overview(tbl, out_dir, zoom_w0=1545, zoom_w1=1545.1):
    wl   = np.asarray(tbl["wavelength"])
    flux = np.asarray(tbl["flux"])
    eflux = np.asarray(tbl["eflux"])
    odd  = np.asarray(tbl["flux_odd"])
    even = np.asarray(tbl["flux_even"])

    # Zoom to requested domain
    mask = (wl >= zoom_w0) & (wl <= zoom_w1)
    wl_z    = wl[mask]
    flux_z  = flux[mask]
    eflux_z = eflux[mask]
    odd_z   = odd[mask]
    even_z  = even[mask]

    # Normalise to median flux in the domain
    med = np.nanmedian(flux_z)
    if med == 0 or not np.isfinite(med):
        med = 1.0
    flux_z  = flux_z  / med
    eflux_z = eflux_z / med
    odd_z   = odd_z   / med
    even_z  = even_z  / med

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                              gridspec_kw={"hspace": 0.05})

    # Panel 1 — Template + uncertainty
    axes[0].fill_between(wl_z, flux_z - eflux_z, flux_z + eflux_z,
                         color=ACCENT, alpha=0.2, label="1-σ uncertainty")
    axes[0].plot(wl_z, flux_z, lw=0.8, color=ACCENT, label="Template (combined)")
    axes[0].set_ylabel("Normalised flux")
    axes[0].set_ylim(*auto_ylim(flux_z))
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title(f"Barycentric template spectrum — {zoom_w0}–{zoom_w1} nm")
    axes[0].grid(True)

    # Panel 2 — Odd vs even
    axes[1].plot(wl_z, odd_z,  lw=0.8, color=GREEN,  alpha=0.8, label="Odd orders")
    axes[1].plot(wl_z, even_z, lw=0.8, color=YELLOW, alpha=0.8, label="Even orders")
    axes[1].set_ylabel("Normalised flux")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylim(*auto_ylim(odd_z, even_z))
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True)

    savefig(fig, os.path.join(out_dir, "fig_spectrum.png"))


# ── Figure 2 — Order zoom ────────────────────────────────────────────────────
def fig_order_zoom(tbl, out_dir, w0=1545, w1=1545.1):
    mask = wavelength_range(tbl, w0, w1)
    if mask.sum() < 10:
        print(f"  Skipping order zoom: no data between {w0}–{w1} nm")
        return

    wl   = np.asarray(tbl["wavelength"])[mask]
    flux = np.asarray(tbl["flux"])[mask]
    ef   = np.asarray(tbl["eflux"])[mask]

    # Normalise to median in the zoom window
    med = np.nanmedian(flux)
    if med == 0 or not np.isfinite(med):
        med = 1.0
    flux = flux / med
    ef   = ef   / med

    d1   = np.asarray(tbl["flux_odd_savgol_d1"])[mask] if "flux_odd_savgol_d1" in tbl.colnames else None

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True,
                              gridspec_kw={"hspace": 0.05, "height_ratios": [3, 1]})

    axes[0].fill_between(wl, flux - ef, flux + ef, color=ACCENT, alpha=0.2)
    axes[0].plot(wl, flux, lw=1.2, color=ACCENT, label="Template")
    axes[0].set_ylabel("Normalised flux")
    axes[0].set_ylim(*auto_ylim(flux))
    axes[0].set_title(f"Template zoom  {w0}–{w1} nm")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    if d1 is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            axes[1].plot(wl, d1, lw=1.0, color=YELLOW)
        axes[1].axhline(0, color=MUTED, lw=0.8)
        axes[1].set_ylabel("∂f/∂lnλ")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].grid(True)

    savefig(fig, os.path.join(out_dir, "fig_order_zoom.png"))


# ── Figure 3 — Spectral derivatives ─────────────────────────────────────────
def fig_derivatives(tbl, out_dir, w0=1600, w1=1650):
    # prefer a region with strong stellar lines
    for band in [(1600, 1650), (1250, 1290), (1080, 1130)]:
        mask = wavelength_range(tbl, *band)
        if mask.sum() > 50:
            w0, w1 = band
            break

    mask = wavelength_range(tbl, w0, w1)
    wl = tbl["wavelength"][mask]

    deriv_cols = []
    for suffix in ("", "_odd", "_even"):
        for d in range(4):
            col = f"flux{suffix}_savgol_d{d}"
            if col in tbl.colnames:
                deriv_cols.append((col, d, suffix))

    # Only plot d0, d1, d2 for combined
    plot_triplets = [
        ("flux_savgol_d0", "d0 — template",        ACCENT),
        ("flux_savgol_d1", "d1 — ∂f/∂lnλ",         GREEN),
        ("flux_savgol_d2", "d2 — ∂²f/∂ln²λ",       YELLOW),
    ]
    plot_triplets = [(c, l, clr) for c, l, clr in plot_triplets if c in tbl.colnames]
    if not plot_triplets:
        print("  Skipping derivatives figure: columns not found")
        return

    fig, axes = plt.subplots(len(plot_triplets), 1, figsize=(12, 2.5 * len(plot_triplets)),
                              sharex=True, gridspec_kw={"hspace": 0.08})
    if len(plot_triplets) == 1:
        axes = [axes]

    for ax, (col, label, clr) in zip(axes, plot_triplets):
        data = tbl[col][mask]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.plot(wl, data, lw=0.9, color=clr)
        ax.axhline(0, color=MUTED, lw=0.7, ls="--")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True)

    axes[-1].set_xlabel("Wavelength (nm)")
    axes[0].set_title(f"Spectral derivatives  {w0}–{w1} nm")
    savefig(fig, os.path.join(out_dir, "fig_derivatives.png"))


# ── Figure 4 — Odd/even residuals ────────────────────────────────────────────
def fig_odd_even_residuals(tbl, out_dir):
    wl   = tbl["wavelength"]
    odd  = tbl["flux_odd"]
    even = tbl["flux_even"]
    flux = tbl["flux"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_odd  = odd  - flux
        res_even = even - flux

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"hspace": 0.05})

    axes[0].plot(wl, odd,  lw=0.5, color=GREEN,  alpha=0.8, label="Odd orders")
    axes[0].plot(wl, even, lw=0.5, color=YELLOW, alpha=0.8, label="Even orders")
    axes[0].plot(wl, flux, lw=0.8, color=ACCENT, label="Combined template")
    axes[0].set_ylabel("Flux (norm.)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True)
    axes[0].set_title("Odd vs Even order comparison")

    axes[1].plot(wl, res_odd,  lw=0.5, color=GREEN,  alpha=0.8, label="Odd − template")
    axes[1].plot(wl, res_even, lw=0.5, color=YELLOW, alpha=0.8, label="Even − template")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    savefig(fig, os.path.join(out_dir, "fig_odd_even.png"))


# ── Figure 5 — Per-pixel S/N ─────────────────────────────────────────────────
def fig_snr(tbl, out_dir):
    wl   = tbl["wavelength"]
    flux = tbl["flux"]
    ef   = tbl["eflux"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snr = np.abs(flux) / ef

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(wl, snr, lw=0.5, color=ACCENT, alpha=0.8)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("S/N  (flux / eflux)")
    ax.set_ylim(bottom=0)
    ax.set_title("Per-pixel signal-to-noise ratio")
    ax.grid(True)

    # Mark O2 bands
    for band, label in [((759, 771), "O₂ A"), ((686, 696), "O₂ B")]:
        if band[0] > wl.min() and band[1] < wl.max():
            ax.axvspan(*band, color=RED, alpha=0.15)
            ax.text(np.mean(band), ax.get_ylim()[1] * 0.9, label,
                    ha="center", color=RED, fontsize=8)

    savefig(fig, os.path.join(out_dir, "fig_snr.png"))


# ── Public API ───────────────────────────────────────────────────────────────
def run_all_figures(tbl, out_dir='figures', zoom_w0=1260, zoom_w1=1295):
    """Generate all documentation figures from a templatepy output table.

    Parameters
    ----------
    tbl : astropy.table.Table or path-like
        Template table (or path to CSV/FITS file).
    out_dir : str
        Directory where PNG figures are written.
    zoom_w0, zoom_w1 : float
        Wavelength range (nm) for the order-zoom panel.
    """
    from astropy.table import Table as _Table
    if not hasattr(tbl, 'colnames'):
        tbl = _Table.read(str(tbl))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating documentation figures in {out_dir}/")
    fig_spectrum_overview(tbl, out_dir)
    fig_order_zoom(tbl, out_dir, zoom_w0, zoom_w1)
    fig_derivatives(tbl, out_dir)
    fig_odd_even_residuals(tbl, out_dir)
    fig_snr(tbl, out_dir)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate documentation figures from a templatepy output CSV."
    )
    parser.add_argument("template_csv", help="Path to the template CSV/FITS file.")
    parser.add_argument("--output_dir", default="figures",
                        help="Directory where PNG figures are written (default: figures).")
    parser.add_argument("--zoom_w0", type=float, default=1260,
                        help="Start wavelength (nm) for the order-zoom panel.")
    parser.add_argument("--zoom_w1", type=float, default=1295,
                        help="End wavelength (nm) for the order-zoom panel.")
    args = parser.parse_args()

    print(f"Loading template: {args.template_csv}")
    tbl = Table.read(args.template_csv)
    print(f"  {len(tbl)} rows, columns: {tbl.colnames}")

    print("Generating figures …")
    run_all_figures(tbl, args.output_dir, zoom_w0=args.zoom_w0, zoom_w1=args.zoom_w1)
    print(f"\nAll figures written to {args.output_dir}/")


if __name__ == "__main__":
    main()
