import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np


def add_at_risk_counts(fitters, ax=None, y_shift=-0.25, colors=None, labels=None,
                       fontsize=10, show_censored_counts=False, bold=True,
                       show_title=False, label_pad=-0.10):
    """
    Add a table of at-risk counts below the plot.
    Re-implemented using ax.text for perfect alignment with X-axis ticks.

    Parameters
    ----------
    fitters : list
        List of fitted KaplanMeierFitter (or AalenJohansenFitter) objects.
    ax : matplotlib Axes
        Target axes.
    y_shift : float
        Vertical offset for the table (axes fraction).
    colors : list of str
        Colors for each fitter row.
    labels : list of str
        Custom labels for each fitter.
    fontsize : int
        Font size for table text.
    show_censored_counts : bool
        If True, display format is "n_at_risk (n_censored)" matching
        JCO/NEJM publication style.
    bold : bool
        Whether table text uses bold weight.
    show_title : bool
        If True, adds a title row above the table (e.g. "No. at risk"
        or "No. at risk (censored)").
    label_pad : float
        X position for row labels in axes fraction. More negative = more
        space between labels and the first data column. Default -0.12.
    """
    if ax is None:
        ax = plt.gca()

    # Get ticks from the plot
    ticks = ax.get_xticks()
    # Filter ticks that make sense AND are within the current view limits
    view_min, view_max = ax.get_xlim()
    valid_ticks = [t for t in ticks if view_min <= t <= view_max]

    # Configuration for layout
    row_height = 0.05
    start_y = y_shift
    font_weight = 'bold' if bold else 'normal'

    # Blended transform: X is data coords (matches ticks), Y is axes coords
    trans_data_axes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Optional title row
    if show_title:
        title_text = "No. at risk (censored)" if show_censored_counts else "No. at risk"
        ax.text(label_pad, start_y, title_text, transform=ax.transAxes,
                ha='right', va='center', weight='bold', color='black',
                fontsize=fontsize, style='italic')
        start_y -= row_height

    for i, fitter in enumerate(fitters):
        y_pos = start_y - (i * row_height)

        # 1. Plot Row Label (Left of Y-axis)
        lbl = labels[i] if labels and i < len(labels) else fitter._label

        # Color resolution
        color = 'black'
        if colors and i < len(colors):
            color = colors[i]

        ax.text(label_pad, y_pos, lbl, transform=ax.transAxes,
                ha='right', va='center', weight=font_weight, color=color, fontsize=fontsize)

        # 2. Plot Counts at each tick
        for t in valid_ticks:
            # Calculate at-risk count
            if t in fitter.event_table.index:
                at_risk = fitter.event_table.loc[t, 'at_risk']
            else:
                sliced = fitter.event_table.loc[:t]
                if sliced.empty:
                    at_risk = fitter.event_table['at_risk'].iloc[0]
                else:
                    last_row = sliced.iloc[-1]
                    at_risk = last_row['at_risk'] - last_row['removed']

            if isinstance(at_risk, (pd.Series, np.ndarray, list)):
                try:
                    at_risk = at_risk.item()
                except:
                    pass
            at_risk = int(at_risk)

            # Calculate cumulative censored count up to this time
            if show_censored_counts:
                sliced_cens = fitter.event_table.loc[:t]
                if not sliced_cens.empty and 'censored' in sliced_cens.columns:
                    cum_censored = int(sliced_cens['censored'].sum())
                else:
                    cum_censored = 0
                display_text = f"{at_risk} ({cum_censored})"
            else:
                display_text = str(at_risk)

            # Plot the number
            ax.text(t, y_pos, display_text, transform=trans_data_axes,
                    ha='center', va='center', color=color, fontsize=fontsize,
                    weight=font_weight)


def add_survival_annotations(fitters, ax, colors=None, labels=None,
                             show_median=False, show_x_year=False, x_year_time=None,
                             line_style='--', line_alpha=0.5, is_cif=False):
    """
    Add auto-computed survival milestone annotations to a KM or CIF plot.
    Draws clean dashed drop-lines only (no text labels on axes).

    Parameters
    ----------
    fitters : list
        Fitted KaplanMeierFitter or AalenJohansenFitter objects.
    ax : matplotlib Axes
    colors : list of str
    labels : list of str
    show_median : bool
        Draw dashed lines at median survival (KM) or median CIF time.
    show_x_year : bool
        Draw dashed lines at a specific timepoint.
    x_year_time : float
        The timepoint for X-year survival annotation.
    line_style : str
    line_alpha : float
    is_cif : bool
        If True, treats fitters as CIF (cumulative incidence) curves.
    """
    if not fitters:
        return

    view_min, view_max = ax.get_xlim()

    # --- Median survival lines ---
    if show_median:
        # Draw horizontal reference at 0.5 (KM) or find median CIF
        if not is_cif:
            ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

        for i, fitter in enumerate(fitters):
            color = colors[i] if colors and i < len(colors) else f'C{i}'
            lbl = labels[i] if labels and i < len(labels) else fitter._label

            if is_cif:
                # For CIF: median is when cumulative incidence first exceeds 0.5
                cdf = fitter.cumulative_density_
                col = cdf.columns[0]
                crossed = cdf[cdf[col] >= 0.5]
                if crossed.empty:
                    continue
                median_t = crossed.index[0]
                median_y = 0.5
            else:
                median_t = fitter.median_survival_time_
                if pd.isna(median_t) or np.isinf(median_t):
                    continue
                median_y = 0.5

            if median_t > view_max:
                continue

            # Vertical line from curve down to x-axis
            ax.plot([median_t, median_t], [0, median_y],
                    linestyle=line_style, color=color, linewidth=1, alpha=line_alpha)
            # Horizontal line from y-axis to curve
            ax.plot([0, median_t], [median_y, median_y],
                    linestyle=line_style, color=color, linewidth=1, alpha=line_alpha)

    # --- X-year survival lines ---
    if show_x_year and x_year_time is not None:
        t = x_year_time
        if t <= view_max:
            # Vertical reference line at the timepoint
            ax.axvline(x=t, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)

            for i, fitter in enumerate(fitters):
                color = colors[i] if colors and i < len(colors) else f'C{i}'

                if is_cif:
                    cdf = fitter.cumulative_density_
                    col = cdf.columns[0]
                    # Interpolate to get value at time t
                    if t in cdf.index:
                        y_val = cdf.loc[t, col]
                    else:
                        combined = cdf.index.union([t]).sort_values()
                        interp = cdf[col].reindex(combined).interpolate(method='index')
                        y_val = interp.loc[t]
                else:
                    sf = fitter.survival_function_
                    col = sf.columns[0]
                    if t in sf.index:
                        y_val = sf.loc[t, col]
                    else:
                        combined = sf.index.union([t]).sort_values()
                        interp = sf[col].reindex(combined).interpolate(method='index')
                        y_val = interp.loc[t]

                if pd.isna(y_val):
                    continue

                # Horizontal line from y-axis to the curve at this timepoint
                ax.plot([0, t], [y_val, y_val],
                        linestyle=line_style, color=color, linewidth=1, alpha=line_alpha)


def _get_survival_at_time(fitter, t, is_cif=False):
    """Get point estimate and 95% CI at a specific timepoint."""
    if is_cif:
        curve = fitter.cumulative_density_
        ci = fitter.confidence_interval_cumulative_density_
    else:
        curve = fitter.survival_function_
        ci = fitter.confidence_interval_survival_function_

    col = curve.columns[0]

    def _interp(series, time):
        if time in series.index:
            return series.loc[time]
        combined = series.index.union([time]).sort_values()
        interp = series.reindex(combined).interpolate(method='index')
        return interp.loc[time]

    est = _interp(curve[col], t)
    ci_lo = _interp(ci.iloc[:, 0], t)
    ci_hi = _interp(ci.iloc[:, 1], t)

    if any(pd.isna(v) for v in [est, ci_lo, ci_hi]):
        return None, None, None
    return float(est), float(ci_lo), float(ci_hi)


def _get_median_with_ci(fitter, is_cif=False):
    """Get median survival time and 95% CI."""
    if is_cif:
        cdf = fitter.cumulative_density_
        col = cdf.columns[0]
        crossed = cdf[cdf[col] >= 0.5]
        if crossed.empty:
            return None, None, None
        median_t = crossed.index[0]
        # CI for median is harder for CIF; use point estimate only
        return float(median_t), None, None
    else:
        median_t = fitter.median_survival_time_
        if pd.isna(median_t) or np.isinf(median_t):
            return None, None, None
        # Get CI from the confidence interval of the median
        try:
            ci = fitter.confidence_interval_median_survival_time_
            ci_lo = float(ci.iloc[0, 0]) if not pd.isna(ci.iloc[0, 0]) else None
            ci_hi = float(ci.iloc[0, 1]) if not pd.isna(ci.iloc[0, 1]) else None
        except Exception:
            ci_lo, ci_hi = None, None
        return float(median_t), ci_lo, ci_hi


def add_estimate_labels(fitters, ax, colors=None, labels=None,
                        mode='timepoint', timepoint=36.0, param_name='OS',
                        placement='on_curve', fontsize=9, is_cif=False):
    """
    Add auto-computed survival/CIF estimate text labels on the plot.

    Parameters
    ----------
    fitters : list
        Fitted KaplanMeierFitter or AalenJohansenFitter objects.
    ax : matplotlib Axes
    colors : list of str
    labels : list of str
    mode : str
        'timepoint' for X-year estimate, 'median' for median survival.
    timepoint : float
        Time for point estimate (used when mode='timepoint').
    param_name : str
        Clinical parameter name (e.g. 'OS', 'RFS', 'EFS', 'CIR').
    placement : str
        'on_curve' — label right above each curve at the timepoint.
        'top' — grouped list near top-left of the plot.
        'bottom' — grouped list near bottom-left of the plot.
    fontsize : int
    is_cif : bool
    """
    if not fitters:
        return

    view_min, view_max = ax.get_xlim()
    texts = []

    for i, fitter in enumerate(fitters):
        color = colors[i] if colors and i < len(colors) else f'C{i}'
        lbl = labels[i] if labels and i < len(labels) else fitter._label

        if mode == 'median':
            med, ci_lo, ci_hi = _get_median_with_ci(fitter, is_cif=is_cif)
            if med is None:
                continue
            if ci_lo is not None and ci_hi is not None:
                txt = f"Median {param_name} {med:.1f} (95%CI {ci_lo:.1f}-{ci_hi:.1f})"
            else:
                txt = f"Median {param_name} {med:.1f}"
            # For on_curve placement, place at median time, y=0.5
            y_curve = 0.5
            x_curve = med
        else:  # timepoint
            t = timepoint
            if t > view_max:
                continue
            est, ci_lo, ci_hi = _get_survival_at_time(fitter, t, is_cif=is_cif)
            if est is None:
                continue
            # Determine time label
            if t % 12 == 0 and t >= 12:
                time_label = f"{int(t // 12)}-year"
            else:
                time_label = f"{int(t)}-month" if t == int(t) else f"{t:.1f}-month"
            pct = est * 100
            ci_lo_pct = ci_lo * 100
            ci_hi_pct = ci_hi * 100
            txt = f"{time_label} {param_name} {pct:.1f}% (95%CI {ci_lo_pct:.1f}-{ci_hi_pct:.1f}%)"
            y_curve = est
            x_curve = t

        texts.append({
            'text': txt, 'color': color, 'label': lbl,
            'x_curve': x_curve, 'y_curve': y_curve
        })

    if not texts:
        return

    if placement == 'on_curve':
        for item in texts:
            ax.annotate(
                item['text'],
                xy=(item['x_curve'], item['y_curve']),
                xytext=(8, 8), textcoords='offset points',
                fontsize=fontsize, color=item['color'], weight='bold',
                ha='left', va='bottom'
            )

    elif placement in ('top', 'bottom'):
        # Build a grouped text block
        y_start = 0.95 if placement == 'top' else 0.15
        line_spacing = 0.045 * (fontsize / 9.0)
        x_pos = 0.35

        for idx, item in enumerate(texts):
            y_pos = y_start - (idx * line_spacing)
            line_text = f"{item['label']}    {item['text']}"
            ax.text(x_pos, y_pos, line_text, transform=ax.transAxes,
                    fontsize=fontsize, color=item['color'], weight='bold',
                    ha='left', va='top')


def create_forest_plot(summary_df, theme_color='#1f77b4', title="Forest Plot",
                       xlabel="Hazard Ratio (95% CI)", reference_line=1.0,
                       figsize=None, label_fontsize=10):
    """
    Create a publication-quality forest plot from Cox regression summary.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Must have columns: 'Hazard Ratio (HR)', 'Lower 95%', 'Upper 95%', 'p-value'.
        Index = variable names.
    theme_color : str
        Color for plot elements.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    reference_line : float
        Where to draw the reference (null) line (1.0 for HR).
    figsize : tuple or None
        Figure size. Auto-calculated if None.
    label_fontsize : int
        Font size for variable labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_data = summary_df.copy()
    plot_data = plot_data.sort_index(ascending=False)

    n_vars = len(plot_data)
    if figsize is None:
        figsize = (10, max(4, n_vars * 0.5 + 1))

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n_vars)

    hrs = plot_data['Hazard Ratio (HR)'].values
    lowers = plot_data['Lower 95%'].values
    uppers = plot_data['Upper 95%'].values
    p_vals = plot_data['p-value'].values

    # Error bars (must be positive distances from center)
    xerr = [
        np.abs(hrs - lowers),
        np.abs(uppers - hrs),
    ]

    # Plot error bars
    ax.errorbar(hrs, y_pos, xerr=xerr,
                fmt='s', color=theme_color, ecolor='black',
                capsize=5, markersize=8, linewidth=1.5)

    # Reference line at HR=1
    ax.axvline(x=reference_line, color='red', linestyle='--', linewidth=1, alpha=0.7)

    # Variable labels on Y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data.index, fontsize=label_fontsize, fontweight='bold')

    # HR annotations on the right side
    for i, (hr, lo, hi, p) in enumerate(zip(hrs, lowers, uppers, p_vals)):
        p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
        annotation = f"{hr:.2f} ({lo:.2f}-{hi:.2f}) {p_str}"
        # Place to the right of the plot
        ax.annotate(annotation, xy=(1.02, y_pos[i]),
                    xycoords=('axes fraction', 'data'),
                    fontsize=8, va='center', ha='left',
                    color='black')

    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontweight='bold')

    # Grid
    ax.grid(True, axis='x', linestyle=':', alpha=0.6)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Adjust limits
    ax.set_ylim(-0.5, n_vars - 0.5 + 0.3)

    # Make room for annotations on the right
    fig.subplots_adjust(right=0.65)

    return fig


def save_plot_to_buffer(fig, fmt="png", dpi=300):
    """
    Save a matplotlib figure to an in-memory buffer.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    fmt : str
        Output format ('png', 'pdf', etc.)
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    io.BytesIO
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    return buf
