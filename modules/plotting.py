import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import numpy as np


def add_at_risk_counts(fitters, ax=None, y_shift=-0.25, colors=None, labels=None,
                       fontsize=10, show_censored_counts=False, bold=True,
                       show_title=False, label_pad=-0.12):
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
