"""
Tab 1: Univariable (Kaplan-Meier) Analysis.
Extracted from app.py for modularity.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from lifelines.utils import median_survival_times


def render(df_clean, time_col, event_col, group_col, config):
    """
    Render the Univariable (Kaplan-Meier) analysis tab.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned dataframe (NaN-dropped for key columns).
    time_col : str
        Column name for survival time.
    event_col : str
        Column name for event indicator.
    group_col : str
        Grouping variable name (or "None").
    config : dict
        Shared configuration from sidebar (themes, fonts, plot settings, etc.).
    """
    from modules.plotting import add_at_risk_counts, save_plot_to_buffer

    dropped_count = config.get("dropped_count", 0)
    cols_to_check = config.get("cols_to_check", [])
    df = config.get("df_original")

    if dropped_count > 0:
        st.warning(
            f"Missing Values Detected: Dropped {dropped_count} rows containing NaNs "
            f"in the selected columns. Analysis based on {len(df_clean)} remaining rows."
        )
        with st.expander("See details of missing values"):
            st.write("Number of missing values per column (in the original selection):")
            if df is not None:
                missing_stats = df[cols_to_check].isna().sum()
                st.dataframe(missing_stats[missing_stats > 0])

    # Calculate P-value for Plot
    p_value_text = None
    if config.get("show_p_val_plot") and group_col != "None" and len(df_clean[group_col].unique()) >= 2:
        try:
            res = multivariate_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
            p_value_text = "p < 0.0001" if res.p_value < 0.0001 else f"p = {res.p_value:.4f}"
        except Exception:
            p_value_text = None

    # --- Plotting ---
    plt.rcParams['font.family'] = config.get("selected_font", "sans-serif")
    fig, ax = plt.subplots(figsize=(config["plot_width"], config["plot_height"]))

    fig.patch.set_facecolor(config.get("plot_bgcolor", "#FFFFFF"))
    ax.set_facecolor(config.get("plot_bgcolor", "#FFFFFF"))
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)

    # Determine Ticks
    max_time = df_clean[time_col].max()
    tick_interval = config.get("tick_interval")
    enable_zoom = config.get("enable_zoom", False)
    zoom_max = config.get("zoom_max", max_time)

    if enable_zoom:
        ax.set_xlim(0, zoom_max)
        if tick_interval:
            ax.set_xticks(np.arange(0, zoom_max + tick_interval, tick_interval))
    elif tick_interval:
        custom_ticks = np.arange(0, max_time + tick_interval, tick_interval)
        ax.set_xticks(custom_ticks)
        ax.set_xlim(0, custom_ticks[-1])
    else:
        ax.set_xlim(left=0)

    y_min = config.get("y_min", 0.0)
    y_max = config.get("y_max", 1.0)
    y_tick_interval = config.get("y_tick_interval")
    if y_tick_interval:
        ax.set_yticks(np.arange(y_min, y_max + y_tick_interval / 10, y_tick_interval))
    ax.set_ylim(y_min, y_max)

    all_themes = config.get("all_themes", {})
    selected_theme = config.get("selected_theme", "Default")
    show_ci = config.get("show_ci", True)
    show_censored = config.get("show_censored", True)
    line_width = config.get("line_width", 1.5)
    groups_ordered = config.get("groups_ordered", [])
    group_labels = config.get("group_labels", {})
    custom_colors = config.get("custom_colors", {})

    fitters = []
    plot_colors = []
    plot_labels = []

    if group_col != "None":
        groups = groups_ordered if groups_ordered else sorted(df_clean[group_col].unique())
        palette = all_themes.get(selected_theme)

        for i, group in enumerate(groups):
            mask = df_clean[group_col] == group
            color = None
            if selected_theme == "Custom":
                color = custom_colors.get(group)
            elif palette:
                color = palette[i % len(palette)]
            if color is None:
                color = f"C{i}"

            plot_colors.append(color)
            label = group_labels.get(group, str(group))
            plot_labels.append(label)

            kmf = KaplanMeierFitter()
            kmf.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=label)
            kmf.plot_survival_function(
                ax=ax, ci_show=show_ci, show_censors=show_censored,
                color=color, linewidth=line_width
            )
            fitters.append(kmf)

        # P-value annotation
        if config.get("show_p_val_plot") and p_value_text:
            bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if config.get("show_p_val_box_main") else None
            ax.text(config.get("pval_x_main", 0.95), config.get("pval_y_main", 0.05),
                    p_value_text, transform=ax.transAxes, ha='right', va='bottom',
                    bbox=bbox_props, fontsize=config.get("p_val_fontsize", 12))

        # Risk Table
        if config.get("show_risk_table"):
            add_at_risk_counts(fitters, ax=ax, y_shift=config.get("table_height", -0.25),
                               colors=plot_colors, labels=plot_labels)

    else:
        # Single group
        color = None
        if selected_theme in all_themes and len(all_themes[selected_theme]) > 0:
            color = all_themes[selected_theme][0]
        elif selected_theme == "Custom":
            color = st.sidebar.color_picker("Color for All Patients", "#1f77b4")

        kmf_all = KaplanMeierFitter()
        kmf_all.fit(df_clean[time_col], df_clean[event_col], label="All Patients")
        kmf_all.plot_survival_function(
            ax=ax, ci_show=show_ci, show_censors=show_censored,
            color=color, linewidth=line_width
        )
        fitters.append(kmf_all)
        plot_colors = [color] if color else None
        plot_labels = ["All Patients"]

        if config.get("show_risk_table"):
            add_at_risk_counts([kmf_all], ax=ax, y_shift=config.get("table_height", -0.25),
                               colors=plot_colors, labels=plot_labels)

    # Apply titles and labels
    ax.set_title(config.get("main_title", "Survival"),
                 fontsize=config.get("title_fontsize", 20),
                 weight=config.get("title_fontweight", "bold"))
    ax.set_xlabel(config.get("x_label", "Time (Months)"), fontsize=config.get("axes_fontsize", 12))
    y_label = config.get("y_label", "Survival Probability")
    if y_label:
        ax.set_ylabel(y_label, fontsize=config.get("axes_fontsize", 12))
    ax.tick_params(axis='both', which='major', labelsize=config.get("axes_fontsize", 12))

    # Legend
    if config.get("show_legend_main", True):
        ax.legend(fontsize=config.get("legend_fontsize", 10),
                  loc=(config.get("leg_x_main", 0.8), config.get("leg_y_main", 0.9)),
                  frameon=config.get("show_legend_box_main", True))
    else:
        if ax.get_legend():
            ax.get_legend().remove()

    # Annotations
    for ann in config.get("main_annotations", []):
        bbox_props = dict(facecolor='white', alpha=0.5, boxstyle='round') if ann.get('box') else None
        ax.text(ann['x'], ann['y'], ann['text'], transform=ax.transAxes,
                ha='center', va='center', bbox=bbox_props, fontsize=ann.get('size', 12))

    st.pyplot(fig)
    st.session_state['report_fig_km'] = fig

    # Download buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download Plot (300 DPI)", save_plot_to_buffer(fig, dpi=300),
                           "survival_plot_300dpi.png", "image/png")
    with col2:
        st.download_button("Download High-Res Plot (600 DPI)", save_plot_to_buffer(fig, dpi=600),
                           "survival_plot_600dpi.png", "image/png")
    with col3:
        st.download_button("Download Plot (PDF)", save_plot_to_buffer(fig, fmt="pdf"),
                           "survival_plot.pdf", "application/pdf")

    # --- Statistics Section ---
    if group_col != "None":
        groups = groups_ordered if groups_ordered else sorted(df_clean[group_col].unique())
        _render_statistics(df_clean, time_col, event_col, group_col, groups, group_labels, fitters, config)


def _render_statistics(df_clean, time_col, event_col, group_col, groups, group_labels, fitters, config):
    """Render the statistics section below the KM plot."""
    st.divider()
    st.subheader("Statistical Analysis")

    # Log-Rank Test
    if len(groups) >= 2:
        result = multivariate_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
        st.write(f"**Log-Rank Test p-value**: {result.p_value:.4f}")

    # Cox PH
    st.subheader("Cox Proportional Hazards (Hazard Ratios)")
    unique_values = sorted(df_clean[group_col].unique())
    reference_group = st.selectbox("Select Reference Group (Baseline)", unique_values, index=0)

    try:
        all_groups = sorted(df_clean[group_col].unique())
        cox_data = df_clean[[time_col, event_col]].copy()
        for group in all_groups:
            if str(group) == str(reference_group):
                continue
            cox_data[str(group)] = (df_clean[group_col] == group).astype(int)

        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col=time_col, event_col=event_col)

        summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
        summary_df = summary_df.rename(columns={
            'exp(coef)': 'Hazard Ratio (HR)',
            'exp(coef) lower 95%': 'Lower 95% CI',
            'exp(coef) upper 95%': 'Upper 95% CI',
            'p': 'p-value'
        })

        st.dataframe(summary_df.style.format("{:.3f}"))
        st.session_state['uv_cox_summary'] = summary_df
        st.session_state['uv_cox_method'] = "Standard Cox Proportional Hazards regression models"

        csv_cox = summary_df.to_csv().encode('utf-8')
        st.download_button("Download Cox HR Table", csv_cox, "cox_ph_table.csv", "text/csv")
        st.caption(f"Reference Group: **{reference_group}**")

    except Exception as e:
        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(cox_data, duration_col=time_col, event_col=event_col)
            st.warning(
                "**Convergence Warning**: Standard Cox model failed. "
                "Automatically applied **Penalized Cox (Ridge, lambda=0.1)**."
            )
            summary_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
            summary_df = summary_df.rename(columns={
                'exp(coef)': 'Hazard Ratio (HR)',
                'exp(coef) lower 95%': 'Lower 95% CI',
                'exp(coef) upper 95%': 'Upper 95% CI',
                'p': 'p-value'
            })
            st.dataframe(summary_df.style.format("{:.3f}"))
            st.session_state['uv_cox_summary'] = summary_df
            st.session_state['uv_cox_method'] = "Penalized Cox Regression (Ridge, Lambda=0.1)"
        except Exception as e2:
            st.error(f"Cox Model Error: {e}")
            st.info(
                "**Tip**: This often happens if a group has too few events or perfectly "
                "predicts survival (separation). Try merging small groups or using penalized regression."
            )

    # Median Survival
    st.subheader("Median Survival Time")
    median_data = []
    for group in groups:
        mask = df_clean[group_col] == group
        kmf_med = KaplanMeierFitter()
        kmf_med.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=str(group))
        median_os = kmf_med.median_survival_time_
        try:
            median_ci_df = median_survival_times(kmf_med.confidence_interval_)
            lower = median_ci_df.iloc[0, 0]
            upper = median_ci_df.iloc[0, 1]
            ci_str = f"({lower:.1f} - {upper:.1f})"
        except Exception:
            ci_str = "(NR - NR)"

        med_str = f"{median_os:.1f}" if not np.isinf(median_os) else "NR"
        label = group_labels.get(group, str(group))
        median_data.append({
            "Group": label,
            "Median Survival": med_str,
            "95% CI (Median)": ci_str,
        })

    median_df = pd.DataFrame(median_data)
    st.table(median_df)
    csv_med = median_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Median Survival Table", csv_med, "median_survival_table.csv", "text/csv")

    # Point-in-Time Estimates
    st.subheader("Point-in-Time Survival Estimates")
    st.write("Calculate survival probability at a specific time (e.g., 2-year OS).")
    target_time = st.number_input("Enter Time Point (e.g., 24 months)", min_value=0.0, value=24.0, step=6.0)

    est_data = []
    for group in groups:
        mask = df_clean[group_col] == group
        kmf_est = KaplanMeierFitter()
        kmf_est.fit(df_clean[time_col][mask], df_clean[event_col][mask], label=str(group))
        surv_prob = kmf_est.survival_function_at_times(target_time).iloc[0]
        ci_df = kmf_est.confidence_interval_survival_function_
        combined_index = ci_df.index.union([target_time]).sort_values()
        ci_df_interp = ci_df.reindex(combined_index).ffill()
        try:
            lower = ci_df_interp.loc[target_time].iloc[0]
            upper = ci_df_interp.loc[target_time].iloc[1]
        except Exception:
            lower, upper = 0, 0

        label = group_labels.get(group, str(group))
        est_data.append({
            "Group": label,
            f"Survival at {target_time}": f"{surv_prob:.1%}",
            "95% CI": f"({lower:.1%} - {upper:.1%})",
        })

    est_df = pd.DataFrame(est_data)
    st.table(est_df)
    csv_pit = est_df.to_csv(index=False).encode('utf-8')
    st.download_button(f"Download Survival Estimates at {target_time}", csv_pit,
                       f"survival_at_{target_time}.csv", "text/csv")

    # Pairwise Comparisons
    if len(groups) > 2:
        st.subheader("Pairwise Log-Rank Comparisons")
        results = pairwise_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
        pairwise_df = results.summary
        st.dataframe(pairwise_df.style.format({"p": "{:.4f}"}))

    # AI Narrator
    st.divider()
    st.subheader("AI Result Narrator")
    if st.button("Generate Summary Text (Univariable)"):
        cox_method_text = st.session_state.get('uv_cox_method', 'Standard Cox Proportional Hazards regression models')
        summary = "**Methods**\n"
        summary += f"Survival estimates were calculated using the Kaplan-Meier method. Comparisons between groups were performed using the Log-rank test. Univariable associations were assessed using {cox_method_text}.\n\n"
        summary += "**Results**\n"

        result = multivariate_logrank_test(df_clean[time_col], df_clean[group_col], df_clean[event_col])
        sig_word = "significantly" if result.p_value < 0.05 else "not significantly"
        summary += f"The Kaplan-Meier survival analysis comparing groups defined by **{group_col}** ({', '.join([str(g) for g in groups])}) revealed that {group_col} was **{sig_word} associated with survival** (Log-rank test p={result.p_value:.4f}). "

        if 'uv_cox_summary' in st.session_state:
            summary += "In the univariable Cox regression:\n"
            cox_df = st.session_state['uv_cox_summary']
            for idx, row in cox_df.iterrows():
                hr = row['Hazard Ratio (HR)']
                p = row['p-value']
                ci_low = row['Lower 95% CI']
                ci_high = row['Upper 95% CI']
                summary += f"* **{idx}**: HR={hr:.2f} (95% CI {ci_low:.2f}-{ci_high:.2f}, p={p:.4f})\n"

        med_details = []
        for item in median_data:
            med_details.append(f"{item['Group']} (Median: {item['Median Survival']}, 95% CI: {item['95% CI (Median)']})")
        summary += "\nMedian survival times were: " + "; ".join(med_details) + "."

        st.success("Summary Generated:")
        st.text_area("Copy this text:", value=summary, height=200)
