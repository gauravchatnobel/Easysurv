"""
AI Narrator Module for EasySurv.

Generates publication-ready methods and results text from statistical outputs.
Supports multiple journal styles, concise/detailed modes, and provides
clinical interpretation of results.
"""

import numpy as np


# ============================================================
# Style Definitions
# ============================================================

def _p_no_leading_zero(p):
    """JCO/Blood house style: p-value with no leading zero, guarded at both
    extremes so a value near 1.0 never renders as '.000' (which reads as
    highly significant) and a value near 0 never renders as '.000' either."""
    if p < 0.001:
        return "P < .001"
    if p >= 0.9995:          # would round to 1.000
        return "P > .99"
    return f"P = .{f'{p:.3f}'[2:]}"


JOURNAL_STYLES = {
    "Standard": {
        "label": "Standard (Most Journals)",
        "ci_format": "({low}-{high})",
        "ci_sep": "-",
        "p_format": lambda p: "p<0.001" if p < 0.001 else ("p>0.999" if p >= 0.9995 else f"p={p:.3f}"),
        "hr_inline": True,
        "bold_significant": True,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
        "p_capitalize": False,
        "p_leading_zero": True,
    },
    "NEJM": {
        "label": "NEJM Style",
        "ci_format": "({low} to {high})",
        "ci_sep": " to ",
        "p_format": lambda p: "P<0.001" if p < 0.001 else ("P>0.99" if p >= 0.995 else f"P={p:.2f}"),
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
        "p_capitalize": True,
        "p_leading_zero": True,
    },
    "Lancet": {
        "label": "Lancet Style",
        "ci_format": "({low}-{high})",
        "ci_sep": "-",
        "p_format": lambda p: "p<0.0001" if p < 0.0001 else ("p>0.9999" if p >= 0.99995 else f"p={p:.4f}"),
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
        "p_capitalize": False,
        "p_leading_zero": True,
    },
    "JCO": {
        "label": "JCO (Journal of Clinical Oncology)",
        "ci_format": "({low} to {high})",
        "ci_sep": " to ",
        "p_format": lambda p: _p_no_leading_zero(p),
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
        "p_capitalize": True,
        "p_leading_zero": False,
        "p_italic": True,
    },
    "Blood": {
        "label": "Blood (ASH)",
        "ci_format": "({low}-{high})",
        "ci_sep": "-",
        "p_format": lambda p: _p_no_leading_zero(p),
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
        "p_capitalize": True,
        "p_leading_zero": False,
        "p_italic": True,
    },
}

DETAIL_LEVELS = ["Concise", "Detailed"]


# ============================================================
# Helper Functions
# ============================================================

def _interpret_hr(hr, variable_name=""):
    """
    Translate a hazard ratio into plain clinical language.

    Uses *hazard* wording rather than absolute "risk", because an HR
    describes the relative hazard rate, not a change in absolute risk.
    "Associated with" keeps the statement non-causal, appropriate for
    observational data. Callers should reserve this effect-size wording
    for statistically significant, precisely estimated results.

    Examples:
        HR=2.45 -> "associated with a 2.5-fold higher hazard"
        HR=0.45 -> "associated with a 55% lower hazard"
        HR=1.02 -> "not meaningfully associated with the hazard"
    """
    if hr > 5.0:
        return f"associated with a markedly higher hazard ({hr:.1f}-fold)"
    elif hr > 2.0:
        return f"associated with a {hr:.1f}-fold higher hazard of the event"
    elif hr > 1.5:
        pct = (hr - 1) * 100
        return f"associated with a {pct:.0f}% higher hazard of the event"
    elif hr > 1.1:
        pct = (hr - 1) * 100
        return f"associated with a modestly higher hazard ({pct:.0f}% higher)"
    elif hr >= 0.9:
        return "not meaningfully associated with the hazard"
    elif hr >= 0.67:
        pct = (1 - hr) * 100
        return f"associated with a {pct:.0f}% lower hazard of the event"
    elif hr >= 0.5:
        pct = (1 - hr) * 100
        return f"associated with a substantially lower hazard ({pct:.0f}% lower)"
    else:
        fold = 1 / hr
        return f"associated with a markedly lower hazard ({fold:.1f}-fold lower)"


def _interpret_shr(shr):
    """Interpret subdistribution hazard ratio (Fine-Gray)."""
    if shr > 2.0:
        return f"associated with a {shr:.1f}-fold higher cumulative incidence"
    elif shr > 1.1:
        pct = (shr - 1) * 100
        return f"associated with a {pct:.0f}% higher cumulative incidence"
    elif shr >= 0.9:
        return "not meaningfully associated with cumulative incidence"
    elif shr >= 0.5:
        pct = (1 - shr) * 100
        return f"associated with a {pct:.0f}% lower cumulative incidence"
    else:
        fold = 1 / shr
        return f"associated with a markedly lower cumulative incidence ({fold:.1f}-fold)"


def _format_ci(low, high, style):
    """Format confidence interval according to journal style."""
    return style["ci_format"].format(low=f"{low:.2f}", high=f"{high:.2f}")


def _format_p(p, style):
    """Format p-value according to journal style (with markdown italic if applicable)."""
    base = style["p_format"](p)
    if style.get("p_italic") and base.startswith("P"):
        base = "*P*" + base[1:]
    return base


def _significance_phrase(p, style):
    """Return significance phrase."""
    if p < 0.05:
        if style.get("bold_significant"):
            return "**significantly**"
        return "significantly"
    return "not significantly"


def _format_median_ci(med_str, ci_str_raw, style):
    """
    Re-format pre-formatted median CI string to match journal style.

    Parameters
    ----------
    med_str : str
        Median value as string (e.g. "18.6" or "NR").
    ci_str_raw : str
        Raw CI string from app (e.g. "(12.3 - 25.1)" or "(NR - NR)").
    style : dict
        Journal style dict.

    Returns
    -------
    str
        Formatted string like "18.6 months (95% CI 12.3-25.1)" or
        "18.6 months (95% CI 12.3 to 25.1)" depending on style.
    """
    # Parse raw CI string: "(12.3 - 25.1)" or "(NR - NR)"
    ci_clean = ci_str_raw.strip().strip("()")
    parts = [p.strip() for p in ci_clean.split("-", 1)]
    if len(parts) == 2:
        lo, hi = parts[0].strip(), parts[1].strip()
    else:
        return f"{med_str} {ci_str_raw}"

    sep = style.get("ci_sep", "-")
    return f"{med_str} (95% CI {lo}{sep}{hi})"


def _worst_median_group(median_data):
    """Return the group label with the shortest (numeric) median survival, or None.

    Non-numeric medians (e.g. 'NR' / 'Not Reached') are treated as the longest
    survival and never selected as worst.
    """
    if not median_data:
        return None
    worst_label, worst_val = None, None
    for item in median_data:
        raw = str(item.get('Median Survival', '')).strip()
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if worst_val is None or val < worst_val:
            worst_val, worst_label = val, item.get('Group')
    return worst_label


def _event_display_name(event_name):
    """
    Convert short parameter codes to readable names.
    OS -> overall survival, RFS -> relapse-free survival, etc.
    Falls back to the raw name if no mapping found.
    """
    mapping = {
        "OS": "overall survival",
        "RFS": "relapse-free survival",
        "EFS": "event-free survival",
        "DFS": "disease-free survival",
        "PFS": "progression-free survival",
        "TTP": "time to progression",
        "LRFS": "local relapse-free survival",
        "DRFS": "distant relapse-free survival",
        "CSS": "cancer-specific survival",
    }
    return mapping.get(event_name.upper(), event_name)


# ============================================================
# Univariable (Kaplan-Meier) Narrator
# ============================================================

def generate_univariable_narrative(
    group_col,
    groups,
    logrank_p,
    cox_summary=None,
    median_data=None,
    point_estimates=None,
    target_time=None,
    cox_method="Cox Proportional Hazards regression",
    style_name="Standard",
    detail_level="Detailed",
    event_name=None,
    n_patients=None,
    n_events=None,
    landmark_time=None,
    median_followup=None,
):
    """
    Generate a narrative for univariable (Kaplan-Meier) survival analysis.

    Parameters
    ----------
    group_col : str
        Name of the grouping variable.
    groups : list
        List of group names.
    logrank_p : float
        Log-rank test p-value.
    cox_summary : pd.DataFrame or None
        Cox regression summary with columns:
        'Hazard Ratio (HR)', 'Lower 95% CI', 'Upper 95% CI', 'p-value'.
    median_data : list of dict or None
        Each dict has 'Group', 'Median Survival', '95% CI (Median)'.
    point_estimates : list of dict or None
        Each dict has 'Group', survival value, and '95% CI'.
    target_time : float or None
        Time point for point estimates.
    cox_method : str
        Description of Cox method used.
    style_name : str
        Journal style name.
    detail_level : str
        "Concise" or "Detailed".
    event_name : str or None
        Parameter name (e.g. "OS", "RFS"). Used for natural language.
    n_patients : int or None
        Number of patients in the analysis.
    n_events : int or None
        Number of events observed.
    landmark_time : float or None
        If > 0, landmark analysis was applied.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    concise = detail_level == "Concise"
    endpoint_name = _event_display_name(event_name) if event_name else "survival"
    endpoint_short = event_name if event_name else "OS"

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    if concise:
        text += (
            f"Kaplan-Meier estimates of {endpoint_name} were compared using the log-rank test. "
            f"Hazard ratios were estimated using {cox_method}."
        )
    else:
        text += (
            f"Survival estimates for {endpoint_name} were calculated using the Kaplan-Meier method. "
            "Between-group comparisons were performed using the log-rank test. "
            f"Univariable hazard ratios (HRs) with 95% confidence intervals (CIs) were estimated using {cox_method}."
        )

    if landmark_time and landmark_time > 0:
        text += (
            f" A landmark analysis at {landmark_time:.0f} months was applied; "
            "only patients event-free at the landmark were included, and time zero was reset to the landmark."
        )

    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Lead-in: median follow-up (reverse Kaplan-Meier) + cohort context, as a
    # single flowing clause that runs into the median-survival sentence.
    lead = ""
    if median_followup and median_followup > 0:
        lead += f"After a median follow-up of {median_followup:.1f} months, "
    if n_patients and n_events:
        lead += f"{'among' if lead else 'Among'} {n_patients} patients ({n_events} events), "
    elif n_patients:
        lead += f"{'among' if lead else 'Among'} {n_patients} patients, "
    text += lead

    # Median survival (lead with the clinical finding)
    if median_data:
        med_parts = []
        for item in median_data:
            med_val = item['Median Survival']
            ci_val = item['95% CI (Median)']
            formatted = _format_median_ci(med_val, ci_val, style)
            label = item['Group']
            med_parts.append(f"{label}: {formatted}")

        text += f"{'median' if lead else 'Median'} {endpoint_short} by **{group_col}** was "
        text += "; ".join(med_parts) + ". "
    elif lead:
        # Close the lead-in clause if there is no median sentence to attach to.
        text = text.rstrip().rstrip(",") + ". "

    # Determine the worst-outcome group for direction-aware phrasing
    worst_group = _worst_median_group(median_data)

    # Log-rank result (direction-aware when significant)
    p_str = _format_p(logrank_p, style)
    if logrank_p < 0.05:
        sig_phrase = "**statistically significant**" if style.get("bold_significant") else "statistically significant"
        text += f"The difference between groups was {sig_phrase} ({p_str})"
        if worst_group is not None and len(median_data) >= 2:
            text += f", with the shortest median {endpoint_short} observed in **{worst_group}**."
        else:
            text += "."
    else:
        text += f"The difference between groups was not statistically significant ({p_str})."

    # Point-in-time estimates
    if point_estimates and target_time is not None:
        text += f"\n\nAt {target_time:.0f} months: "
        pit_parts = []
        for item in point_estimates:
            surv_key = next((k for k in item.keys() if k.startswith("Survival")), None)
            if surv_key:
                pit_parts.append(f"{item['Group']} {item[surv_key]} (95% CI {item['95% CI']})")
        text += "; ".join(pit_parts) + "."

    # Cox regression results with clinical interpretation
    if cox_summary is not None and len(cox_summary) > 0:
        text += "\n\n"
        if concise:
            # Compact bullet list
            for idx, row in cox_summary.iterrows():
                hr = row['Hazard Ratio (HR)']
                p = row['p-value']
                if 'Lower 95% CI' in row.index:
                    ci_low, ci_high = row['Lower 95% CI'], row['Upper 95% CI']
                else:
                    ci_low, ci_high = row['Lower 95%'], row['Upper 95%']
                ci_str = _format_ci(ci_low, ci_high, style)
                p_str = _format_p(p, style)
                text += f"* **{idx}**: HR {hr:.2f}, 95% CI {ci_str}, {p_str}\n"
        else:
            text += "Univariable Cox regression:\n"
            # Only narrate an effect size for statistically significant results;
            # for non-significant HRs, report the numbers without interpretation
            # (a wide/non-significant HR is not evidence of a real effect).
            _sig, _nonsig = [], []
            for idx, row in cox_summary.iterrows():
                hr = row['Hazard Ratio (HR)']
                p = row['p-value']
                if 'Lower 95% CI' in row.index:
                    ci_low, ci_high = row['Lower 95% CI'], row['Upper 95% CI']
                else:
                    ci_low, ci_high = row['Lower 95%'], row['Upper 95%']
                (_sig if p < 0.05 else _nonsig).append((idx, hr, p, ci_low, ci_high))

            for idx, hr, p, ci_low, ci_high in _sig:
                ci_str = _format_ci(ci_low, ci_high, style)
                p_str = _format_p(p, style)
                interpretation = _interpret_hr(hr, idx)
                text += f"* **{idx}** was {interpretation} (HR {hr:.2f}, 95% CI {ci_str}, {p_str}).\n"

            if _nonsig:
                text += "The following were not statistically significant:\n"
                for idx, hr, p, ci_low, ci_high in _nonsig:
                    ci_str = _format_ci(ci_low, ci_high, style)
                    p_str = _format_p(p, style)
                    text += f"* **{idx}**: HR {hr:.2f}, 95% CI {ci_str}, {p_str}.\n"

    return text


# ============================================================
# Multivariable (Cox Regression) Narrator
# ============================================================

def generate_multivariable_narrative(
    summary_df,
    use_penalizer=False,
    penalizer_value=0.0,
    l1_ratio=0.0,
    n_patients=None,
    n_events=None,
    style_name="Standard",
    detail_level="Detailed",
    event_name=None,
    landmark_time=None,
    median_followup=None,
):
    """
    Generate a narrative for multivariable Cox regression analysis.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Regression results with columns:
        'Hazard Ratio (HR)', 'Lower 95%', 'Upper 95%', 'p-value'.
    use_penalizer : bool
        Whether penalized regression was used.
    penalizer_value : float
        Lambda penalty value.
    l1_ratio : float
        L1 mixing ratio (0=Ridge, 1=Lasso).
    n_patients : int or None
        Number of patients in the analysis.
    n_events : int or None
        Number of events observed.
    style_name : str
        Journal style name.
    detail_level : str
        "Concise" or "Detailed".
    event_name : str or None
        Parameter name (e.g. "OS", "RFS").
    landmark_time : float or None
        If > 0, landmark analysis was applied.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    concise = detail_level == "Concise"
    endpoint_name = _event_display_name(event_name) if event_name else "the time-to-event outcome"

    # --- Methods ---
    text = f"{style['methods_header']}\n"

    if use_penalizer and penalizer_value > 0:
        if l1_ratio == 0:
            penalty_type = "Ridge"
        elif l1_ratio == 1:
            penalty_type = "Lasso"
        else:
            penalty_type = "Elastic Net"

        if concise:
            text += (
                f"Penalized Cox regression ({penalty_type}, "
                f"\u03bb={penalizer_value:.4f}) was used for multivariable analysis."
            )
        else:
            text += (
                f"Multivariable analysis was performed using penalized Cox regression "
                f"({penalty_type}, lambda={penalizer_value:.4f}, L1 ratio={l1_ratio:.1f}) "
                f"to account for potential multicollinearity and prevent overfitting."
            )
    else:
        if concise:
            text += (
                f"Multivariable Cox regression was used to identify independent predictors of {endpoint_name}."
            )
        else:
            text += (
                f"Multivariable analysis was performed using the Cox proportional hazards model "
                f"to identify independent predictors of {endpoint_name}."
            )

    if n_patients and n_events:
        text += f" The analysis included {n_patients} patients with {n_events} events."

    if median_followup and median_followup > 0:
        text += f" The median follow-up was {median_followup:.1f} months."

    if landmark_time and landmark_time > 0:
        text += (
            f" Landmark analysis at {landmark_time:.0f} months was applied."
        )

    if not concise:
        text += " Hazard ratios (HRs) with 95% confidence intervals (CIs) are reported."

    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Separate significant and non-significant
    sig_vars = []
    nonsig_vars = []

    for idx, row in summary_df.iterrows():
        hr = row['Hazard Ratio (HR)']
        p = row['p-value']
        ci_low = row['Lower 95%']
        ci_high = row['Upper 95%']
        entry = {'name': idx, 'hr': hr, 'p': p, 'ci_low': ci_low, 'ci_high': ci_high}
        if p < 0.05:
            sig_vars.append(entry)
        else:
            nonsig_vars.append(entry)

    if concise:
        # All variables in a compact list, significant first
        for v in sig_vars + nonsig_vars:
            ci_str = _format_ci(v['ci_low'], v['ci_high'], style)
            p_str = _format_p(v['p'], style)
            sig_marker = " *" if v['p'] < 0.05 else ""
            text += f"* **{v['name']}**: HR {v['hr']:.2f}, 95% CI {ci_str}, {p_str}{sig_marker}\n"
    else:
        # Report significant predictors first with interpretation
        if sig_vars:
            text += "The following variables were independently associated with the outcome:\n"
            for v in sig_vars:
                ci_str = _format_ci(v['ci_low'], v['ci_high'], style)
                p_str = _format_p(v['p'], style)
                interp = _interpret_hr(v['hr'], v['name'])
                text += f"* **{v['name']}** was {interp} (HR {v['hr']:.2f}, 95% CI {ci_str}, {p_str}).\n"

        if nonsig_vars:
            text += "\nThe following variables were not statistically significant in the adjusted model:\n"
            for v in nonsig_vars:
                ci_str = _format_ci(v['ci_low'], v['ci_high'], style)
                p_str = _format_p(v['p'], style)
                text += f"* **{v['name']}**: HR {v['hr']:.2f}, 95% CI {ci_str}, {p_str}.\n"

    return text


# ============================================================
# Competing Risks (CIF) Narrator
# ============================================================

def generate_cif_narrative(
    cif_median_data=None,
    cif_est_data=None,
    cif_target_time=None,
    fg_summary=None,
    fg_mv_summary=None,
    style_name="Standard",
    detail_level="Detailed",
    event_of_interest=None,
    competing_event=None,
    n_patients=None,
    n_primary_events=None,
    n_competing_events=None,
    landmark_time=None,
    median_followup=None,
):
    """
    Generate a narrative for competing risks analysis.

    Parameters
    ----------
    cif_median_data : list of dict or None
        Each dict has 'Group', 'Median Time to Incidence', '95% CI (Median)'.
    cif_est_data : list of dict or None
        Point-in-time cumulative incidence estimates.
    cif_target_time : float or None
        Time point for point estimates.
    fg_summary : pd.DataFrame or None
        Univariable Fine-Gray regression summary with columns:
        'Subdist HR', 'Lower 95%', 'Upper 95%', 'p-value'.
    fg_mv_summary : pd.DataFrame or None
        Multivariable Fine-Gray regression summary with columns:
        'Variable', 'aSHR', 'Lower 95%', 'Upper 95%', 'p-value'.
    style_name : str
        Journal style name.
    detail_level : str
        "Concise" or "Detailed".
    event_of_interest : str or None
        Name of the primary event (e.g. "relapse", "NRM").
    competing_event : str or None
        Name of the competing event (e.g. "death without relapse").
    n_patients : int or None
        Total patients.
    n_primary_events : int or None
        Number of primary events.
    n_competing_events : int or None
        Number of competing events.
    landmark_time : float or None
        If > 0, landmark analysis was applied.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    concise = detail_level == "Concise"
    event_label = event_of_interest or "the event of interest"
    compete_label = competing_event or "competing events"

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    if concise:
        text += (
            f"Cumulative incidence of {event_label} was estimated using the Aalen-Johansen method "
            f"with {compete_label} as a competing risk."
        )
    else:
        text += (
            f"Cumulative incidence functions (CIF) for {event_label} were estimated using the "
            f"Aalen-Johansen method, accounting for {compete_label} as a competing risk. "
            "The effect of covariates on the subdistribution hazard was assessed using "
            "the Fine-Gray model."
        )

    if landmark_time and landmark_time > 0:
        text += f" Landmark analysis at {landmark_time:.0f} months was applied."

    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Cohort context
    if median_followup and median_followup > 0:
        text += f"After a median follow-up of {median_followup:.1f} months, "
    if n_patients:
        parts = [f"{n_patients} patients"]
        if n_primary_events is not None:
            parts.append(f"{n_primary_events} {event_label} events")
        if n_competing_events is not None:
            parts.append(f"{n_competing_events} {compete_label}")
        _lead = "among" if (median_followup and median_followup > 0) else "Among"
        text += _lead + " " + ", ".join(parts) + ". "

    # Point-in-time cumulative incidence
    if cif_est_data and cif_target_time is not None:
        col_name = f"Cumulative Incidence at {cif_target_time}"
        pit_parts = []
        for item in cif_est_data:
            val = item.get(col_name, "N/A")
            ci = item.get("95% CI", "")
            pit_parts.append(f"{item['Group']} {val} (95% CI {ci})")
        text += f"At {cif_target_time:.0f} months, the cumulative incidence of {event_label} was: "
        text += "; ".join(pit_parts) + ". "

    # Median time to incidence
    if cif_median_data:
        med_parts = []
        for item in cif_median_data:
            med_val = item['Median Time to Incidence']
            ci_val = item['95% CI (Median)']
            formatted = _format_median_ci(med_val, ci_val, style)
            med_parts.append(f"{item['Group']}: {formatted}")
        text += "Median time to incidence was " + "; ".join(med_parts) + "."

    # Fine-Gray regression with clinical interpretation
    if fg_summary is not None:
        text += "\n\n"
        if concise:
            for idx, row in fg_summary.iterrows():
                shr = row['Subdist HR']
                p = row['p-value']
                ci_low = row['Lower 95%']
                ci_high = row['Upper 95%']
                ci_str = _format_ci(ci_low, ci_high, style)
                p_str = _format_p(p, style)
                text += f"* **{idx}**: SHR {shr:.2f}, 95% CI {ci_str}, {p_str}\n"
        else:
            text += "**Fine-Gray regression** (subdistribution hazard model):\n"
            # Interpret the effect size only for significant SHRs; otherwise report
            # the numbers without asserting a real effect (consistent with the Cox narrator).
            for idx, row in fg_summary.iterrows():
                shr = row['Subdist HR']
                p = row['p-value']
                ci_low = row['Lower 95%']
                ci_high = row['Upper 95%']
                ci_str = _format_ci(ci_low, ci_high, style)
                p_str = _format_p(p, style)
                if p < 0.05:
                    interp = _interpret_shr(shr)
                    text += f"* **{idx}** was {interp} (SHR {shr:.2f}, 95% CI {ci_str}, {p_str}).\n"
                else:
                    text += f"* **{idx}**: SHR {shr:.2f}, 95% CI {ci_str}, {p_str} (not statistically significant).\n"

    # Multivariable Fine-Gray (adjusted SHRs)
    if fg_mv_summary is not None and len(fg_mv_summary) > 0:
        text += "\n\n"
        if concise:
            text += "**Multivariable Fine-Gray (adjusted SHRs):**\n"
            for _, row in fg_mv_summary.iterrows():
                shr = row['aSHR']
                p = row['p-value']
                ci_low = row['Lower 95%']
                ci_high = row['Upper 95%']
                ci_str = _format_ci(ci_low, ci_high, style)
                p_str = _format_p(p, style)
                text += f"* **{row['Variable']}**: aSHR {shr:.2f}, 95% CI {ci_str}, {p_str}\n"
        else:
            text += (
                "**Multivariable Fine-Gray regression** "
                "(adjusted subdistribution hazard model):\n"
            )
            # Separate significant and non-significant
            sig_rows = fg_mv_summary[fg_mv_summary['p-value'] < 0.05]
            nonsig_rows = fg_mv_summary[fg_mv_summary['p-value'] >= 0.05]

            if len(sig_rows) > 0:
                text += "\nIndependent predictors of cumulative incidence:\n"
                for _, row in sig_rows.iterrows():
                    shr = row['aSHR']
                    p = row['p-value']
                    ci_str = _format_ci(row['Lower 95%'], row['Upper 95%'], style)
                    p_str = _format_p(p, style)
                    interp = _interpret_shr(shr)
                    text += (
                        f"* **{row['Variable']}** was independently {interp} "
                        f"(aSHR {shr:.2f}, 95% CI {ci_str}, {p_str}).\n"
                    )

            if len(nonsig_rows) > 0:
                ns_names = [f"**{r['Variable']}**" for _, r in nonsig_rows.iterrows()]
                if len(ns_names) == 1:
                    text += f"\n{ns_names[0]} was not independently associated with {event_label} "
                else:
                    text += f"\n{', '.join(ns_names[:-1])} and {ns_names[-1]} were not independently associated with {event_label} "
                text += "after adjustment for other covariates.\n"

    return text


# ============================================================
# Diagnostic (2x2) Narrator
# ============================================================

def generate_diagnostic_narrative(res, style_name="Standard", detail_level="Detailed"):
    """
    Generate a narrative for diagnostic accuracy analysis (2x2 table).

    Parameters
    ----------
    res : dict
        Dictionary containing: test_var, ref_var, n_total,
        sens, sens_l, sens_h, spec, spec_l, spec_h,
        ppv, ppv_l, ppv_h, npv, npv_l, npv_h, p_val.
    style_name : str
        Journal style name.
    detail_level : str
        "Concise" or "Detailed".

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    concise = detail_level == "Concise"

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    if concise:
        text += (
            f"Diagnostic performance of **{res['test_var']}** was evaluated against "
            f"**{res['ref_var']}** (N={res['n_total']}). "
            "Sensitivity, specificity, PPV, and NPV were calculated with Wilson 95% CIs."
        )
    else:
        text += (
            f"Diagnostic performance of **{res['test_var']}** was evaluated against "
            f"**{res['ref_var']}** as the reference standard (N={res['n_total']}). "
            "Sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV) "
            "were calculated from a 2\u00d72 contingency table. "
            "95% confidence intervals were estimated using the Wilson score method. "
            "Association was assessed using the Pearson chi-square test."
        )
    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Lead with the clinical finding
    p_str = _format_p(res['p_val'], style)
    if res['p_val'] < 0.05:
        text += f"**{res['test_var']}** showed a statistically significant association with **{res['ref_var']}** ({p_str}). "
    else:
        text += f"**{res['test_var']}** was not significantly associated with **{res['ref_var']}** ({p_str}). "

    # Performance metrics
    text += (
        f"Sensitivity {res['sens']:.1%} "
        f"(95% CI {res['sens_l']:.1%}{style.get('ci_sep', '-')}{res['sens_h']:.1%}), "
        f"specificity {res['spec']:.1%} "
        f"(95% CI {res['spec_l']:.1%}{style.get('ci_sep', '-')}{res['spec_h']:.1%})"
    )

    if concise:
        text += (
            f", PPV {res['ppv']:.1%} "
            f"({res['ppv_l']:.1%}{style.get('ci_sep', '-')}{res['ppv_h']:.1%}), "
            f"NPV {res['npv']:.1%} "
            f"({res['npv_l']:.1%}{style.get('ci_sep', '-')}{res['npv_h']:.1%})."
        )
    else:
        text += ". "
        # Interpret sensitivity/specificity
        if res['sens'] >= 0.9 and res['spec'] >= 0.9:
            text += "The test demonstrated excellent overall diagnostic accuracy. "
        elif res['sens'] >= 0.9:
            text += "The test was highly sensitive (good for ruling out) but had limited specificity. "
        elif res['spec'] >= 0.9:
            text += "The test was highly specific (good for ruling in) but had limited sensitivity. "
        elif res['sens'] >= 0.7 and res['spec'] >= 0.7:
            text += "The test showed moderate diagnostic performance. "
        else:
            text += "The test showed limited diagnostic utility in this population. "

        text += (
            f"Predictive values were: PPV {res['ppv']:.1%} "
            f"({res['ppv_l']:.1%}{style.get('ci_sep', '-')}{res['ppv_h']:.1%}), "
            f"NPV {res['npv']:.1%} "
            f"({res['npv_l']:.1%}{style.get('ci_sep', '-')}{res['npv_h']:.1%})."
        )

    return text


# ============================================================
# Prognostic (C-Index) Narrator
# ============================================================

def generate_prognostic_narrative(res_list, style_name="Standard", detail_level="Detailed",
                                  n_bootstrap=50):
    """
    Generate a narrative for prognostic model comparison (C-Index).

    Parameters
    ----------
    res_list : list of dict
        Each dict has 'Label', 'C-Index', 'Lower', 'Upper', 'Vars'.
    style_name : str
        Journal style name.
    detail_level : str
        "Concise" or "Detailed".
    n_bootstrap : int
        Number of bootstrap iterations used.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    concise = detail_level == "Concise"

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    if concise:
        text += (
            f"Harrell's C-index from Cox models was used to compare {len(res_list)} "
            f"prognostic models (bootstrap 95% CIs, n={n_bootstrap})."
        )
    else:
        text += (
            "Discriminative ability was assessed using Harrell's concordance index (C-index) "
            "derived from Cox proportional hazards models. 95% confidence intervals were estimated "
            f"using bootstrap resampling (n={n_bootstrap}, normal approximation)."
        )
    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Find models
    r_a = next((r for r in res_list if r["Label"] == "Model A"), None)
    r_b = next((r for r in res_list if r["Label"] == "Model B"), None)
    r_c = next((r for r in res_list if r["Label"] == "Model C"), None)

    # Report each model
    for r in res_list:
        ci_str = _format_ci(r['Lower'], r['Upper'], style)
        if concise:
            text += f"* **{r['Label']}** ({r.get('Vars', 'N/A')}): C-index {r['C-Index']:.3f} {ci_str}\n"
        else:
            interp = _interpret_c_index(r['C-Index'])
            text += f"* **{r['Label']}** ({r.get('Vars', 'N/A')}): C-index {r['C-Index']:.3f} {ci_str} \u2014 {interp}.\n"

    if not concise:
        # Deltas with interpretation
        if r_a and r_b:
            delta = r_b["C-Index"] - r_a["C-Index"]
            direction = "improved" if delta > 0 else "decreased"
            magnitude = _interpret_delta(abs(delta))
            text += f"\nAdding covariates from Model A to B {direction} discrimination by {abs(delta):.3f} ({magnitude}). "

        if r_b and r_c:
            delta = r_c["C-Index"] - r_b["C-Index"]
            direction = "improved" if delta > 0 else "decreased"
            magnitude = _interpret_delta(abs(delta))
            text += f"Further addition (Model B to C) {direction} discrimination by {abs(delta):.3f} ({magnitude}). "

    # Overall recommendation
    best = max(res_list, key=lambda x: x["C-Index"])
    text += f"\n\nBest model: **{best['Label']}** (C-index {best['C-Index']:.3f})."

    return text


def _interpret_c_index(c):
    """Interpret C-index value."""
    if c >= 0.9:
        return "outstanding discrimination"
    elif c >= 0.8:
        return "excellent discrimination"
    elif c >= 0.7:
        return "good discrimination"
    elif c >= 0.6:
        return "moderate discrimination"
    else:
        return "poor discrimination"


def _interpret_delta(delta):
    """Interpret the magnitude of C-index change."""
    if delta >= 0.05:
        return "clinically meaningful improvement"
    elif delta >= 0.02:
        return "modest improvement"
    elif delta >= 0.01:
        return "marginal improvement"
    else:
        return "negligible difference"


# ============================================================
# Available journal styles (for UI dropdowns)
# ============================================================

def get_available_styles():
    """Return list of style names for UI dropdowns."""
    return list(JOURNAL_STYLES.keys())


def get_style_labels():
    """Return dict of style_name -> display_label for UI."""
    return {k: v["label"] for k, v in JOURNAL_STYLES.items()}


def format_p_value(p, style_name="Standard", context="text"):
    """
    Format a p-value according to the selected journal style.

    Public API for use by app.py in plots, tables, and inline text.

    Parameters
    ----------
    p : float
        The p-value.
    style_name : str
        Journal style name (e.g. "Standard", "JCO", "Blood").
    context : str
        "text" — for Streamlit text / markdown (uses *italic* for JCO/Blood P).
        "plot" — for matplotlib text (uses $\\it{P}$ mathtext for italic).
        "table" — for table display (plain text, italic not applicable).

    Returns
    -------
    str
        Formatted p-value string.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])
    base = style["p_format"](p)

    if style.get("p_italic") and context == "text":
        # Markdown italic: replace leading "P" with "*P*"
        if base.startswith("P"):
            base = "*P*" + base[1:]
    elif style.get("p_italic") and context == "plot":
        # Matplotlib mathtext italic
        if base.startswith("P"):
            base = "$\\it{P}$" + base[1:]

    return base
