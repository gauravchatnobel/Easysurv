"""
AI Narrator Module for EasySurv.

Generates publication-ready methods and results text from statistical outputs.
Supports multiple journal styles and provides clinical interpretation of results.
"""


# ============================================================
# Style Definitions
# ============================================================

JOURNAL_STYLES = {
    "Standard": {
        "label": "Standard (Most Journals)",
        "ci_format": "({low:.2f}-{high:.2f})",
        "p_format": lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}",
        "hr_inline": True,
        "bold_significant": True,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
    },
    "NEJM": {
        "label": "NEJM Style",
        "ci_format": "({low:.2f} to {high:.2f})",
        "p_format": lambda p: "P<0.001" if p < 0.001 else f"P={p:.2f}",
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
    },
    "Lancet": {
        "label": "Lancet Style",
        "ci_format": "({low:.2f}-{high:.2f})",
        "p_format": lambda p: "p<0.0001" if p < 0.0001 else f"p={p:.4f}",
        "hr_inline": True,
        "bold_significant": False,
        "methods_header": "**Methods**",
        "results_header": "**Results**",
    },
}


# ============================================================
# Helper Functions
# ============================================================

def _interpret_hr(hr, variable_name=""):
    """
    Translate a hazard ratio into plain clinical language.

    Examples:
        HR=2.45 -> "associated with a 2.5-fold increased risk"
        HR=0.45 -> "associated with a 55% reduced risk"
        HR=1.02 -> "not meaningfully associated with risk"
    """
    if hr > 5.0:
        return f"associated with a markedly elevated risk ({hr:.1f}-fold increase)"
    elif hr > 2.0:
        return f"associated with a {hr:.1f}-fold increased risk of the event"
    elif hr > 1.5:
        pct = (hr - 1) * 100
        return f"associated with a {pct:.0f}% increased risk of the event"
    elif hr > 1.1:
        pct = (hr - 1) * 100
        return f"associated with a modestly increased risk ({pct:.0f}% higher)"
    elif hr >= 0.9:
        return "not meaningfully associated with altered risk"
    elif hr >= 0.67:
        pct = (1 - hr) * 100
        return f"associated with a {pct:.0f}% reduced risk of the event"
    elif hr >= 0.5:
        pct = (1 - hr) * 100
        return f"associated with a substantial risk reduction ({pct:.0f}% lower)"
    else:
        fold = 1 / hr
        return f"associated with a markedly reduced risk ({fold:.1f}-fold lower)"


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
    return style["ci_format"].format(low=low, high=high)


def _format_p(p, style):
    """Format p-value according to journal style."""
    return style["p_format"](p)


def _significance_phrase(p, style):
    """Return significance phrase."""
    if p < 0.05:
        if style.get("bold_significant"):
            return "**significantly**"
        return "significantly"
    return "not significantly"


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

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    text += (
        "Survival estimates were calculated using the Kaplan-Meier method. "
        "Between-group comparisons were performed using the log-rank test. "
        f"Univariable hazard ratios (HRs) were estimated using {cox_method}. "
    )
    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Median survival (lead with the clinical finding)
    if median_data:
        med_parts = []
        for item in median_data:
            med_val = item['Median Survival']
            ci_val = item['95% CI (Median)']
            med_parts.append(f"{item['Group']}: {med_val} {ci_val}")
        text += f"Median survival by **{group_col}** was " + "; ".join(med_parts) + ". "

    # Log-rank result
    p_str = _format_p(logrank_p, style)
    sig = _significance_phrase(logrank_p, style)
    text += f"The difference between groups was {sig} ({p_str}).\n\n"

    # Cox regression results with clinical interpretation
    if cox_summary is not None and len(cox_summary) > 0:
        text += "Univariable Cox regression results:\n"
        for idx, row in cox_summary.iterrows():
            hr = row['Hazard Ratio (HR)']
            p = row['p-value']
            # Handle both 'Lower 95% CI' and 'Lower 95%' column names
            if 'Lower 95% CI' in row.index:
                ci_low, ci_high = row['Lower 95% CI'], row['Upper 95% CI']
            else:
                ci_low, ci_high = row['Lower 95%'], row['Upper 95%']

            ci_str = _format_ci(ci_low, ci_high, style)
            p_str = _format_p(p, style)
            interpretation = _interpret_hr(hr, idx)

            text += f"* **{idx}** was {interpretation} (HR {hr:.2f}, 95% CI {ci_str}, {p_str}).\n"

    # Point-in-time estimates
    if point_estimates and target_time is not None:
        text += f"\nAt {target_time} months: "
        pit_parts = []
        for item in point_estimates:
            surv_key = [k for k in item.keys() if k.startswith("Survival")][0] if any(k.startswith("Survival") for k in item) else None
            if surv_key:
                pit_parts.append(f"{item['Group']} {item[surv_key]} (95% CI {item['95% CI']})")
        text += "; ".join(pit_parts) + "."

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

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])

    # --- Methods ---
    text = f"{style['methods_header']}\n"

    if use_penalizer and penalizer_value > 0:
        if l1_ratio == 0:
            penalty_type = "Ridge"
        elif l1_ratio == 1:
            penalty_type = "Lasso"
        else:
            penalty_type = "Elastic Net"
        text += (
            f"Multivariable analysis was performed using penalized Cox regression "
            f"({penalty_type}, lambda={penalizer_value:.4f}, L1 ratio={l1_ratio:.1f}) "
            f"to account for potential multicollinearity and prevent overfitting. "
        )
    else:
        text += (
            "Multivariable analysis was performed using the Cox proportional hazards model "
            "to identify independent predictors of the time-to-event outcome. "
        )

    if n_patients and n_events:
        text += f"The analysis included {n_patients} patients with {n_events} events. "

    text += "Hazard ratios (HRs) with 95% confidence intervals (CIs) are reported.\n\n"

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

    # Report significant predictors first
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
    style_name="Standard",
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
        Fine-Gray regression summary with columns:
        'Subdist HR', 'Lower 95%', 'Upper 95%', 'p-value'.
    style_name : str
        Journal style name.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    text += (
        "Cumulative incidence functions (CIF) were estimated using the Aalen-Johansen method "
        "to account for competing risks. The effect of covariates on the cumulative incidence "
        "was assessed using the Fine-Gray subdistribution hazard model. "
    )
    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"

    # Median time to incidence (lead with this)
    if cif_median_data:
        med_parts = []
        for item in cif_median_data:
            med_val = item['Median Time to Incidence']
            ci_val = item['95% CI (Median)']
            med_parts.append(f"{item['Group']}: {med_val} {ci_val}")
        text += "Median time to incidence was " + "; ".join(med_parts) + ". "

    # Point-in-time cumulative incidence
    if cif_est_data and cif_target_time is not None:
        col_name = f"Cumulative Incidence at {cif_target_time}"
        pit_parts = []
        for item in cif_est_data:
            val = item.get(col_name, "N/A")
            ci = item.get("95% CI", "")
            pit_parts.append(f"{item['Group']} {val} (95% CI {ci})")
        text += f"At {cif_target_time} months, the cumulative incidence was: " + "; ".join(pit_parts) + ".\n\n"

    # Fine-Gray regression with clinical interpretation
    if fg_summary is not None:
        text += "**Fine-Gray regression** (subdistribution hazard model):\n"
        for idx, row in fg_summary.iterrows():
            shr = row['Subdist HR']
            p = row['p-value']
            ci_low = row['Lower 95%']
            ci_high = row['Upper 95%']
            ci_str = _format_ci(ci_low, ci_high, style)
            p_str = _format_p(p, style)
            interp = _interpret_shr(shr)

            sig = _significance_phrase(p, style)
            text += f"* **{idx}** was {sig} {interp} (SHR {shr:.2f}, 95% CI {ci_str}, {p_str}).\n"

    return text


# ============================================================
# Diagnostic (2x2) Narrator
# ============================================================

def generate_diagnostic_narrative(res, style_name="Standard"):
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

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    text += (
        f"Diagnostic performance of **{res['test_var']}** was evaluated against "
        f"**{res['ref_var']}** as the reference standard (N={res['n_total']}). "
        "Sensitivity, specificity, positive predictive value (PPV), and negative predictive value (NPV) "
        "were calculated from a 2x2 contingency table. "
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
        f"The sensitivity was {res['sens']:.1%} "
        f"(95% CI {res['sens_l']:.1%}-{res['sens_h']:.1%}) "
        f"and specificity was {res['spec']:.1%} "
        f"(95% CI {res['spec_l']:.1%}-{res['spec_h']:.1%}). "
    )

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
        f"({res['ppv_l']:.1%}-{res['ppv_h']:.1%}), "
        f"NPV {res['npv']:.1%} "
        f"({res['npv_l']:.1%}-{res['npv_h']:.1%})."
    )

    return text


# ============================================================
# Prognostic (C-Index) Narrator
# ============================================================

def generate_prognostic_narrative(res_list, style_name="Standard"):
    """
    Generate a narrative for prognostic model comparison (C-Index).

    Parameters
    ----------
    res_list : list of dict
        Each dict has 'Label', 'C-Index', 'Lower', 'Upper', 'Vars'.
    style_name : str
        Journal style name.

    Returns
    -------
    str
        Formatted narrative text.
    """
    style = JOURNAL_STYLES.get(style_name, JOURNAL_STYLES["Standard"])

    # --- Methods ---
    text = f"{style['methods_header']}\n"
    text += (
        "Discriminative ability was assessed using Harrell's concordance index (C-index) "
        "derived from Cox proportional hazards models. 95% confidence intervals were estimated "
        "using bootstrap resampling (n=50, normal approximation). "
    )
    text += "\n\n"

    # --- Results ---
    text += f"{style['results_header']}\n"
    text += f"We compared {len(res_list)} prognostic models:\n\n"

    # Find models
    r_a = next((r for r in res_list if r["Label"] == "Model A"), None)
    r_b = next((r for r in res_list if r["Label"] == "Model B"), None)
    r_c = next((r for r in res_list if r["Label"] == "Model C"), None)

    # Report each model
    for r in res_list:
        ci_str = _format_ci(r['Lower'], r['Upper'], style)
        interp = _interpret_c_index(r['C-Index'])
        text += f"* **{r['Label']}** ({r.get('Vars', 'N/A')}): C-index {r['C-Index']:.3f} {ci_str} — {interp}.\n"

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
    text += f"\n\nThe best performing model was **{best['Label']}** (C-index {best['C-Index']:.3f})."

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
