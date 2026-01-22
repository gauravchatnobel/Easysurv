
def generate_diagnostic_narrative(res):
    """
    Generates a text report for the Diagnostic (2x2) analysis.
    res: Dictionary containing results (sens, spec, p_val, etc.)
    """
    narrative = "**Methods**\n"
    narrative += f"Diagnostic performance was evaluated using a 2x2 contingency table with **{res['ref_var']}** as the reference standard (N={res['n_total']}). "
    narrative += "Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV) were calculated. "
    narrative += "95% Confidence Intervals (CIs) were estimated using the Wilson Score method. "
    narrative += "Association between the test and reference was assessed using the Pearson Chi-Square test.\n\n"
    
    narrative += "**Results**\n"
    narrative += f"Evaluating **{res['test_var']}** against **{res['ref_var']}**:\n"
    narrative += f"- **Sensitivity**: {res['sens']:.1%} (95% CI {res['sens_l']:.1%}-{res['sens_h']:.1%}).\n"
    narrative += f"- **Specificity**: {res['spec']:.1%} (95% CI {res['spec_l']:.1%}-{res['spec_h']:.1%}).\n"
    narrative += f"- **Predictive Values**: PPV {res['ppv']:.1%} ({res['ppv_l']:.1%}-{res['ppv_h']:.1%}), NPV {res['npv']:.1%} ({res['npv_l']:.1%}-{res['npv_h']:.1%}).\n"
    narrative += f"- **Statistical Association**: Chi-square p={res['p_val']:.4f}."
    
    return narrative

def generate_prognostic_narrative(res_list):
    """
    Generates a text report for the Prognostic (C-Index) analysis.
    res_list: List of dictionaries with model results.
    """
    narrative = "**Methods**\n"
    narrative += "Discriminative power was assessed using Harrell's Concordance Index (C-Index) derived from Cox Proportional Hazards models. "
    narrative += "To estimate uncertainty, 95% Confidence Intervals (CIs) for the C-Index were calculated using a bootstrap approach with 50 resamples (Normal Approximation).\n\n"
    
    narrative += "**Results**\n"
    narrative += f"We compared {len(res_list)} proportional hazards models to evaluate incremental discriminative power.\n"
    
    # Delta Logic
    r_a = next((r for r in res_list if r["Label"] == "Model A"), None)
    r_b = next((r for r in res_list if r["Label"] == "Model B"), None)
    r_c = next((r for r in res_list if r["Label"] == "Model C"), None)
    
    delta_ab = 0
    delta_bc = 0
    if r_a and r_b: delta_ab = r_b["C-Index"] - r_a["C-Index"]
    if r_b and r_c: delta_bc = r_c["C-Index"] - r_b["C-Index"]
    
    # Find Best Model
    best_mod = max(res_list, key=lambda x: x["C-Index"])
    narrative += f"- The best performing model was **{best_mod['Label']}** with a C-Index of **{best_mod['C-Index']:.3f}** (95% CI {best_mod['Lower']:.3f}-{best_mod['Upper']:.3f}).\n"
    
    if r_a and r_b:
        narrative += f"- **Model B vs A**: Adding the selected covariates improved discrimination by **{delta_ab:+.3f} points**.\n"
    
    if r_b and r_c:
        narrative += f"- **Model C vs B**: Further addition of covariates changed discrimination by **{delta_bc:+.3f} points**.\n"
        
    return narrative
