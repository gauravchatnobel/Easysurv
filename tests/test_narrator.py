"""
Tests for the AI Narrator module.

Run with: python -m pytest tests/test_narrator.py -v
"""

import sys
import os
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.narrator import (
    _interpret_hr,
    _interpret_shr,
    _interpret_c_index,
    _interpret_delta,
    generate_univariable_narrative,
    generate_multivariable_narrative,
    generate_cif_narrative,
    generate_diagnostic_narrative,
    generate_prognostic_narrative,
    get_available_styles,
    get_style_labels,
)


# ============================================================
# HR Interpretation Tests
# ============================================================

class TestInterpretHR:
    def test_high_risk(self):
        result = _interpret_hr(3.0)
        assert "3.0-fold increased risk" in result

    def test_very_high_risk(self):
        result = _interpret_hr(7.0)
        assert "markedly elevated" in result

    def test_moderate_risk(self):
        result = _interpret_hr(1.6)
        assert "60%" in result
        assert "increased" in result

    def test_modest_risk(self):
        result = _interpret_hr(1.2)
        assert "modestly" in result

    def test_neutral(self):
        result = _interpret_hr(1.0)
        assert "not meaningfully" in result

    def test_reduced_risk(self):
        result = _interpret_hr(0.7)
        assert "30%" in result
        assert "reduced" in result

    def test_substantial_reduction(self):
        result = _interpret_hr(0.5)
        assert "substantial" in result or "50%" in result

    def test_marked_reduction(self):
        result = _interpret_hr(0.3)
        assert "markedly reduced" in result


class TestInterpretSHR:
    def test_high_incidence(self):
        result = _interpret_shr(2.5)
        assert "2.5-fold higher" in result

    def test_neutral(self):
        result = _interpret_shr(0.95)
        assert "not meaningfully" in result

    def test_lower_incidence(self):
        result = _interpret_shr(0.6)
        assert "40%" in result
        assert "lower" in result


class TestInterpretCIndex:
    def test_outstanding(self):
        assert "outstanding" in _interpret_c_index(0.95)

    def test_excellent(self):
        assert "excellent" in _interpret_c_index(0.85)

    def test_good(self):
        assert "good" in _interpret_c_index(0.75)

    def test_moderate(self):
        assert "moderate" in _interpret_c_index(0.65)

    def test_poor(self):
        assert "poor" in _interpret_c_index(0.55)


class TestInterpretDelta:
    def test_meaningful(self):
        assert "clinically meaningful" in _interpret_delta(0.06)

    def test_modest(self):
        assert "modest" in _interpret_delta(0.03)

    def test_marginal(self):
        assert "marginal" in _interpret_delta(0.015)

    def test_negligible(self):
        assert "negligible" in _interpret_delta(0.005)


# ============================================================
# Univariable Narrator Tests
# ============================================================

class TestUnivariableNarrator:
    def test_basic_output(self):
        text = generate_univariable_narrative(
            group_col="Treatment",
            groups=["A", "B"],
            logrank_p=0.003,
        )
        assert "Kaplan-Meier" in text
        assert "log-rank" in text
        assert "p" in text
        assert "Methods" in text
        assert "Results" in text

    def test_significant_logrank(self):
        text = generate_univariable_narrative(
            group_col="Risk",
            groups=["Low", "High"],
            logrank_p=0.001,
            style_name="Standard",
        )
        assert "significantly" in text.lower()

    def test_nonsignificant_logrank(self):
        text = generate_univariable_narrative(
            group_col="Risk",
            groups=["Low", "High"],
            logrank_p=0.45,
        )
        assert "not significantly" in text.lower()

    def test_with_cox_summary(self):
        cox_df = pd.DataFrame({
            "Hazard Ratio (HR)": [2.5],
            "Lower 95% CI": [1.2],
            "Upper 95% CI": [5.3],
            "p-value": [0.01],
        }, index=["High_Risk"])
        text = generate_univariable_narrative(
            group_col="Risk",
            groups=["Low", "High"],
            logrank_p=0.01,
            cox_summary=cox_df,
        )
        assert "High_Risk" in text
        assert "2.5" in text
        assert "increased risk" in text

    def test_with_median_data(self):
        median_data = [
            {"Group": "A", "Median Survival": "24.5", "95% CI (Median)": "(18.0 - 30.0)"},
            {"Group": "B", "Median Survival": "12.3", "95% CI (Median)": "(8.0 - 16.0)"},
        ]
        text = generate_univariable_narrative(
            group_col="Treatment",
            groups=["A", "B"],
            logrank_p=0.01,
            median_data=median_data,
        )
        assert "24.5" in text
        assert "12.3" in text

    def test_nejm_style(self):
        text = generate_univariable_narrative(
            group_col="Treatment",
            groups=["A", "B"],
            logrank_p=0.003,
            style_name="NEJM",
        )
        # NEJM uses "P=" not "p="
        assert "P=" in text or "P<" in text

    def test_lancet_style(self):
        text = generate_univariable_narrative(
            group_col="Treatment",
            groups=["A", "B"],
            logrank_p=0.00003,
            style_name="Lancet",
        )
        assert "p<0.0001" in text


# ============================================================
# Multivariable Narrator Tests
# ============================================================

class TestMultivariableNarrator:
    @pytest.fixture
    def summary_df(self):
        return pd.DataFrame({
            "Hazard Ratio (HR)": [2.5, 0.7, 1.1],
            "Lower 95%": [1.2, 0.4, 0.8],
            "Upper 95%": [5.3, 1.2, 1.5],
            "p-value": [0.01, 0.2, 0.6],
        }, index=["Age_High", "Treatment_B", "Gender_M"])

    def test_basic_output(self, summary_df):
        text = generate_multivariable_narrative(summary_df)
        assert "Methods" in text
        assert "Results" in text
        assert "Cox" in text

    def test_separates_significant(self, summary_df):
        text = generate_multivariable_narrative(summary_df)
        assert "independently associated" in text
        assert "not statistically significant" in text

    def test_clinical_interpretation(self, summary_df):
        text = generate_multivariable_narrative(summary_df)
        # Age_High HR=2.5 should mention fold-increase
        assert "2.5-fold" in text

    def test_penalized_method(self, summary_df):
        text = generate_multivariable_narrative(
            summary_df,
            use_penalizer=True,
            penalizer_value=0.1,
            l1_ratio=0.0,
        )
        assert "Ridge" in text
        assert "penalized" in text.lower()

    def test_lasso_method(self, summary_df):
        text = generate_multivariable_narrative(
            summary_df,
            use_penalizer=True,
            penalizer_value=0.5,
            l1_ratio=1.0,
        )
        assert "Lasso" in text

    def test_patient_counts(self, summary_df):
        text = generate_multivariable_narrative(
            summary_df,
            n_patients=200,
            n_events=85,
        )
        assert "200" in text
        assert "85" in text


# ============================================================
# CIF Narrator Tests
# ============================================================

class TestCIFNarrator:
    def test_basic_output(self):
        text = generate_cif_narrative()
        assert "Aalen-Johansen" in text
        assert "Fine-Gray" in text

    def test_with_median_data(self):
        median_data = [
            {"Group": "X", "Median Time to Incidence": "18.0", "95% CI (Median)": "(12.0 - 24.0)"},
        ]
        text = generate_cif_narrative(cif_median_data=median_data)
        assert "18.0" in text

    def test_with_fg_summary(self):
        fg_df = pd.DataFrame({
            "Subdist HR": [1.8],
            "Lower 95%": [1.1],
            "Upper 95%": [2.9],
            "p-value": [0.02],
        }, index=["Treatment"])
        text = generate_cif_narrative(fg_summary=fg_df)
        assert "Treatment" in text
        assert "1.8" in text or "80%" in text


# ============================================================
# Diagnostic Narrator Tests
# ============================================================

class TestDiagnosticNarrator:
    @pytest.fixture
    def diag_results(self):
        return {
            "test_var": "LSC_Status",
            "ref_var": "NGS_Result",
            "n_total": 150,
            "sens": 0.85, "sens_l": 0.75, "sens_h": 0.92,
            "spec": 0.72, "spec_l": 0.60, "spec_h": 0.82,
            "ppv": 0.68, "ppv_l": 0.55, "ppv_h": 0.79,
            "npv": 0.87, "npv_l": 0.78, "npv_h": 0.93,
            "p_val": 0.001,
        }

    def test_basic_output(self, diag_results):
        text = generate_diagnostic_narrative(diag_results)
        assert "LSC_Status" in text
        assert "NGS_Result" in text
        assert "Wilson" in text

    def test_clinical_interpretation(self, diag_results):
        text = generate_diagnostic_narrative(diag_results)
        # Sens=0.85, Spec=0.72 -> "moderate diagnostic performance"
        assert "moderate" in text.lower()

    def test_high_sensitivity(self):
        res = {
            "test_var": "Test", "ref_var": "Gold", "n_total": 100,
            "sens": 0.95, "sens_l": 0.88, "sens_h": 0.99,
            "spec": 0.60, "spec_l": 0.48, "spec_h": 0.71,
            "ppv": 0.65, "ppv_l": 0.52, "ppv_h": 0.76,
            "npv": 0.94, "npv_l": 0.84, "npv_h": 0.99,
            "p_val": 0.01,
        }
        text = generate_diagnostic_narrative(res)
        assert "highly sensitive" in text.lower()
        assert "ruling out" in text.lower()

    def test_excellent_accuracy(self):
        res = {
            "test_var": "Test", "ref_var": "Gold", "n_total": 100,
            "sens": 0.95, "sens_l": 0.88, "sens_h": 0.99,
            "spec": 0.92, "spec_l": 0.84, "spec_h": 0.97,
            "ppv": 0.90, "ppv_l": 0.81, "ppv_h": 0.96,
            "npv": 0.96, "npv_l": 0.89, "npv_h": 0.99,
            "p_val": 0.0001,
        }
        text = generate_diagnostic_narrative(res)
        assert "excellent" in text.lower()


# ============================================================
# Prognostic Narrator Tests
# ============================================================

class TestPrognosticNarrator:
    @pytest.fixture
    def res_list(self):
        return [
            {"Label": "Model A", "C-Index": 0.65, "Lower": 0.58, "Upper": 0.72, "Vars": "Age"},
            {"Label": "Model B", "C-Index": 0.78, "Lower": 0.71, "Upper": 0.85, "Vars": "Age, LSC"},
        ]

    def test_basic_output(self, res_list):
        text = generate_prognostic_narrative(res_list)
        assert "C-index" in text or "concordance" in text.lower()
        assert "Model A" in text
        assert "Model B" in text

    def test_delta_interpretation(self, res_list):
        text = generate_prognostic_narrative(res_list)
        # Delta = 0.13 -> "clinically meaningful"
        assert "clinically meaningful" in text

    def test_discrimination_labels(self, res_list):
        text = generate_prognostic_narrative(res_list)
        # Model A = 0.65 -> "moderate", Model B = 0.78 -> "good"
        assert "moderate" in text
        assert "good" in text

    def test_best_model_identified(self, res_list):
        text = generate_prognostic_narrative(res_list)
        assert "best performing model was **Model B**" in text


# ============================================================
# Style System Tests
# ============================================================

class TestStyleSystem:
    def test_available_styles(self):
        styles = get_available_styles()
        assert "Standard" in styles
        assert "NEJM" in styles
        assert "Lancet" in styles

    def test_style_labels(self):
        labels = get_style_labels()
        assert "Standard" in labels
        assert isinstance(labels["Standard"], str)

    def test_all_narrators_accept_style(self):
        """Every narrator function should accept style_name without error."""
        for style in get_available_styles():
            generate_univariable_narrative("Group", ["A", "B"], 0.05, style_name=style)
            generate_diagnostic_narrative({
                "test_var": "T", "ref_var": "R", "n_total": 10,
                "sens": 0.8, "sens_l": 0.5, "sens_h": 0.9,
                "spec": 0.7, "spec_l": 0.4, "spec_h": 0.8,
                "ppv": 0.6, "ppv_l": 0.3, "ppv_h": 0.8,
                "npv": 0.8, "npv_l": 0.5, "npv_h": 0.9,
                "p_val": 0.05,
            }, style_name=style)
            generate_prognostic_narrative([
                {"Label": "Model A", "C-Index": 0.7, "Lower": 0.6, "Upper": 0.8, "Vars": "X"},
            ], style_name=style)
            generate_cif_narrative(style_name=style)
