"""
Tests for the EasySurv statistical functions.
Validates core computations against known results to ensure
clinical correctness.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.statistics import (
    calculate_wilson_ci,
    check_epv,
    get_correlation_matrix,
    calculate_vif,
    check_collinearity,
    check_separation,
    compute_fine_gray_weights,
    get_c_index_bootstrap,
    summarize_model_risk,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def simple_survival_df():
    """Simple survival dataset with known properties."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "Time": np.random.exponential(scale=20, size=n),
        "Event": np.random.binomial(1, 0.6, size=n),
        "Group": np.random.choice(["A", "B"], size=n),
        "Age": np.random.normal(60, 10, size=n),
        "Score": np.random.uniform(0, 1, size=n),
    })


@pytest.fixture
def competing_risks_df():
    """Dataset for competing risks analysis (events 0, 1, 2)."""
    np.random.seed(123)
    n = 80
    return pd.DataFrame({
        "Time": np.random.exponential(scale=15, size=n),
        "Event": np.random.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3]),
        "Group": np.random.choice(["X", "Y"], size=n),
    })


@pytest.fixture
def collinear_df():
    """Dataset with known collinearity."""
    np.random.seed(7)
    n = 100
    x1 = np.random.normal(0, 1, n)
    x2 = x1 + np.random.normal(0, 0.01, n)  # Nearly identical to x1
    x3 = np.random.normal(0, 1, n)           # Independent
    return pd.DataFrame({
        "Time": np.abs(np.random.normal(10, 5, n)) + 0.1,
        "Event": np.random.binomial(1, 0.5, n),
        "X1": x1,
        "X2": x2,
        "X3": x3,
    })


# ============================================================
# Wilson CI Tests
# ============================================================

class TestWilsonCI:
    def test_known_values(self):
        """Wilson CI for 7/10 successes should give known bounds."""
        lower, upper = calculate_wilson_ci(7, 10, alpha=0.95)
        # Known Wilson interval for 7/10 at 95%: approx (0.3968, 0.8922)
        assert 0.35 < lower < 0.45, f"Lower bound {lower} out of expected range"
        assert 0.85 < upper < 0.95, f"Upper bound {upper} out of expected range"

    def test_zero_denominator(self):
        """n=0 should return (0, 0) without error."""
        lower, upper = calculate_wilson_ci(0, 0)
        assert lower == 0.0
        assert upper == 0.0

    def test_perfect_proportion(self):
        """10/10 should return CI near (0.72, 1.0)."""
        lower, upper = calculate_wilson_ci(10, 10)
        assert lower > 0.65
        assert upper <= 1.0

    def test_zero_proportion(self):
        """0/10 should return CI near (0.0, 0.28)."""
        lower, upper = calculate_wilson_ci(0, 10)
        assert lower >= 0.0
        assert upper < 0.35

    def test_bounds_ordering(self):
        """Lower bound should always be <= upper bound."""
        for k in range(0, 11):
            lower, upper = calculate_wilson_ci(k, 10)
            assert lower <= upper, f"Bounds inverted for {k}/10: {lower} > {upper}"

    def test_large_sample(self):
        """Large sample should give tight CI around the proportion."""
        lower, upper = calculate_wilson_ci(500, 1000)
        assert 0.46 < lower < 0.50
        assert 0.50 < upper < 0.54


# ============================================================
# EPV Tests
# ============================================================

class TestCheckEPV:
    def test_robust_epv(self, simple_survival_df):
        """With ~60 events and 1-2 covariates, EPV should be green."""
        result = check_epv(simple_survival_df, "Event", ["Age"])
        assert result["status"] == "green"
        assert result["value"] > 15

    def test_low_epv(self, simple_survival_df):
        """With many covariates relative to events, EPV should be yellow/red."""
        # Create many fake covariates
        df = simple_survival_df.copy()
        for i in range(20):
            df[f"Var_{i}"] = np.random.choice(["A", "B", "C"], len(df))
        covariates = [f"Var_{i}" for i in range(20)]
        result = check_epv(df, "Event", covariates)
        assert result["status"] in ("yellow", "red")
        assert result["value"] < 15

    def test_no_covariates(self, simple_survival_df):
        """Empty covariate list should return green with inf EPV."""
        result = check_epv(simple_survival_df, "Event", [])
        assert result["status"] == "green"
        assert result["value"] == float("inf")

    def test_categorical_dof(self, simple_survival_df):
        """Categorical variable should count (levels-1) degrees of freedom."""
        # Group has 2 levels -> 1 DoF
        # Age is continuous -> 1 DoF
        result = check_epv(simple_survival_df, "Event", ["Group", "Age"])
        n_events = simple_survival_df["Event"].sum()
        expected_epv = n_events / 2  # 2 total DoF
        assert abs(result["value"] - expected_epv) < 1.0


# ============================================================
# Correlation & VIF Tests
# ============================================================

class TestCorrelation:
    def test_correlation_matrix_shape(self, simple_survival_df):
        """Correlation matrix should be square with correct dimensions."""
        mat = get_correlation_matrix(simple_survival_df, ["Age", "Score"])
        assert mat is not None
        assert mat.shape[0] == mat.shape[1]

    def test_correlation_single_variable(self, simple_survival_df):
        """Single variable should return None."""
        mat = get_correlation_matrix(simple_survival_df, ["Age"])
        assert mat is None

    def test_detect_collinearity(self, collinear_df):
        """Nearly identical variables should be flagged as collinear."""
        pairs = check_collinearity(collinear_df, ["X1", "X2", "X3"], threshold=0.7)
        # X1 and X2 should be found
        found = any(
            ("X1" in p[0] and "X2" in p[1]) or ("X2" in p[0] and "X1" in p[1])
            for p in pairs
        )
        assert found, f"Expected X1-X2 collinearity, got: {pairs}"

    def test_no_collinearity(self, simple_survival_df):
        """Independent variables should not be flagged."""
        pairs = check_collinearity(simple_survival_df, ["Age", "Score"], threshold=0.9)
        assert len(pairs) == 0

    def test_vif_high_collinearity(self, collinear_df):
        """VIF should be very high for collinear variables."""
        vif_df = calculate_vif(collinear_df, ["X1", "X2", "X3"])
        assert vif_df is not None
        max_vif = vif_df["VIF"].max()
        assert max_vif > 10, f"Expected high VIF, got {max_vif}"

    def test_vif_single_variable(self, simple_survival_df):
        """Single variable should return None for VIF."""
        assert calculate_vif(simple_survival_df, ["Age"]) is None


# ============================================================
# Fine-Gray Weights Tests
# ============================================================

class TestFineGray:
    def test_output_has_required_columns(self, competing_risks_df):
        """Fine-Gray weighted dataset should have start/stop/status/weight."""
        result = compute_fine_gray_weights(
            competing_risks_df, "Time", "Event", event_of_interest=1
        )
        for col in ["start", "stop", "status", "weight"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_weights_are_positive(self, competing_risks_df):
        """All weights should be positive."""
        result = compute_fine_gray_weights(
            competing_risks_df, "Time", "Event", event_of_interest=1
        )
        assert (result["weight"] > 0).all()

    def test_event_subjects_weight_one(self, competing_risks_df):
        """Subjects with event of interest should have weight=1 in their main row."""
        result = compute_fine_gray_weights(
            competing_risks_df, "Time", "Event", event_of_interest=1
        )
        # For event subjects, the first row (start=0) should have weight=1
        event_first_rows = result[(result["start"] == 0) & (result["status"] == 1)]
        assert (event_first_rows["weight"] == 1.0).all()

    def test_output_is_not_empty(self, competing_risks_df):
        """Result should not be empty."""
        result = compute_fine_gray_weights(
            competing_risks_df, "Time", "Event", event_of_interest=1
        )
        assert len(result) > 0


# ============================================================
# C-Index Bootstrap Tests
# ============================================================

class TestCIndexBootstrap:
    def test_returns_dict(self, simple_survival_df):
        """Should return a dict with expected keys."""
        result = get_c_index_bootstrap(
            simple_survival_df, "Time", "Event", ["Age"], label="Test", n_boot=10
        )
        assert result is not None
        assert "C-Index" in result
        assert "Lower" in result
        assert "Upper" in result
        assert "Label" in result

    def test_c_index_range(self, simple_survival_df):
        """C-Index should be between 0 and 1."""
        result = get_c_index_bootstrap(
            simple_survival_df, "Time", "Event", ["Age"], n_boot=10
        )
        assert 0.0 <= result["C-Index"] <= 1.0
        assert 0.0 <= result["Lower"] <= 1.0
        assert 0.0 <= result["Upper"] <= 1.0

    def test_ci_ordering(self, simple_survival_df):
        """Lower bound should be <= point estimate <= upper bound."""
        result = get_c_index_bootstrap(
            simple_survival_df, "Time", "Event", ["Age"], n_boot=20
        )
        assert result["Lower"] <= result["C-Index"] <= result["Upper"]

    def test_empty_covariates(self, simple_survival_df):
        """Empty covariate list should return None."""
        result = get_c_index_bootstrap(
            simple_survival_df, "Time", "Event", [], n_boot=10
        )
        assert result is None


# ============================================================
# Separation Check Tests
# ============================================================

class TestSeparationCheck:
    def test_normal_model_no_warnings(self, simple_survival_df):
        """A well-behaved model should produce no separation warnings."""
        from lifelines import CoxPHFitter

        df = simple_survival_df[["Time", "Event", "Age"]].dropna()
        cph = CoxPHFitter()
        cph.fit(df, duration_col="Time", event_col="Event")
        warnings = check_separation(cph)
        assert len(warnings) == 0


# ============================================================
# Model Risk Summary Tests
# ============================================================

class TestSummarizeModelRisk:
    def test_all_green(self):
        """Clean inputs should produce green status."""
        epv = {"status": "green", "message": "EPV=30", "value": 30}
        result = summarize_model_risk(epv, [], None, [])
        assert result["status"] == "green"
        assert result["label"] == "Robust"

    def test_red_on_separation(self):
        """Separation warnings should force red."""
        epv = {"status": "green", "message": "EPV=30", "value": 30}
        result = summarize_model_risk(epv, [], None, ["Separation detected"])
        assert result["status"] == "red"
        assert result["label"] == "High Risk"

    def test_yellow_on_low_epv(self):
        """EPV between 10-15 should give yellow."""
        epv = {"status": "yellow", "message": "EPV=12", "value": 12}
        result = summarize_model_risk(epv, [], None, [])
        assert result["status"] == "yellow"
        assert result["label"] == "Caution"

    def test_red_on_critical_epv(self):
        """EPV < 10 should give red."""
        epv = {"status": "red", "message": "EPV=5", "value": 5}
        result = summarize_model_risk(epv, [], None, [])
        assert result["status"] == "red"

    def test_high_vif_triggers_red(self):
        """VIF > 10 should trigger red status."""
        epv = {"status": "green", "message": "EPV=30", "value": 30}
        vif_df = pd.DataFrame({"Feature": ["X1"], "VIF": [15.0]})
        result = summarize_model_risk(epv, [], vif_df, [])
        assert result["status"] == "red"
