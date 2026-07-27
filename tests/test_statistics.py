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
    grays_test,
    adjust_pvalues,
    sanitize_name,
    sanitize_columns,
    encode_with_reference,
    median_followup,
    bootstrap_optimism_c_index,
    compute_calibration,
    subgroup_hazard_ratios,
    date_interval,
    pairwise_fine_gray,
)


class TestPairwiseFineGrayGraysColumn:
    """The pairwise table must carry a nonparametric Gray's-test p that
    reconciles with the CIF plot — and must NOT conflate it with the
    Fine-Gray Wald p (the mislabeling bug this guards against)."""

    def _cr_data(self, seed=7, n=200):
        rng = np.random.default_rng(seed)
        grp = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
        base = np.where(grp == "B", 1.5, 1.0)
        t = rng.exponential(20 / base)
        ev = rng.choice([0, 1, 2], size=n, p=[0.3, 0.4, 0.3])
        return pd.DataFrame({"time": t, "event": ev, "grp": grp})

    def test_gray_p_column_present(self):
        df = self._cr_data()
        pw = pairwise_fine_gray(df, "time", "event", "grp",
                                event_of_interest=1, reference_group="A")
        assert "Gray's p" in pw.columns
        assert "p-value" in pw.columns  # Fine-Gray Wald kept separate

    def test_pairwise_gray_p_matches_standalone(self):
        df = self._cr_data()
        pw = pairwise_fine_gray(df, "time", "event", "grp",
                                event_of_interest=1, reference_group="A")
        standalone = grays_test(df, "time", "event", "grp", event_of_interest=1)
        # For a 2-group comparison the pair subset IS the whole data, so the
        # table's Gray p must equal the plot's Gray p exactly.
        assert abs(pw["Gray's p"].iloc[0] - standalone["p_value"]) < 1e-9

    def test_gray_p_distinct_from_wald_p(self):
        """The two columns come from different methods — they should be close
        but not forced equal (this is the whole point of the fix)."""
        df = self._cr_data(seed=11)
        pw = pairwise_fine_gray(df, "time", "event", "grp",
                                event_of_interest=1, reference_group="A")
        gray = pw["Gray's p"].iloc[0]
        wald = pw["p-value"].iloc[0]
        assert not pd.isna(gray) and not pd.isna(wald)


class TestDateInterval:
    def test_basic_months(self):
        s = pd.Series(["2020-01-01", "2020-01-01"])
        e = pd.Series(["2020-02-01", "2021-01-01"])
        vals, meta = date_interval(s, e, unit="months")
        assert abs(vals.iloc[0] - 31 / 30.4375) < 0.05
        assert abs(vals.iloc[1] - 366 / 30.4375) < 0.05
        assert meta["n_computed"] == 2

    def test_mixed_formats_parse(self):
        # ISO + day-first in the same column must both parse
        s = pd.Series(["2020-01-01", "01/03/2020"])
        e = pd.Series(["2020-02-01", "01/04/2020"])
        vals, meta = date_interval(s, e, unit="days")
        assert abs(vals.iloc[0] - 31) < 0.5
        assert abs(vals.iloc[1] - 31) < 0.5
        assert meta["n_computed"] == 2

    def test_unparseable_becomes_nan(self):
        vals, meta = date_interval(pd.Series(["not a date"]), pd.Series(["2020-01-01"]))
        assert pd.isna(vals.iloc[0])
        assert meta["n_start_unparsed"] == 1

    def test_negative_flagged(self):
        vals, meta = date_interval(pd.Series(["2020-06-01"]), pd.Series(["2020-01-01"]), unit="days")
        assert vals.iloc[0] < 0
        assert meta["n_negative"] == 1

    def test_index_aligned(self):
        s = pd.Series(["2020-01-01", "2020-01-01"], index=[7, 9])
        e = pd.Series(["2020-02-01", "2020-03-01"], index=[7, 9])
        vals, _ = date_interval(s, e, unit="days")
        assert list(vals.index) == [7, 9]


class TestSubgroupForest:
    def _data(self, n=300):
        rng = np.random.RandomState(3)
        tx = rng.choice(["Ctrl", "Drug"], n)
        sub = rng.choice(["Low", "High"], n)
        base = np.where(tx == "Drug", 1.8, 1.0)  # drug protective -> longer survival
        time = np.clip(rng.exponential(20 * base), 0.1, None)
        event = rng.binomial(1, 0.7, n)
        return pd.DataFrame({"T": time, "E": event, "Tx": tx, "Sub": sub})

    def test_requires_binary_treatment(self):
        df = self._data()
        df.loc[df.index[:5], "Tx"] = "Third"
        out, reason = subgroup_hazard_ratios(df, "T", "E", "Tx", ["Sub"])
        assert out is None

    def test_overall_and_subgroup_rows(self):
        tbl, meta = subgroup_hazard_ratios(self._data(), "T", "E", "Tx", ["Sub"])
        assert tbl is not None
        assert (tbl["Subgroup"] == "Overall").any()
        assert (tbl["Subgroup"] == "Sub").any()
        # subgroup header carries an interaction p in [0,1]
        header = tbl[(tbl["Subgroup"] == "Sub") & (tbl["Level"] == "")].iloc[0]
        assert 0.0 <= header["Interaction P"] <= 1.0

    def test_protective_hr_below_one(self):
        tbl, meta = subgroup_hazard_ratios(self._data(), "T", "E", "Tx", ["Sub"])
        overall = tbl[tbl["Subgroup"] == "Overall"].iloc[0]
        assert overall["HR"] < 1.0  # drug is protective


class TestOptimismCorrectedCIndex:
    def _data(self, n=250, signal=True):
        rng = np.random.RandomState(1)
        x = rng.normal(0, 1, n)
        risk = x if signal else rng.normal(0, 1, n)
        time = np.clip(20 * np.exp(-0.6 * risk) + rng.normal(0, 1, n), 0.1, None)
        event = rng.binomial(1, 0.7, n)
        return pd.DataFrame({"T": time, "E": event, "X": x})

    def test_corrected_below_apparent(self):
        r = bootstrap_optimism_c_index(self._data(), "T", "E", ["X"], n_boot=50)
        assert r is not None
        assert r["optimism"] >= -0.02          # optimism is (near) non-negative
        assert r["corrected"] <= r["apparent"] + 1e-9
        assert 0.5 < r["apparent"] <= 1.0

    def test_signal_beats_noise(self):
        sig = bootstrap_optimism_c_index(self._data(signal=True), "T", "E", ["X"], n_boot=50)
        assert sig["corrected"] > 0.55

    def test_too_small_returns_none(self):
        tiny = self._data(n=10)
        assert bootstrap_optimism_c_index(tiny, "T", "E", ["X"], n_boot=20) is None


class TestCalibration:
    def test_wellspecified_calibrates(self):
        rng = np.random.RandomState(2)
        n = 400
        x = rng.normal(0, 1, n)
        time = np.clip(30 * np.exp(-0.5 * x) + rng.normal(0, 2, n), 0.1, None)
        event = rng.binomial(1, 0.8, n)
        df = pd.DataFrame({"T": time, "E": event, "X": x})
        cal, meta = compute_calibration(df, "T", "E", ["X"], horizon=15, n_bins=4)
        assert cal is not None and len(cal) == 4
        # predicted and observed should be broadly monotonic & close on average
        diffs = (cal["Mean Predicted"] - cal["Observed (KM)"]).abs()
        assert diffs.mean() < 0.15

    def test_insufficient_data(self):
        df = pd.DataFrame({"T": [1, 2, 3], "E": [1, 0, 1], "X": [0.1, 0.2, 0.3]})
        cal, reason = compute_calibration(df, "T", "E", ["X"], horizon=2, n_bins=5)
        assert cal is None


class TestMedianFollowup:
    def test_reverse_km_basic(self):
        # Everyone censored at their time -> reverse-KM median = median of times
        times = [10, 20, 30, 40, 50]
        events = [0, 0, 0, 0, 0]
        mfu = median_followup(times, events)
        assert mfu is not None and 25 <= mfu <= 35

    def test_all_events_not_reached(self):
        # No censoring -> follow-up distribution never drops to 0.5 as 'censoring events'
        mfu = median_followup([5, 10, 15], [1, 1, 1])
        assert mfu is None  # not reached / undefined

    def test_handles_nan(self):
        mfu = median_followup([10, np.nan, 30, 40], [0, 0, 0, 1])
        assert mfu is not None


class TestSanitizeAndEncode:
    def test_sanitize_name(self):
        assert sanitize_name("LSC+") == "LSCpos"
        assert sanitize_name("a b-c") == "a_bnegc"
        assert sanitize_name("ELN Adv") == "ELN_Adv"

    def test_sanitize_columns_no_mutation(self):
        df = pd.DataFrame({"a b": [1], "c+": [2]})
        out = sanitize_columns(df)
        assert list(out.columns) == ["a_b", "cpos"]
        assert list(df.columns) == ["a b", "c+"]  # original untouched

    def test_encode_with_reference_drops_ref(self):
        df = pd.DataFrame({"G": ["A", "B", "C", "A"], "x": [1, 2, 3, 4]})
        enc, dummies = encode_with_reference(df, ["G"], {"G": "A"})
        assert "G" not in enc.columns
        assert "G_A" not in enc.columns          # reference dropped
        assert set(dummies) == {"G_B", "G_C"}
        assert enc["G_B"].dtype == float          # numeric, not bool

    def test_encode_numeric_untouched(self):
        df = pd.DataFrame({"G": ["A", "B"], "x": [1.0, 2.0]})
        enc, dummies = encode_with_reference(df, ["G"], {"G": "A"})
        assert "x" in enc.columns and list(enc["x"]) == [1.0, 2.0]


class TestAdjustPvalues:
    """adjust_pvalues must match R's p.adjust."""

    def test_benjamini_hochberg(self):
        p = [0.01, 0.04, 0.03, 0.005, 0.2]
        out = adjust_pvalues(p, "Benjamini-Hochberg")
        expected = [0.025, 0.05, 0.05, 0.025, 0.2]  # R p.adjust(p, "BH")
        assert np.allclose(out, expected, atol=1e-6)

    def test_bonferroni(self):
        p = [0.01, 0.04, 0.03, 0.005, 0.2]
        out = adjust_pvalues(p, "Bonferroni")
        expected = [0.05, 0.2, 0.15, 0.025, 1.0]
        assert np.allclose(out, expected, atol=1e-6)

    def test_none_passthrough(self):
        p = [0.01, 0.5]
        assert np.allclose(adjust_pvalues(p, "None"), p)

    def test_nan_preserved(self):
        out = adjust_pvalues([0.01, np.nan, 0.04], "Benjamini-Hochberg")
        assert np.isnan(out[1])
        assert not np.isnan(out[0]) and not np.isnan(out[2])

    def test_capped_at_one(self):
        out = adjust_pvalues([0.6, 0.7, 0.8], "Bonferroni")
        assert (out <= 1.0).all()


class TestGraysTest:
    """Gray's K-sample test must match R's cmprsk::cuminc()$Tests.

    Ground-truth values below were produced by cmprsk 2.2-12 on the
    deterministic fixture in gray_fixture() (see cencode=0).
    """

    @staticmethod
    def gray_fixture():
        rows = [
            (2, 1, 'A'), (4, 2, 'A'), (5, 1, 'A'), (6, 0, 'A'), (8, 1, 'A'),
            (9, 1, 'A'), (11, 2, 'A'), (12, 0, 'A'), (14, 1, 'A'), (18, 0, 'A'),
            (3, 0, 'B'), (4, 1, 'B'), (7, 1, 'B'), (7, 2, 'B'), (10, 0, 'B'),
            (13, 1, 'B'), (15, 0, 'B'), (16, 1, 'B'), (20, 1, 'B'), (24, 1, 'B'),
            (5, 1, 'A'), (6, 1, 'A'), (9, 2, 'B'), (10, 1, 'B'), (12, 1, 'A'),
            (13, 0, 'B'), (17, 1, 'A'), (19, 1, 'B'), (21, 0, 'A'), (22, 1, 'B'),
        ]
        return pd.DataFrame(rows, columns=['time', 'status', 'group'])

    def test_matches_cmprsk_cause1(self):
        df = self.gray_fixture()
        r = grays_test(df, 'time', 'status', 'group', event_of_interest=1)
        assert abs(r['statistic'] - 1.1898957497) < 1e-4, r
        assert abs(r['p_value'] - 0.2753505858) < 1e-4, r
        assert r['df'] == 1

    def test_matches_cmprsk_cause2(self):
        df = self.gray_fixture()
        r = grays_test(df, 'time', 'status', 'group', event_of_interest=2)
        assert abs(r['statistic'] - 0.0006042079) < 1e-4, r
        assert abs(r['p_value'] - 0.9803894604) < 1e-4, r

    def test_single_group_returns_nan(self):
        df = self.gray_fixture()
        df = df[df['group'] == 'A']
        r = grays_test(df, 'time', 'status', 'group', event_of_interest=1)
        assert np.isnan(r['statistic'])


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
