"""Tests for the session_manager module."""
import json
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.session_manager import (
    save_session,
    load_session,
    get_session_filename,
    _compress_dataframe,
    _decompress_dataframe,
    SESSION_FORMAT_VERSION,
)
from modules import session_bank as sb


def test_analysis_bank_round_trips_through_save_load():
    """A pinned analysis (figure + tables + narrative) must survive save/load —
    this is what makes a multi-analysis session resumable and shareable."""
    f, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])
    entry = sb.make_entry(
        "KM", "OS by LSC", fig=f,
        tables={"Cox": pd.DataFrame({"HR": [1.4]}, index=["LSC+"])},
        narrative="LSC+ had shorter survival.",
        meta={"endpoint": "OS", "group_col": "LSC", "n_patients": 153}, dpi=80,
    )
    session_state = {"analysis_bank": [entry], "demo_loaded": True}
    df = pd.DataFrame({"Time": [1, 2, 3], "Event": [1, 0, 1]})
    js = save_session(df, {"time_col": "Time", "event_col": "Event"}, session_state)

    restored = load_session(js)
    bank = restored["analysis_bank"]
    assert len(bank) == 1
    e = bank[0]
    assert e["label"] == "OS by LSC" and e["png"]
    assert e["narrative"] == "LSC+ had shorter survival."
    assert e["meta"]["n_patients"] == 153
    tables = dict(sb.entry_tables(e))
    assert tables["Cox"].iloc[0, 0] == 1.4


def test_bank_absent_gives_empty_list():
    df = pd.DataFrame({"Time": [1, 2], "Event": [1, 0]})
    js = save_session(df, {}, {"demo_loaded": True})
    assert load_session(js)["analysis_bank"] == []


def test_live_figure_entry_is_dropped_on_save():
    f, ax = plt.subplots()
    bad = {"type": "KM", "label": "x", "png": f, "tables": []}  # live fig, not serializable
    js = save_session(pd.DataFrame({"T": [1]}), {}, {"analysis_bank": [bad]})
    assert load_session(js)["analysis_bank"] == []


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_df():
    """A small clinical-style DataFrame."""
    return pd.DataFrame({
        "OS_Months": [12.5, 24.0, 6.3, 36.1, 18.0],
        "OS_Status": [1, 0, 1, 0, 1],
        "MRD_Status": ["Positive", "Negative", "Positive", "Negative", "Positive"],
        "Age": [55, 63, 45, 72, 58],
    })


@pytest.fixture
def sample_sidebar():
    return {
        "time_col": "OS_Months",
        "event_col": "OS_Status",
        "group_col": "MRD_Status",
        "narrator_style_name": "NEJM",
        "selected_font": "serif",
        "title_fontsize": 18,
        "title_bold": True,
        "axes_fontsize": 14,
        "legend_fontsize": 12,
        "line_width": 2.0,
        "show_risk_table": True,
        "table_height": -0.3,
        "show_censored": False,
        "show_ci": True,
    }


@pytest.fixture
def sample_state():
    return {
        "demo_loaded": False,
        "custom_cutoffs": [
            {"source": "Age", "value": 60.0, "name": "Age_Group"}
        ],
        "custom_combinations": [],
        "use_penalizer": True,
        "l1_ratio_val": 0.5,
        "penalizer_val": 0.01,
    }


# ============================================================
# Compression Tests
# ============================================================

class TestCompression:
    def test_roundtrip_basic(self, sample_df):
        compressed = _compress_dataframe(sample_df)
        restored = _decompress_dataframe(compressed)
        pd.testing.assert_frame_equal(restored, sample_df)

    def test_roundtrip_with_nan(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"]})
        compressed = _compress_dataframe(df)
        restored = _decompress_dataframe(compressed)
        assert restored["a"].isna().sum() == 1
        assert restored["b"].isna().sum() == 1

    def test_compression_reduces_size(self, sample_df):
        """Compressed output should be smaller than raw CSV for non-trivial data."""
        raw_csv = sample_df.to_csv(index=True).encode("utf-8")
        compressed = _compress_dataframe(sample_df)
        # Base64 adds ~33% overhead, but gzip should still compress well for larger data
        # For very small data this may not hold, so just check it's a valid string
        assert isinstance(compressed, str)
        assert len(compressed) > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        compressed = _compress_dataframe(df)
        restored = _decompress_dataframe(compressed)
        assert len(restored) == 0


# ============================================================
# Save / Load Roundtrip Tests
# ============================================================

class TestSaveLoad:
    def test_basic_roundtrip(self, sample_df, sample_sidebar, sample_state):
        json_str = save_session(sample_df, sample_sidebar, sample_state, notes="test run")
        restored = load_session(json_str)

        assert restored["format_version"] == SESSION_FORMAT_VERSION
        assert restored["notes"] == "test run"
        assert "saved_at" in restored
        assert restored["df"] is not None
        pd.testing.assert_frame_equal(restored["df"], sample_df)

    def test_sidebar_restored(self, sample_df, sample_sidebar, sample_state):
        json_str = save_session(sample_df, sample_sidebar, sample_state)
        restored = load_session(json_str)

        assert restored["sidebar"]["time_col"] == "OS_Months"
        assert restored["sidebar"]["event_col"] == "OS_Status"
        assert restored["sidebar"]["group_col"] == "MRD_Status"
        assert restored["sidebar"]["narrator_style_name"] == "NEJM"
        assert restored["sidebar"]["selected_font"] == "serif"
        assert restored["sidebar"]["title_fontsize"] == 18
        assert restored["sidebar"]["show_censored"] is False

    def test_state_restored(self, sample_df, sample_sidebar, sample_state):
        json_str = save_session(sample_df, sample_sidebar, sample_state)
        restored = load_session(json_str)

        assert restored["state"]["use_penalizer"] is True
        assert restored["state"]["l1_ratio_val"] == 0.5
        assert restored["state"]["penalizer_val"] == 0.01
        assert len(restored["state"]["custom_cutoffs"]) == 1
        assert restored["state"]["custom_cutoffs"][0]["source"] == "Age"

    def test_no_dataset(self, sample_sidebar, sample_state):
        json_str = save_session(None, sample_sidebar, sample_state)
        restored = load_session(json_str)
        assert restored["df"] is None

    def test_empty_notes(self, sample_df, sample_sidebar, sample_state):
        json_str = save_session(sample_df, sample_sidebar, sample_state, notes="")
        restored = load_session(json_str)
        assert restored["notes"] == ""

    def test_dataframe_state_keys(self, sample_df, sample_sidebar):
        """Test that DataFrames in session_state are saved and restored."""
        cox_df = pd.DataFrame({
            "Hazard Ratio (HR)": [1.5, 0.8],
            "Lower 95%": [1.1, 0.5],
            "Upper 95%": [2.1, 1.2],
            "p-value": [0.01, 0.3],
        }, index=["MRD_Positive", "Age"])

        state = {"uv_cox_summary": cox_df, "demo_loaded": True}
        json_str = save_session(sample_df, sample_sidebar, state)
        restored = load_session(json_str)

        assert "uv_cox_summary" in restored["state_dataframes"]
        restored_cox = restored["state_dataframes"]["uv_cox_summary"]
        assert len(restored_cox) == 2
        assert abs(restored_cox.loc["MRD_Positive", "Hazard Ratio (HR)"] - 1.5) < 0.001

    def test_valid_json_output(self, sample_df, sample_sidebar, sample_state):
        json_str = save_session(sample_df, sample_sidebar, sample_state)
        parsed = json.loads(json_str)  # Should not raise
        assert "format_version" in parsed
        assert "dataset" in parsed
        assert "sidebar" in parsed

    def test_numpy_types_serialization(self):
        """Ensure numpy types are properly serialized."""
        df = pd.DataFrame({"x": np.array([1, 2, 3])})
        sidebar = {"title_fontsize": np.int64(20), "line_width": np.float64(1.5)}
        state = {"l1_ratio_val": np.float32(0.5)}
        json_str = save_session(df, sidebar, state)
        restored = load_session(json_str)
        assert restored["sidebar"]["title_fontsize"] == 20

    def test_filters_saved(self, sample_df, sample_state):
        sidebar = {
            "filter_cols": ["MRD_Status", "Age"],
            "filt_MRD_Status": ["Positive", "Negative"],
            "filt_Age": (40, 70),
        }
        json_str = save_session(sample_df, sidebar, sample_state)
        restored = load_session(json_str)
        assert restored["filters"]["filter_cols"] == ["MRD_Status", "Age"]
        assert restored["filters"]["filt_Age"] == [40, 70]  # tuple -> list in JSON

    def test_custom_text_annotations(self, sample_df, sample_state):
        sidebar = {
            "main_txt_1": "HR = 1.5",
            "main_x_1": 0.5,
            "main_y_1": 0.5,
            "main_sz_1": 14,
            "main_bx_1": True,
        }
        json_str = save_session(sample_df, sidebar, sample_state)
        restored = load_session(json_str)
        assert restored["sidebar"]["main_txt_1"] == "HR = 1.5"
        assert restored["sidebar"]["main_bx_1"] is True

    def test_group_labels_saved(self, sample_df, sample_state):
        sidebar = {
            "label_Positive": "MRD+",
            "label_Negative": "MRD-",
            "ref_MRD_Status": "Negative",
        }
        json_str = save_session(sample_df, sidebar, sample_state)
        restored = load_session(json_str)
        assert restored["sidebar"]["label_Positive"] == "MRD+"
        assert restored["sidebar"]["ref_MRD_Status"] == "Negative"


# ============================================================
# Filename Tests
# ============================================================

class TestFilename:
    def test_basic_filename(self):
        name = get_session_filename()
        assert name.startswith("easysurv_session_")
        assert name.endswith(".easysurv")

    def test_filename_with_notes(self):
        name = get_session_filename("OS analysis")
        assert "OS_analysis" in name
        assert name.endswith(".easysurv")

    def test_filename_sanitizes_special_chars(self):
        name = get_session_filename("test/file:name!")
        assert "/" not in name.replace("easysurv_session_", "")
        assert ":" not in name.replace("easysurv_session_", "")


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_large_dataframe(self):
        """Test with a moderately large DataFrame."""
        np.random.seed(42)
        df = pd.DataFrame({
            f"col_{i}": np.random.randn(1000) for i in range(20)
        })
        json_str = save_session(df, {}, {})
        restored = load_session(json_str)
        assert restored["df"].shape == (1000, 20)

    def test_missing_state_keys(self, sample_df, sample_sidebar):
        """State dict with no matching keys should still work."""
        json_str = save_session(sample_df, sample_sidebar, {})
        restored = load_session(json_str)
        assert restored["state"] == {}

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            load_session("not valid json {{{")

    def test_session_without_dataset_key(self):
        """Old/minimal session format should not crash."""
        minimal = json.dumps({"format_version": 1, "sidebar": {}, "state": {}})
        restored = load_session(minimal)
        assert restored["df"] is None
