"""
Tests for the data validation module.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.validation import validate_dataset, format_validation_report


@pytest.fixture
def good_df():
    """A clean dataset with no issues."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame({
        "Time": np.random.exponential(20, n),
        "Event": np.random.binomial(1, 0.5, n),
        "Group": np.random.choice(["A", "B"], n),
    })


@pytest.fixture
def bad_df():
    """A dataset with several known issues."""
    return pd.DataFrame({
        "Time": [10, -5, 0, np.nan, 20, 30, 40, 50, 60, 70],
        "Event": [1, 0, 1, 1, 0, 3, 1, 0, np.nan, 1],
        "Group": ["A", "A", "B", "B", "A", "B", "A", "B", "A", None],
    })


class TestValidateDataset:
    def test_clean_data_no_errors(self, good_df):
        """Clean data should produce no errors."""
        issues = validate_dataset(good_df, "Time", "Event", "Group")
        errors = [i for i in issues if i["level"] == "error"]
        assert len(errors) == 0

    def test_empty_dataframe(self):
        """Empty dataframe should return error."""
        issues = validate_dataset(pd.DataFrame())
        assert any(i["level"] == "error" for i in issues)

    def test_none_dataframe(self):
        """None should return error."""
        issues = validate_dataset(None)
        assert any(i["level"] == "error" for i in issues)

    def test_negative_times(self, bad_df):
        """Negative time values should be flagged as error."""
        issues = validate_dataset(bad_df, time_col="Time")
        messages = [i["message"] for i in issues if i["level"] == "error"]
        assert any("negative" in m.lower() for m in messages)

    def test_zero_times(self, bad_df):
        """Zero time values should produce a warning."""
        issues = validate_dataset(bad_df, time_col="Time")
        messages = [i["message"] for i in issues if i["level"] == "warning"]
        assert any("zero" in m.lower() for m in messages)

    def test_missing_time(self, bad_df):
        """Missing values in time column should be flagged."""
        issues = validate_dataset(bad_df, time_col="Time")
        messages = [i["message"] for i in issues]
        assert any("missing" in m.lower() and "Time" in m for m in messages)

    def test_unexpected_event_codes(self, bad_df):
        """Event code=3 should trigger warning about unexpected values."""
        issues = validate_dataset(bad_df, event_col="Event")
        messages = [i["message"] for i in issues if i["level"] == "warning"]
        assert any("unexpected" in m.lower() for m in messages)

    def test_missing_group_values(self, bad_df):
        """Missing group values should be flagged."""
        issues = validate_dataset(bad_df, group_col="Group")
        messages = [i["message"] for i in issues]
        assert any("missing" in m.lower() and "Group" in m for m in messages)

    def test_no_events(self):
        """Dataset with zero events should be flagged as error."""
        df = pd.DataFrame({
            "Time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Event": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        issues = validate_dataset(df, event_col="Event")
        errors = [i for i in issues if i["level"] == "error"]
        assert any("zero events" in e["message"].lower() for e in errors)

    def test_duplicate_columns(self):
        """Duplicate column names should be flagged."""
        df = pd.DataFrame([[1, 2, 3]], columns=["A", "B", "A"])
        issues = validate_dataset(df)
        assert any("duplicate" in i["message"].lower() for i in issues)

    def test_many_groups_warning(self, good_df):
        """Group column with too many unique values should warn."""
        df = good_df.copy()
        df["ManyGroups"] = range(len(df))
        issues = validate_dataset(df, group_col="ManyGroups")
        messages = [i["message"] for i in issues if i["level"] == "warning"]
        assert any("unique values" in m.lower() for m in messages)

    def test_date_column_detection(self):
        """Date-like columns should be flagged as info."""
        df = pd.DataFrame({
            "Time": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Event": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "DiagDate": [
                "2020-01-15", "2020-02-20", "2020-03-10", "2020-04-05",
                "2020-05-12", "2020-06-18", "2020-07-22", "2020-08-30",
                "2020-09-14", "2020-10-28",
            ],
        })
        issues = validate_dataset(df)
        infos = [i for i in issues if i["level"] == "info"]
        assert any("date" in i["message"].lower() for i in infos)


class TestFormatValidationReport:
    def test_format_groups_correctly(self):
        """Format function should separate by level."""
        issues = [
            {"level": "error", "message": "err1", "column": None},
            {"level": "warning", "message": "warn1", "column": None},
            {"level": "info", "message": "info1", "column": None},
        ]
        errors, warnings, infos = format_validation_report(issues)
        assert errors == ["err1"]
        assert warnings == ["warn1"]
        assert infos == ["info1"]

    def test_empty_issues(self):
        """Empty list should return empty groups."""
        errors, warnings, infos = format_validation_report([])
        assert errors == []
        assert warnings == []
        assert infos == []
