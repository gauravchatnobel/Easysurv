import sys, os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tableone import generate_table_one


@pytest.fixture
def cohort():
    np.random.seed(0)
    n = 120
    grp = np.array(["A"] * 60 + ["B"] * 60)
    age = np.where(grp == "A", np.random.normal(55, 8, n), np.random.normal(65, 8, n))
    sex = np.random.choice(["M", "F"], n)
    stage = np.random.choice(["I", "II", "III"], n)
    return pd.DataFrame({"Group": grp, "Age": age, "Sex": sex, "Stage": stage})


def test_overall_only_no_tests(cohort):
    tbl, meta = generate_table_one(cohort, group_col=None, variables=["Age", "Sex"])
    assert "Overall" in tbl.columns
    assert "p-value" not in tbl.columns
    assert meta["groups"] == []


def test_grouped_has_group_columns_and_p(cohort):
    tbl, meta = generate_table_one(cohort, group_col="Group", variables=["Age", "Sex", "Stage"])
    assert {"Overall", "A", "B", "p-value"}.issubset(set(tbl.columns))
    # n row present with correct sizes
    n_row = tbl[tbl["Characteristic"] == "n"].iloc[0]
    assert n_row["A"] == "60" and n_row["B"] == "60"


def test_continuous_detects_group_difference(cohort):
    # Age differs strongly by group -> small p
    tbl, _ = generate_table_one(cohort, group_col="Group", variables=["Age"])
    age_row = tbl[tbl["Characteristic"].str.startswith("Age")].iloc[0]
    assert age_row["p-value"] in ("<0.001",) or float(age_row["p-value"]) < 0.05
    assert "[" in age_row["Overall"]  # median [IQR] format


def test_categorical_levels_sum_to_100(cohort):
    tbl, _ = generate_table_one(cohort, group_col="Group", variables=["Stage"])
    lvl_rows = tbl[tbl["Characteristic"].str.startswith("    ")]
    pcts = [float(v.split("(")[1].rstrip(")")) for v in lvl_rows["Overall"]]
    assert abs(sum(pcts) - 100.0) < 0.5


def test_mean_sd_style(cohort):
    tbl, _ = generate_table_one(cohort, group_col="Group", variables=["Age"],
                                nonnormal_vars=[], continuous_style="mean")
    age_row = tbl[tbl["Characteristic"].str.startswith("Age")].iloc[0]
    assert "mean (SD)" in age_row["Characteristic"]
    assert "(" in age_row["Overall"] and "[" not in age_row["Overall"]


def test_fisher_for_small_2x2():
    df = pd.DataFrame({
        "Group": ["A"] * 10 + ["B"] * 10,
        "Marker": ["+"] * 2 + ["-"] * 8 + ["+"] * 9 + ["-"] * 1,
    })
    tbl, meta = generate_table_one(df, group_col="Group", variables=["Marker"])
    assert meta["tests"]["Marker"] in ("Fisher's exact", "Chi-square")
