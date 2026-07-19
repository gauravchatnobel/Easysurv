"""Baseline characteristics ("Table 1") generation for clinical cohorts.

Produces a by-group summary table with appropriate descriptive statistics and
hypothesis tests, in the layout expected by oncology/haematology journals.

Design choices (documented in the Methodology tab):
  * Continuous variables: median [Q1-Q3] by default (robust to the skewed
    distributions common in clinical data); mean (SD) optionally.
  * Categorical variables: n (%) within each group.
  * Tests: Mann-Whitney U / Kruskal-Wallis for continuous (or t-test / ANOVA
    if the variable is declared normal); Pearson chi-square for categorical,
    falling back to Fisher's exact test for 2x2 tables with a low expected
    count. Tests are omitted when no grouping variable is supplied.
"""

import numpy as np
import pandas as pd


def _fmt(x, dp=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{dp}f}"


def _is_continuous(series, max_unique=10):
    """Treat numeric columns with more than max_unique distinct values as continuous."""
    s = series.dropna()
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > max_unique


def _continuous_summary(series, nonnormal=True, dp=1):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "—"
    if nonnormal:
        q1, med, q3 = np.percentile(s, [25, 50, 75])
        return f"{med:.{dp}f} [{q1:.{dp}f}–{q3:.{dp}f}]"
    return f"{s.mean():.{dp}f} ({s.std(ddof=1):.{dp}f})"


def _continuous_test(df, var, group_col, groups, nonnormal=True):
    from scipy import stats
    samples = [pd.to_numeric(df.loc[df[group_col] == g, var], errors="coerce").dropna().values
               for g in groups]
    samples = [s for s in samples if len(s) > 1]
    if len(samples) < 2:
        return np.nan, ""
    try:
        if nonnormal:
            if len(samples) == 2:
                stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
                return p, "Mann-Whitney U"
            stat, p = stats.kruskal(*samples)
            return p, "Kruskal-Wallis"
        else:
            if len(samples) == 2:
                stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False)
                return p, "Welch t-test"
            stat, p = stats.f_oneway(*samples)
            return p, "ANOVA"
    except Exception:
        return np.nan, ""


def _categorical_test(df, var, group_col, groups):
    from scipy import stats
    sub = df[[var, group_col]].dropna()
    if sub.empty:
        return np.nan, ""
    ct = pd.crosstab(sub[var], sub[group_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan, ""
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        # Fisher's exact for 2x2 with a low expected cell
        if ct.shape == (2, 2) and (expected < 5).any():
            _, p = stats.fisher_exact(ct.values)
            return p, "Fisher's exact"
        return p, "Chi-square"
    except Exception:
        return np.nan, ""


def generate_table_one(df, group_col=None, variables=None, nonnormal_vars=None,
                       continuous_style="median", max_unique=10):
    """Build a baseline-characteristics table.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str or None
        Column to stratify by. If None, only an 'Overall' column is produced.
    variables : list[str] or None
        Variables to summarise (defaults to all columns except group_col).
    nonnormal_vars : list[str] or None
        Continuous variables to summarise as median [IQR] / test non-parametrically.
        If None, ALL continuous variables are treated as non-normal.
    continuous_style : "median" (median [IQR]) or "mean" (mean (SD)) — the default
        for variables not explicitly listed in nonnormal_vars.
    max_unique : numeric columns with <= max_unique distinct values are categorical.

    Returns
    -------
    (table_df, meta) : a display DataFrame and a dict with the tests used.
    """
    if variables is None:
        variables = [c for c in df.columns if c != group_col]

    groups = []
    if group_col is not None and group_col in df.columns:
        groups = [g for g in sorted(df[group_col].dropna().unique())]

    rows = []
    tests_used = {}

    # Header row with group sizes
    n_overall = len(df)
    header = {"Characteristic": "n", "Overall": str(n_overall)}
    for g in groups:
        header[str(g)] = str(int((df[group_col] == g).sum()))
    if groups:
        header["p-value"] = ""
        header["Test"] = ""
    rows.append(header)

    for var in variables:
        if var not in df.columns:
            continue
        continuous = _is_continuous(df[var], max_unique=max_unique)
        if nonnormal_vars is None:
            nonnormal = True
        else:
            nonnormal = var in nonnormal_vars or continuous_style == "median"

        if continuous:
            label = f"{var}, {'median [IQR]' if nonnormal else 'mean (SD)'}"
            row = {"Characteristic": label,
                   "Overall": _continuous_summary(df[var], nonnormal)}
            for g in groups:
                row[str(g)] = _continuous_summary(df.loc[df[group_col] == g, var], nonnormal)
            if groups:
                p, test = _continuous_test(df, var, group_col, groups, nonnormal)
                row["p-value"] = p
                row["Test"] = test
                tests_used[var] = test
            rows.append(row)
        else:
            # Categorical: one row per level, n (%)
            s = df[var].dropna()
            levels = sorted(s.unique(), key=lambda x: str(x))
            # Parent row carries the test
            parent = {"Characteristic": f"{var}, n (%)", "Overall": ""}
            for g in groups:
                parent[str(g)] = ""
            if groups:
                p, test = _categorical_test(df, var, group_col, groups)
                parent["p-value"] = p
                parent["Test"] = test
                tests_used[var] = test
            rows.append(parent)
            for lvl in levels:
                def pct(mask_df):
                    sub = mask_df[var].dropna()
                    n = int((sub == lvl).sum())
                    d = len(sub)
                    return f"{n} ({100.0 * n / d:.1f})" if d else "—"
                row = {"Characteristic": f"    {lvl}", "Overall": pct(df)}
                for g in groups:
                    row[str(g)] = pct(df[df[group_col] == g])
                if groups:
                    row["p-value"] = ""
                    row["Test"] = ""
                rows.append(row)

    table = pd.DataFrame(rows)
    # Format numeric p-values for display
    if "p-value" in table.columns:
        table["p-value"] = table["p-value"].apply(
            lambda v: "" if (v == "" or v is None or (isinstance(v, float) and np.isnan(v)))
            else ("<0.001" if v < 0.001 else f"{v:.3f}"))
    return table, {"tests": tests_used, "groups": [str(g) for g in groups]}
