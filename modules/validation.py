"""
Data validation module for EasySurv.
Performs early checks on uploaded clinical data to catch common issues
before they cause cryptic errors during analysis.
"""

import pandas as pd
import numpy as np


def validate_dataset(df, time_col=None, event_col=None, group_col=None):
    """
    Validates a clinical dataset and returns a list of issues found.

    Each issue is a dict with:
        - level: 'error', 'warning', or 'info'
        - message: Human-readable description
        - column: The affected column (if applicable)

    Returns:
        list[dict]: List of validation issues, empty if none found.
    """
    issues = []

    if df is None or df.empty:
        issues.append({
            "level": "error",
            "message": "Dataset is empty or could not be loaded.",
            "column": None,
        })
        return issues

    # --- General checks ---
    if len(df) < 10:
        issues.append({
            "level": "warning",
            "message": f"Dataset has only {len(df)} rows. Most survival analyses require larger samples for reliable results.",
            "column": None,
        })

    # Check for duplicate column names
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        issues.append({
            "level": "error",
            "message": f"Duplicate column names detected: {', '.join(dup_cols)}. Please rename before analysis.",
            "column": None,
        })

    # --- Time column checks ---
    if time_col and time_col in df.columns:
        time_series = df[time_col]

        if not pd.api.types.is_numeric_dtype(time_series):
            coerced = pd.to_numeric(time_series, errors="coerce")
            n_failed = coerced.isna().sum() - time_series.isna().sum()
            if n_failed > 0:
                issues.append({
                    "level": "error",
                    "message": f"Time column '{time_col}' contains {n_failed} non-numeric values that cannot be converted.",
                    "column": time_col,
                })
        else:
            # Negative times
            n_negative = (time_series < 0).sum()
            if n_negative > 0:
                issues.append({
                    "level": "error",
                    "message": f"Time column '{time_col}' has {n_negative} negative values. Survival times must be >= 0.",
                    "column": time_col,
                })

            # Zero times
            n_zero = (time_series == 0).sum()
            if n_zero > 0:
                issues.append({
                    "level": "warning",
                    "message": f"Time column '{time_col}' has {n_zero} zero values. These patients had events at time 0.",
                    "column": time_col,
                })

            # Outliers (values > 3 IQR above Q3)
            q1 = time_series.quantile(0.25)
            q3 = time_series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                upper_fence = q3 + 3 * iqr
                n_outliers = (time_series > upper_fence).sum()
                if n_outliers > 0:
                    issues.append({
                        "level": "info",
                        "message": f"Time column '{time_col}' has {n_outliers} potential outlier(s) (> {upper_fence:.1f}). Verify these are valid.",
                        "column": time_col,
                    })

        # Missing values
        n_missing = time_series.isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            issues.append({
                "level": "warning" if pct < 20 else "error",
                "message": f"Time column '{time_col}' has {n_missing} missing values ({pct:.1f}%). These rows will be excluded.",
                "column": time_col,
            })

    # --- Event column checks ---
    if event_col and event_col in df.columns:
        event_series = df[event_col]

        if not pd.api.types.is_numeric_dtype(event_series):
            coerced = pd.to_numeric(event_series, errors="coerce")
            n_failed = coerced.isna().sum() - event_series.isna().sum()
            if n_failed > 0:
                issues.append({
                    "level": "error",
                    "message": f"Event column '{event_col}' contains {n_failed} non-numeric values.",
                    "column": event_col,
                })
        else:
            unique_vals = sorted(event_series.dropna().unique())

            # Check for unexpected event codes
            if len(unique_vals) > 0:
                expected_binary = {0, 1}
                expected_competing = {0, 1, 2}
                val_set = set(unique_vals)

                if not (val_set <= expected_binary or val_set <= expected_competing):
                    issues.append({
                        "level": "warning",
                        "message": f"Event column '{event_col}' has unexpected values: {unique_vals}. Expected 0/1 (binary) or 0/1/2 (competing risks).",
                        "column": event_col,
                    })

            # No events at all
            n_events = (event_series == 1).sum()
            if n_events == 0:
                issues.append({
                    "level": "error",
                    "message": f"Event column '{event_col}' has zero events (no rows with value=1). Cannot perform survival analysis.",
                    "column": event_col,
                })
            elif n_events < 5:
                issues.append({
                    "level": "warning",
                    "message": f"Event column '{event_col}' has only {n_events} events. Results may be unreliable.",
                    "column": event_col,
                })

        # Missing values
        n_missing = event_series.isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            issues.append({
                "level": "warning" if pct < 20 else "error",
                "message": f"Event column '{event_col}' has {n_missing} missing values ({pct:.1f}%).",
                "column": event_col,
            })

    # --- Group column checks ---
    if group_col and group_col != "None" and group_col in df.columns:
        group_series = df[group_col]

        n_missing = group_series.isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            issues.append({
                "level": "warning",
                "message": f"Grouping variable '{group_col}' has {n_missing} missing values ({pct:.1f}%).",
                "column": group_col,
            })

        unique_groups = group_series.dropna().unique()
        if len(unique_groups) > 20:
            issues.append({
                "level": "warning",
                "message": f"Grouping variable '{group_col}' has {len(unique_groups)} unique values. This may be a continuous variable rather than a categorical grouping.",
                "column": group_col,
            })

        # Check for groups with very few observations
        if time_col and event_col:
            group_counts = group_series.value_counts()
            small_groups = group_counts[group_counts < 5]
            if len(small_groups) > 0:
                names = ", ".join(str(g) for g in small_groups.index[:5])
                issues.append({
                    "level": "warning",
                    "message": f"Grouping variable '{group_col}' has groups with < 5 observations: {names}. Consider merging small groups.",
                    "column": group_col,
                })

    # --- Date column detection ---
    for col in df.columns.unique():
        sample = df[col].dropna().head(20)
        if isinstance(sample, pd.DataFrame):
            continue  # Skip duplicate column names
        if sample.dtype == object:
            date_like = 0
            for val in sample:
                s = str(val)
                # Simple heuristic: looks like a date
                if any(sep in s for sep in ["/", "-"]) and any(c.isdigit() for c in s) and len(s) >= 8:
                    date_like += 1
            if date_like > len(sample) * 0.7 and len(sample) > 3:
                issues.append({
                    "level": "info",
                    "message": f"Column '{col}' appears to contain dates. If this represents a time variable, convert it to numeric duration before analysis.",
                    "column": col,
                })

    return issues


def format_validation_report(issues):
    """
    Formats validation issues into grouped markdown text.

    Returns:
        tuple: (errors: list[str], warnings: list[str], infos: list[str])
    """
    errors = [i["message"] for i in issues if i["level"] == "error"]
    warnings = [i["message"] for i in issues if i["level"] == "warning"]
    infos = [i["message"] for i in issues if i["level"] == "info"]
    return errors, warnings, infos
