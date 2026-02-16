"""
Session Manager for EasySurv.

Provides save/load functionality for analysis sessions. Sessions are exported
as self-contained JSON files (.easysurv) that include the dataset (compressed)
and all user configuration, allowing users to resume work later.
"""

import json
import base64
import gzip
import io
import datetime
import pandas as pd
import numpy as np

# Current session file format version
SESSION_FORMAT_VERSION = 1

# ============================================================
# Keys to save from st.session_state
# ============================================================

# These session_state keys hold serialisable analysis config/results.
# Matplotlib figures and non-serialisable objects are excluded —
# they will be regenerated when the user re-runs the analysis.

SAVEABLE_STATE_KEYS = [
    # Core flags
    "demo_loaded",
    # User-created variables
    "custom_cutoffs",
    "custom_combinations",
    # Penalizer settings (persist across interactions)
    "use_penalizer",
    "l1_ratio_val",
    "penalizer_val",
    "lambda_slider",
    # Multivariable analysis
    "mv_analysis_active",
    "uv_cox_method",
    "tune_options",
    # Competing risks two-column setup
    "two_col_cif_time",
    "two_col_cif_event",
    "two_col_cif_interest",
    # Optimal cutoff
    "optimal_cut",
    # Diagnostic / Prognostic results
    "diag_results",
    "prog_results",
    # Covariate selections (for restoring multivariable tab)
    "_saved_covariates",
]

# These session_state keys hold DataFrames that should be saved as CSV
DATAFRAME_STATE_KEYS = [
    "uv_cox_summary",
    "mv_summary_df",
    "two_col_cif_df",
]

# Sidebar widget values to save (variable name -> default value)
SIDEBAR_CONFIG_KEYS = {
    # Core data selection
    "time_col": None,
    "event_col": None,
    "group_col": "None",
    # Narrator
    "narrator_style_name": "Standard",
    # Typography
    "selected_font": "sans-serif",
    "title_fontsize": 20,
    "title_bold": True,
    "axes_fontsize": 12,
    "legend_fontsize": 10,
    "line_width": 1.5,
    # Global plot config
    "show_risk_table": True,
    "risk_table_format": "At-risk only",
    "table_height": -0.25,
    "show_censored": True,
    "show_ci": True,
    # Main KM plot settings
    "main_title": "Survival",
    "x_label": "Time (Months)",
    "y_label": "Survival Probability",
    "tick_interval": 12.0,
    "y_min": 0.0,
    "y_tick_interval": 0.1,
    "y_max": 1.0,
    "plot_height": 6,
    "plot_width": 10,
    "show_legend_main": True,
    "show_legend_box_main": True,
    "leg_x_main": 0.8,
    "leg_y_main": 0.9,
    "show_p_val_plot": False,
    "show_p_val_box_main": True,
    "pval_x_main": 0.95,
    "pval_y_main": 0.05,
    # CIF plot settings
    "cif_title": "Cumulative Incidence",
    "cif_y_label": "Cumulative Incidence Probability",
    "cif_y_min": 0.0,
    "cif_y_tick_interval": 0.1,
    "cif_y_max": 1.05,
    "show_legend_cif": True,
    "show_legend_box_cif": True,
    "leg_x_cif": 0.8,
    "leg_y_cif": 0.8,
    "show_p_val_plot_cif": False,
    "show_p_val_box_cif": True,
    "pval_x_cif": 0.95,
    "pval_y_cif": 0.2,
    # Theme / colors
    "selected_theme": "Default",
    "plot_bgcolor": "#FFFFFF",
    # Analysis settings
    "landmark_time": 0.0,
    "target_time": 24.0,
}


# ============================================================
# Compression helpers
# ============================================================

def _compress_dataframe(df):
    """Compress a DataFrame to a base64-encoded gzipped CSV string."""
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    compressed = gzip.compress(csv_bytes)
    return base64.b64encode(compressed).decode("ascii")


def _decompress_dataframe(b64_str):
    """Decompress a base64-encoded gzipped CSV string back to a DataFrame."""
    compressed = base64.b64decode(b64_str)
    csv_bytes = gzip.decompress(compressed)
    return pd.read_csv(io.BytesIO(csv_bytes), index_col=0)


# ============================================================
# Custom JSON encoder for numpy/pandas types
# ============================================================

class _SessionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


# ============================================================
# Save / Load
# ============================================================

def save_session(df, sidebar_config, session_state, notes=""):
    """
    Build a session dict and return it as a JSON string.

    Parameters
    ----------
    df : pd.DataFrame
        The currently loaded dataset.
    sidebar_config : dict
        Current sidebar widget values (key -> value).
    session_state : dict-like
        Streamlit session_state (or a plain dict for testing).
    notes : str
        Optional free-text notes from the user.

    Returns
    -------
    str
        JSON string representing the full session.
    """
    session = {
        "format_version": SESSION_FORMAT_VERSION,
        "saved_at": datetime.datetime.now().isoformat(),
        "notes": notes,
    }

    # 1. Dataset
    if df is not None and len(df) > 0:
        session["dataset"] = _compress_dataframe(df)
        session["dataset_shape"] = list(df.shape)
        session["dataset_columns"] = list(df.columns)
    else:
        session["dataset"] = None

    # 2. Sidebar config
    session["sidebar"] = {}
    for key, default in SIDEBAR_CONFIG_KEYS.items():
        val = sidebar_config.get(key, default)
        session["sidebar"][key] = val

    # Save custom text annotations (dynamic keys like main_txt_1, cif_txt_1, etc.)
    for prefix in ("main_txt_", "main_x_", "main_y_", "main_sz_", "main_bx_",
                    "cif_txt_", "cif_x_", "cif_y_", "cif_sz_", "cif_bx_"):
        for i in range(1, 6):
            key = f"{prefix}{i}"
            if key in sidebar_config:
                session["sidebar"][key] = sidebar_config[key]

    # Save dynamic group labels, colors, and reference groups
    for key, val in sidebar_config.items():
        if key.startswith("label_") or key.startswith("ref_") or key.startswith("color_"):
            session["sidebar"][key] = val

    # Also scan session_state for dynamic widget keys not in sidebar_config
    # (color_*, label_*, ref_* are set automatically by Streamlit widget keys)
    for key in list(session_state.keys()):
        if isinstance(key, str) and (
            key.startswith("color_") or key.startswith("label_") or key.startswith("ref_")
        ):
            if key not in session["sidebar"]:
                session["sidebar"][key] = session_state[key]

    # 3. Session state (serialisable scalars / lists / dicts)
    session["state"] = {}
    for key in SAVEABLE_STATE_KEYS:
        if key in session_state:
            session["state"][key] = session_state[key]

    # 4. Session state DataFrames (stored as compressed CSV)
    session["state_dataframes"] = {}
    for key in DATAFRAME_STATE_KEYS:
        if key in session_state and session_state[key] is not None:
            try:
                session["state_dataframes"][key] = _compress_dataframe(
                    session_state[key]
                )
            except Exception:
                pass  # Skip non-serialisable entries

    # 5. Filter state (dynamic keys)
    session["filters"] = {}
    if "filter_cols" in sidebar_config:
        session["filters"]["filter_cols"] = sidebar_config["filter_cols"]
    for key, val in sidebar_config.items():
        if key.startswith("filt_"):
            # Convert tuples to lists for JSON
            session["filters"][key] = list(val) if isinstance(val, tuple) else val

    return json.dumps(session, cls=_SessionEncoder, indent=2)


def load_session(json_str):
    """
    Parse a session JSON string and return structured data for restoration.

    Parameters
    ----------
    json_str : str
        JSON string from a .easysurv session file.

    Returns
    -------
    dict with keys:
        "df" : pd.DataFrame or None
        "sidebar" : dict of sidebar config values
        "state" : dict of session_state scalars
        "state_dataframes" : dict of key -> pd.DataFrame
        "filters" : dict of filter settings
        "notes" : str
        "saved_at" : str
        "format_version" : int
    """
    session = json.loads(json_str)

    result = {
        "format_version": session.get("format_version", 0),
        "saved_at": session.get("saved_at", ""),
        "notes": session.get("notes", ""),
    }

    # 1. Dataset
    if session.get("dataset"):
        result["df"] = _decompress_dataframe(session["dataset"])
    else:
        result["df"] = None

    # 2. Sidebar config
    result["sidebar"] = session.get("sidebar", {})

    # 3. Session state scalars
    result["state"] = session.get("state", {})

    # 4. Session state DataFrames
    result["state_dataframes"] = {}
    for key, b64 in session.get("state_dataframes", {}).items():
        try:
            result["state_dataframes"][key] = _decompress_dataframe(b64)
        except Exception:
            pass

    # 5. Filters
    result["filters"] = session.get("filters", {})

    return result


def get_session_filename(notes=""):
    """Generate a timestamped filename for the session file."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if notes:
        # Sanitise notes for filename
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in notes[:30])
        return f"easysurv_session_{safe}_{ts}.easysurv"
    return f"easysurv_session_{ts}.easysurv"
