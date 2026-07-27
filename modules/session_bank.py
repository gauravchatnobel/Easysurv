"""Serializable 'Analysis Bank' for accumulating analyses across a session.

Each pinned analysis is stored as *pure JSON-serializable content* — a base64
PNG of the figure, base64-gzipped CSVs of its tables, the AI-narrative text, and
metadata — never a live matplotlib figure or DataFrame. This lets the bank:
  * survive Streamlit reruns (no fragile object references),
  * be written into and restored from a saved-session file, and
  * be shared: whoever loads the session sees every analysis.

Entry schema:
    {
      "type": "KM" | "CIF" | "Cox" | "FineGray" | "RMTL" | "Table1" | ...,
      "label": str,                 # user-facing name
      "title": str,                 # plot title (optional)
      "png": base64 str | None,     # figure at `dpi` (None for table-only entries)
      "tables": [ {"name": str, "data": base64-gzip-csv} ],
      "narrative": str | None,      # AI narrator text
      "meta": { ... arbitrary JSON-serializable metadata ... },
    }
"""

import io
import gzip
import base64

import pandas as pd


def fig_to_png_b64(fig, dpi=300):
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def png_b64_to_bytes(b64):
    """Decode a base64 PNG string back to raw PNG bytes (for st.image / plt.imread)."""
    if not b64:
        return None
    return base64.b64decode(b64)


def table_to_b64(df):
    """Compress a DataFrame to a base64-gzipped CSV string (index preserved)."""
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    return base64.b64encode(gzip.compress(csv_bytes)).decode("ascii")


def b64_to_table(b64):
    """Restore a DataFrame from a base64-gzipped CSV string."""
    raw = gzip.decompress(base64.b64decode(b64))
    return pd.read_csv(io.BytesIO(raw), index_col=0)


def make_entry(entry_type, label, fig=None, title="", tables=None,
               narrative=None, meta=None, dpi=300):
    """Build one fully-serializable bank entry.

    tables : dict[str, DataFrame|None] — table name -> DataFrame (None skipped).
    """
    serial_tables = []
    for name, df in (tables or {}).items():
        if df is None:
            continue
        try:
            serial_tables.append({"name": name, "data": table_to_b64(df)})
        except Exception:
            continue
    return {
        "type": entry_type,
        "label": label,
        "title": title or label,
        "png": fig_to_png_b64(fig, dpi=dpi) if fig is not None else None,
        "tables": serial_tables,
        "narrative": narrative,
        "meta": meta or {},
    }


def entry_tables(entry):
    """Yield (name, DataFrame) for each stored table in an entry."""
    for t in entry.get("tables", []):
        try:
            yield t["name"], b64_to_table(t["data"])
        except Exception:
            continue


def is_serializable_entry(entry):
    """True if an entry contains no live objects (safe to JSON-dump / save)."""
    if not isinstance(entry, dict):
        return False
    if entry.get("png") is not None and not isinstance(entry["png"], str):
        return False
    for t in entry.get("tables", []):
        if not isinstance(t.get("data"), str):
            return False
    return True
