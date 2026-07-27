import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import session_bank as sb


def _fig():
    f, ax = plt.subplots()
    ax.plot([0, 1, 2], [2, 1, 0])
    return f


def test_fig_roundtrip_is_base64_str():
    b64 = sb.fig_to_png_b64(_fig(), dpi=80)
    assert isinstance(b64, str) and len(b64) > 100
    raw = sb.png_b64_to_bytes(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic number


def test_table_roundtrip():
    df = pd.DataFrame({"HR": [1.5, 0.8], "p": [0.01, 0.4]}, index=["A", "B"])
    b64 = sb.table_to_b64(df)
    assert isinstance(b64, str)
    back = sb.b64_to_table(b64)
    pd.testing.assert_frame_equal(df, back)


def test_make_entry_is_json_serializable():
    entry = sb.make_entry(
        "KM", "OS by LSC", fig=_fig(),
        tables={"Median": pd.DataFrame({"m": [12.0]}), "Empty": None},
        narrative="Some text", meta={"endpoint": "OS", "n": 153}, dpi=80,
    )
    # The whole entry must survive a JSON round-trip (the crux of save/share)
    s = json.dumps(entry)
    back = json.loads(s)
    assert back["type"] == "KM"
    assert isinstance(back["png"], str)
    assert len(back["tables"]) == 1          # None table skipped
    assert back["narrative"] == "Some text"
    assert sb.is_serializable_entry(back)


def test_entry_tables_restores_dataframes():
    df = pd.DataFrame({"x": [1, 2, 3]})
    entry = sb.make_entry("Table1", "Baseline", tables={"T1": df}, dpi=80)
    got = dict(sb.entry_tables(entry))
    assert "T1" in got
    pd.testing.assert_frame_equal(got["T1"], df)


def test_table_only_entry_has_no_png():
    entry = sb.make_entry("Table1", "Baseline", tables={"T1": pd.DataFrame({"x": [1]})})
    assert entry["png"] is None
    assert sb.is_serializable_entry(entry)


def test_is_serializable_rejects_live_fig():
    bad = {"type": "KM", "label": "x", "png": _fig(), "tables": []}
    assert not sb.is_serializable_entry(bad)
