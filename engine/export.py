"""
Export engine results to CSV and Excel (.xlsx).

The engine's outputs — screen signals, validation reports, bake-off/gauntlet
verdicts, the live journal — are the product. This module turns any of them into
flat tables and writes them out: a single CSV, or a multi-sheet Excel workbook
(one sheet per result kind). It is pure marshalling (no analysis), so it stays
stable as scanners come and go.

`export(sheets, path)` dispatches on the file extension: `.xlsx` -> one workbook
with a sheet per entry; `.csv` -> one file (single sheet) or `<stem>_<name>.csv`
per sheet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd


# --------------------------------------------------------------------------- #
# Result -> DataFrame converters
# --------------------------------------------------------------------------- #
def signals_to_df(signals: Iterable) -> pd.DataFrame:
    return pd.DataFrame([s.as_dict() for s in signals])


def reports_to_df(reports: Iterable) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "n_events": r.n_events,
                "n_folds": r.n_folds,
                "oos_edge_r": r.oos_net_expectancy_r,
                "oos_hit_rate": r.oos_hit_rate,
                "oos_auc": r.oos_auc,
                "p_value": r.p_value,
                "p_value_fdr": r.p_value_fdr,
                "decay": r.decay,
                "n_signals": r.n_total_signals,
            }
            for r in reports
        ]
    )


def screen_to_frames(result) -> dict[str, pd.DataFrame]:
    """A ScreenResult -> {signals, reports} sheets."""
    return {"signals": signals_to_df(result.signals), "reports": reports_to_df(result.reports)}


def bakeoff_to_df(rows: Iterable) -> pd.DataFrame:
    out = []
    for r in rows:
        t = r.result
        out.append(
            {
                "version": r.version,
                "model": r.model_kind,
                "validated_edge_r": t.validated_edge_r,
                "realized_edge_r": t.realized_edge_r,
                "forward_decay_r": t.forward_decay_r,
                "holdout_auc": t.holdout_auc,
                "n_holdout_signals": t.n_holdout_signals,
                "n_holdout_days": t.n_holdout_days,
                "realized_p": t.realized_p,
                "p_value_fdr": r.p_value_fdr,
                "persisted": t.persisted,
                "promoted": r.promoted,
            }
        )
    return pd.DataFrame(out)


def gauntlet_to_frames(v) -> dict[str, pd.DataFrame]:
    """A GauntletVerdict -> {summary, bakeoff, rolling} sheets."""
    summary = pd.DataFrame(
        [
            {
                "direction": v.direction,
                "n_events": v.n_events,
                "distinct_days": v.distinct_days,
                "passed": v.passed,
            }
        ]
    )
    rolling = pd.DataFrame(
        [
            {
                "model": m,
                "n_windows": rr.n_windows,
                "n_persisted": rr.n_persisted,
                "pooled_realized_edge_r": rr.pooled_realized_edge_r,
                "pooled_realized_p": rr.pooled_realized_p,
                "pooled_holdout_days": rr.pooled_holdout_days,
                "robust": rr.robust,
            }
            for m, rr in v.rolling.items()
        ]
    )
    return {"summary": summary, "bakeoff": bakeoff_to_df(v.bakeoff), "rolling": rolling}


def journal_to_df(entries: Iterable[dict]) -> pd.DataFrame:
    return pd.DataFrame(list(entries))


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_excel(sheets: Mapping[str, pd.DataFrame], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        any_written = False
        for name, df in sheets.items():
            d = df if (df is not None and not df.empty) else pd.DataFrame({"(no rows)": []})
            d.to_excel(xl, sheet_name=str(name)[:31], index=False)  # Excel: 31-char sheet cap
            any_written = True
        if not any_written:
            pd.DataFrame({"(no rows)": []}).to_excel(xl, sheet_name="results", index=False)
    return path


def export(sheets: Mapping[str, pd.DataFrame], path: str | Path) -> list[Path]:
    """Write result sheets by file extension.

    `.xlsx`/`.xls` -> one multi-sheet workbook. `.csv` -> one file if there is a
    single sheet, else one `<stem>_<name>.csv` per sheet.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return [write_excel(sheets, path)]
    sheets = dict(sheets)
    if len(sheets) <= 1:
        df = next(iter(sheets.values())) if sheets else pd.DataFrame()
        return [write_csv(df, path)]
    return [write_csv(df, path.with_name(f"{path.stem}_{name}.csv")) for name, df in sheets.items()]
