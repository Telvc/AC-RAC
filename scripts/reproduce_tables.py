from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.covid.run_diagnostics import run_covid_diagnostics  # noqa: E402
from rac.data import COVID_ACTION_NAMES, COVID_LABEL_NAMES, COVID_UTILITY_MATRIX, MOVIELENS_LABEL_NAMES, MOVIELENS_UTILITY_MATRIX  # noqa: E402


METHOD_ORDER = ["AC-RAC", "RAC", "Score-1", "Score-2"]


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    path.with_suffix(".md").write_text(_to_markdown(df), encoding="utf-8")
    path.with_suffix(".tex").write_text(_to_latex(df), encoding="utf-8")


def _to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    widths = [len(c) for c in cols]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]
    header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep] + body) + "\n"


def _latex_escape(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_latex(df: pd.DataFrame) -> str:
    cols = [_latex_escape(c) for c in df.columns]
    lines = [
        r"\begin{tabular}{" + "l" * len(cols) + "}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_latex_escape(v) for v in row.tolist()) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _table1() -> pd.DataFrame:
    rows = []
    for y, label in COVID_LABEL_NAMES.items():
        row = {"True Label": f"{label} ({y})"}
        for a, name in COVID_ACTION_NAMES.items():
            row[name] = int(COVID_UTILITY_MATRIX[y, a])
        rows.append(row)
    return pd.DataFrame(rows)


def _table2() -> pd.DataFrame:
    rows = []
    for action, vals in [("No-Recommend", MOVIELENS_UTILITY_MATRIX[:, 0]), ("Recommend", MOVIELENS_UTILITY_MATRIX[:, 1])]:
        row = {"Action": action}
        for idx, label in MOVIELENS_LABEL_NAMES.items():
            row[label] = int(vals[idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _table3(exp1: pd.DataFrame) -> pd.DataFrame:
    return exp1.pivot_table(index="alpha", columns="method", values="set_size_mean").reset_index()[["alpha"] + METHOD_ORDER]


def _table4(exp4_fdr: pd.DataFrame) -> pd.DataFrame:
    out = exp4_fdr[["method", "set_size_mean", "fdr_overall"]].copy()
    return out.rename(columns={"method": "Method", "set_size_mean": "Mean prediction-set size", "fdr_overall": "Overall FDR"})


def _table5(exp4_freq: pd.DataFrame) -> pd.DataFrame:
    sub = exp4_freq.copy()
    sub["percent"] = 100.0 * sub["frequency"]
    pivot = sub.pivot_table(index="method", columns="action_name", values="percent").reset_index()
    return pivot.rename(columns={"method": "Method"})


def _table6(exp3: pd.DataFrame) -> pd.DataFrame:
    sub = exp3[exp3["method"].isin(["AC-RAC", "RAC"])].copy()
    sub["Target prevalence"] = sub["scenario"].replace({"full": "5.3% (full)"})
    return sub[["Target prevalence", "method", "target_cov", "coverage_overall", "set_size_mean"]].rename(
        columns={
            "method": "Method",
            "target_cov": "Target-action coverage",
            "coverage_overall": "Overall coverage",
            "set_size_mean": "Mean set size",
        }
    )


def _table7(exp2: pd.DataFrame) -> pd.DataFrame:
    sub = exp2[exp2["method"].isin(["AC-RAC", "RAC"])].copy()
    return sub[["|A|", "method", "set_size_mean", "coverage_overall", "critical_error_rate", "num_actions_used"]].rename(
        columns={
            "method": "Method",
            "set_size_mean": "Mean set size",
            "coverage_overall": "Overall coverage",
            "critical_error_rate": "Critical error rate",
            "num_actions_used": "Actions used",
        }
    )


def reproduce_tables(
    fast: bool = True,
    output_dir: str | Path = "artifacts/tables",
    covid_cache_dir: str | Path = "data/cached/covid",
    covid_cache_prefix: str = "",
) -> None:
    out = Path(output_dir)
    dfs = run_covid_diagnostics(fast=fast, cache_dir=covid_cache_dir, cache_prefix=covid_cache_prefix)
    tables = {
        "table1_covid_utility": _table1(),
        "table2_movielens_utility": _table2(),
        "table3_covid_set_size": _table3(dfs["exp1"]),
        "table4_covid_fdr": _table4(dfs["exp4_fdr"]),
        "table5_covid_action_frequencies": _table5(dfs["exp4_freq"]),
        "table6_covid_rare_action": _table6(dfs["exp3"]),
        "table7_covid_action_scaling": _table7(dfs["exp2"]),
    }
    for name, df in tables.items():
        _write(df, out / f"{name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use cached results where available.")
    parser.add_argument("--output-dir", default="artifacts/tables")
    parser.add_argument("--covid-cache-dir", default="data/cached/covid")
    parser.add_argument("--covid-cache-prefix", default="")
    args = parser.parse_args()
    reproduce_tables(
        fast=args.fast,
        output_dir=args.output_dir,
        covid_cache_dir=args.covid_cache_dir,
        covid_cache_prefix=args.covid_cache_prefix,
    )
