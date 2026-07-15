import re
import pandas as pd
from pathlib import Path

keyranges_dir = Path("nsys-bench")

_STEP_RE = re.compile(r" @ step_\d+$")


def consolidate_step_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-step rows (Range ends with '@ step_N') into a single averaged row."""
    step_mask = df["Range"].str.contains(_STEP_RE.pattern, regex=True)
    non_step_df = df[~step_mask].copy()
    step_df = df[step_mask].copy()

    if step_df.empty:
        return df

    step_df["_step_num"] = step_df["Range"].str.extract(r"@ step_(\d+)$")[0].astype(int)
    step_df["Range"] = step_df["Range"].str.replace(_STEP_RE.pattern, "", regex=True)

    numeric_cols = step_df.select_dtypes(include="number").columns.tolist()
    agg: dict = {col: "mean" for col in numeric_cols}
    if "Range Instances" in agg:
        agg["Range Instances"] = "sum"
    if "Proj Min (ms)" in agg:
        agg["Proj Min (ms)"] = "min"
    if "Proj Max (ms)" in agg:
        agg["Proj Max (ms)"] = "max"
    if "_step_num" in agg:
        agg["_step_num"] = "first"
    for col in step_df.columns:
        if col not in numeric_cols and col != "Range":
            agg[col] = "first"

    step_ranges = step_df.groupby("Range")["_step_num"].agg(["min", "max"])
    consolidated = step_df.groupby("Range", as_index=False).agg(agg)
    consolidated["Range"] = consolidated["Range"].map(
        lambda r: f"{r} @ step_{step_ranges.loc[r, 'min']}-{step_ranges.loc[r, 'max']}"
    )
    consolidated = consolidated.drop(columns=["_step_num"])
    return pd.concat([non_step_df, consolidated], ignore_index=True)


# ── Phase 1: summarize keyranges ─────────────────────────────────────────────

dfs = []
for csv_path in sorted(keyranges_dir.glob("*keyranges.csv")):
    print(f"Processing {csv_path}")
    if csv_path.stat().st_size == 0:
        print(f"  Skipping empty file: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"  Skipping file with no rows: {csv_path}")
        continue
    df["prof_name"] = csv_path.name.split(".")[0]
    df = consolidate_step_rows(df)
    df = df.sort_values("Avg Range Lvl")
    proj_cols = [c for c in df.columns if c.startswith("Proj")]
    df = df[["prof_name", "Range", "Range Instances", "Avg Range Lvl"] + proj_cols]
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

output_str = combined.to_string(float_format=lambda x: f"{x:.1f}")
print(output_str)

(keyranges_dir / "summarized_keyranges.txt").write_text(output_str)
combined.to_csv(keyranges_dir / "summarized_keyranges.csv", index=False, float_format="%.1f")

# ── Phase 2: pivot into per-metric table ──────────────────────────────────────

METRIC_COLS = [
    "tr-forward", "tr-backward", "tr-optimize",
    "fw.Attn", "fw.MoE",
    "dispatch.fw", "experts.fw", "combine.fw",
    "fw.a2a_dispatch", "fw.a2a_combine",
    "bw.a2a_combine", "bw.a2a_dispatch",
]

pivot_df = combined.copy()
pivot_df["Range"] = pivot_df["Range"].str.lstrip(":")

rows = []
for prof_name, group in pivot_df.groupby("prof_name", sort=False):
    top = group[group["Avg Range Lvl"] == 0.0]
    ep_match = re.search(r"ep=(\S+)", top["Range"].iloc[0]) if not top.empty else None
    ep_val = ep_match.group(1).split()[0] if ep_match else None
    train_step_ms = top["Proj Avg (ms)"].iloc[0] if not top.empty else None

    if "olmoe" in prof_name:
        model = "olmoe"
    elif "qwen3" in prof_name:
        model = "qwen3"
    else:
        model = "unknown"

    row = {"prof_name": prof_name, "model": model, "ep": ep_val, "train_step_ms": train_step_ms}
    for col in METRIC_COLS:
        match = group[group["Range"] == col]
        row[col] = match["Proj Avg (ms)"].iloc[0] if not match.empty else None

    rows.append(row)

result = pd.DataFrame(rows, columns=["prof_name", "model", "ep", "train_step_ms"] + METRIC_COLS)
result.to_csv(keyranges_dir / "tabulated_keyranges.csv", index=False, float_format="%.1f")

print(result.to_string(index=False))
