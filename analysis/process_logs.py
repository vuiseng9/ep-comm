#!/usr/bin/env python3
"""
Extract key metrics from logs/* and print a summary table.
Each row corresponds to one log file (keyed by basename without .log).

Extracted from lines of the form:
  |R0| ─────────  <steps> steps, <ep> ep ─── <ms> ms, <gb> gb ─── <loss> (loss)
"""

import re
import sys
import glob
from pathlib import Path

import pandas as pd

ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

# Match the metric line (after ANSI stripping); loss may be numeric or 'nan'
METRIC_RE = re.compile(
    r'\|R0\|'
    r'.*?(\d+)\s+steps,'
    r'\s+([\d.]+)\s+ep'
    r'\s+───\s+([\d.]+)\s+ms,'
    r'\s+([\d.]+)\s+gb'
    r'\s+───\s+([\d.]+|nan)\s+\(loss\)',
    re.IGNORECASE,
)


def parse_log(filepath: Path) -> dict | None:
    """Return metrics from the last reported step in a log file."""
    last = None
    with open(filepath, errors='replace') as f:
        for raw in f:
            line = ANSI_ESCAPE.sub('', raw)
            m = METRIC_RE.search(line)
            if m:
                loss_raw = m.group(5)
                last = {
                    'name': filepath.stem,
                    'steps': int(m.group(1)),
                    'epoch': float(m.group(2)),
                    'elapsed_ms': float(m.group(3)),
                    'memory_gb': float(m.group(4)),
                    'loss': float('nan') if loss_raw.lower() == 'nan' else float(loss_raw),
                }
    return last


def main():
    log_dir = Path(__file__).parent / 'logs'
    patterns = sys.argv[1:] or [str(log_dir / '*.log')]

    rows = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            result = parse_log(Path(path))
            if result:
                rows.append(result)
            else:
                rows.append({'name': Path(path).stem, 'steps': None,
                             'epoch': None, 'elapsed_ms': None,
                             'memory_gb': None, 'loss': None})

    if not rows:
        print('No log files found.')
        return

    df = pd.DataFrame(rows).rename(columns={'elapsed_ms': 'ms', 'memory_gb': 'gb'})
    df = df.set_index('name')
    out_csv = log_dir / 'summary.csv'
    df.to_csv(out_csv)
    print(f'Written: {out_csv}')

    fmt1 = lambda x: f'{x:.1f}'
    fmt4 = lambda x: f'{x:.4f}' if pd.notna(x) else 'NaN'
    print(df.to_string(formatters={
        'steps': lambda x: f'{int(x)}' if pd.notna(x) else '',
        'epoch': fmt4,
        'ms': fmt1,
        'gb': fmt1,
        'loss': fmt4,
    }))


if __name__ == '__main__':
    main()
