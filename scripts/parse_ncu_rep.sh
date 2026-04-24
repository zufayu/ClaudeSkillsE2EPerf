#!/usr/bin/env bash
# Parse .ncu-rep to extract kernel names and counts
# Usage: bash parse_ncu_rep.sh /tmp/ncu_trtllm.ncu-rep
set -euo pipefail

REP=${1:-/tmp/ncu_trtllm.ncu-rep}

echo "=== File ==="
ls -lh "$REP"

echo ""
echo "=== Total profiled kernels ==="
ncu -i "$REP" --csv --page details 2>/dev/null | tail -n +2 | wc -l

echo ""
echo "=== Unique kernel names (count | name) ==="
ncu -i "$REP" --csv --page details 2>/dev/null | tail -n +2 | \
    python3 -c "
import sys, csv
reader = csv.reader(sys.stdin)
names = []
for row in reader:
    if len(row) > 3:
        names.append(row[4])  # kernel name column (ID=0, PID=1, ProcName=2, Host=3, KernelName=4)
from collections import Counter
for name, count in Counter(names).most_common():
    print(f'  {count:6d}  {name[:100]}')
print(f'  ------')
print(f'  {len(names):6d}  TOTAL')
print(f'  {len(set(names)):6d}  UNIQUE TYPES')
"

echo ""
echo "=== Kernel launch sequence (first 50) ==="
ncu -i "$REP" --csv --page details 2>/dev/null | tail -n +2 | \
    python3 -c "
import sys, csv
reader = csv.reader(sys.stdin)
for i, row in enumerate(reader):
    if i >= 50: break
    if len(row) > 3:
        print(f'  {i:4d}  {row[4][:80]}')
"

echo ""
echo "=== Kernel launch sequence (last 50) ==="
ncu -i "$REP" --csv --page details 2>/dev/null | tail -n +2 | \
    python3 -c "
import sys, csv
rows = list(csv.reader(sys.stdin))
total = len(rows)
start = max(0, total - 50)
for i, row in enumerate(rows[start:], start):
    if len(row) > 3:
        print(f'  {i:4d}  {row[4][:80]}')
print(f'  TOTAL: {total}')
"
