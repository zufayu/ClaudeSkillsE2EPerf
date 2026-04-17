---
name: Do not upload large trace files
description: Large profiling trace files (.json.gz from PyTorch Profiler) should never be committed to git repo
type: feedback
---

Large trace files (.json.gz from PyTorch/Torch Profiler) should NOT be uploaded/committed to the git repo.

**Why:** Trace files are typically 100MB-1GB+, exceeding GitHub's 100MB file size limit. They also bloat the repo and are only needed locally for analysis.

**How to apply:**
- Workflow git add steps must exclude `.json.gz` files
- Only commit derived artifacts: `summary.md`, `result_profiled_*.json` (e2e metrics), `decode_walltime_*.csv`, logs
- Trace files stay on the machine for local analysis / manual copy
