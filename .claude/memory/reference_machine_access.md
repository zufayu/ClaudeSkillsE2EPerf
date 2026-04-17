---
name: Machine access via GitHub Actions runners
description: How to run commands on B200/355X — use GitHub Actions workflow_dispatch via API, runners are self-hosted on each machine
type: reference
---

## Access Method: GitHub Actions Self-Hosted Runners

Both B200 and MI355X have self-hosted GitHub Actions runners. To run commands remotely:

**Runners (check status via API):**
- `b200-runner` — labels: `[self-hosted, Linux, X64, b200]`
- `mi355x-runner` — labels: `[self-hosted, Linux, X64, mi355x]`

**How to trigger a workflow:**
```bash
cd /home/kqian/ClaudeSkillsE2EPerf
GH_TOKEN=$(git remote get-url origin | grep -oP 'ghp_[^@]+')
curl -s -X POST \
  -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/zufayu/ClaudeSkillsE2EPerf/actions/workflows/<WORKFLOW>.yml/dispatches \
  -d '{"ref":"main","inputs":{...}}'
```

**How to check runner status:**
```bash
curl -s -H "Authorization: token $GH_TOKEN" \
  https://api.github.com/repos/zufayu/ClaudeSkillsE2EPerf/actions/runners \
  | python3 -c "import sys,json; [print(f'{r[\"name\"]} ({r[\"status\"]})') for r in json.load(sys.stdin).get('runners',[])]"
```

**How to check run status / get logs:**
```bash
# Latest run for a workflow
curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/zufayu/ClaudeSkillsE2EPerf/actions/runs?workflow_id=<WORKFLOW>.yml&per_page=1" \
  | python3 -c "import sys,json; r=json.load(sys.stdin)['workflow_runs'][0]; print(f'#{r[\"id\"]} {r[\"status\"]} ({r[\"conclusion\"] or \"running\"})')"

# Get job logs
curl -s -H "Authorization: token $GH_TOKEN" \
  "https://api.github.com/repos/zufayu/ClaudeSkillsE2EPerf/actions/runs/<RUN_ID>/jobs" \
  | python3 -c "import sys,json; [print(f'{j[\"name\"]}: {j[\"status\"]}') for j in json.load(sys.stdin)['jobs']]"
```

**Machine details:**
- B200 tunnel name: `hungry-hippo-fin-03-2` (stable)
- MI355X tunnel name: `my-gpu-41` (may change — if runner offline, remind user)
- SGLang container on B200: `docker exec -it zufa_sglang bash`
- B200 repo path: `/home/ubuntu/zufa/ClaudeSkillsE2EPerf/` (symlink `/SFS-aGqda6ct/ubuntu/zufa/ClaudeSkillsE2EPerf`)
- Large files (traces, nsys-rep) must be manually copied by user — part of workflow
