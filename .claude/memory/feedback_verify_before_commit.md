---
name: Verify commands before committing
description: Must test/verify commands locally or check docs before committing to workflows — never commit untested code based on assumptions
type: feedback
---

Never commit untested commands or untested syntax to workflows or scripts. If a command hasn't been verified to work, do NOT commit it.

**Why:** User flagged that I committed `nsys export --type perfetto` without testing — it's wrong. The correct format is `--type jsonlines` outputting `.json`. This wastes CI runs and the user's time.

**How to apply:**
1. Before committing shell commands, verify syntax by checking `--help` output or docs
2. If running on a remote machine, at least verify the command exists and the flag is valid
3. When unsure about exact syntax, ask the user or flag uncertainty before committing
4. Especially critical for workflow files that trigger CI runs on expensive GPU machines
