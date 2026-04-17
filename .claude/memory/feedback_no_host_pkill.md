---
name: No host-level pkill in GitHub Actions workflows
description: Host-level pkill in self-hosted runner workflows kills the runner itself (exit 137). Always use podman exec for container process cleanup.
type: feedback
originSessionId: 49e0962a-ee55-4b7c-9023-e07ec969a46b
---
Never use host-level `pkill -9 -f "python.*vllm\|python.*atom"` in GitHub Actions workflow steps. Also, `podman exec bash -c 'pkill -9 -f "python|..."'` returns exit 137 because the pkill pattern "python" matches the `bash -c` process's own cmdline.

**Why:** (1) Host-level pkill matches the runner's own bash process. (2) Inside podman exec, `pkill -f "python"` matches the bash -c wrapper process cmdline (which contains the word "python" as an argument), killing it and returning 137 to the caller.

**How to apply:** Always add `continue-on-error: true` to GPU cleanup steps. Also change `|| true` to `; true` inside the bash -c quotes (the `|| true` never runs since the bash process itself is killed). The standalone GPU cleanup workflow (`mi355x_gpu_cleanup.yml`) can use host-level kill in its own dedicated step.
