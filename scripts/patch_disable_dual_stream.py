#!/usr/bin/env python3
"""Patch SGLang deepseek_v2.py to disable dual-stream (alt_stream=None).

Usage:
    python3 patch_disable_dual_stream.py          # patch
    python3 patch_disable_dual_stream.py --restore # restore from .bak
"""
import shutil
import sys


def main():
    import sglang.srt.models.deepseek_v2 as m
    path = m.__file__

    if "--restore" in sys.argv:
        bak = path + ".bak"
        shutil.copy2(bak, path)
        print(f"Restored: {path}")
        return

    # Backup
    shutil.copy2(path, path + ".bak")

    with open(path) as f:
        src = f.read()

    old = (
        "self.alt_stream = (\n"
        "            torch.cuda.Stream()\n"
        "            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get()\n"
        "            else None\n"
        "        )"
    )
    new = "self.alt_stream = None  # EXPERIMENT: disabled dual-stream"

    if old not in src:
        print(f"ERROR: pattern not found in {path}")
        sys.exit(1)

    with open(path, "w") as f:
        f.write(src.replace(old, new))

    print(f"Patched: {path}")


if __name__ == "__main__":
    main()
