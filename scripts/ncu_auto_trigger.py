"""Auto-trigger cudaProfilerStart/Stop after a delay.

Drop into site-packages as sitecustomize.py or set PYTHONSTARTUP.
Every Python process (including multiprocessing.spawn workers) loads this.

Usage:
    NCU_TRIGGER=1 NCU_DELAY=300 NCU_DURATION=60 ncu --profile-from-start off ... python3 ...

Env vars:
    NCU_TRIGGER: set to "1" to enable (otherwise no-op)
    NCU_DELAY: seconds to wait before cudaProfilerStart (default: 300)
    NCU_DURATION: seconds of profiling before cudaProfilerStop (default: 60)
"""
import os

if os.environ.get("NCU_TRIGGER") == "1":
    import threading

    def _ncu_delayed_profile():
        import time
        import ctypes

        delay = int(os.environ.get("NCU_DELAY", "300"))
        duration = int(os.environ.get("NCU_DURATION", "60"))

        time.sleep(delay)

        try:
            libcudart = ctypes.CDLL("libcudart.so")
            ret = libcudart.cudaProfilerStart()
            pid = os.getpid()
            print(f"[NCU_TRIGGER pid={pid}] cudaProfilerStart() returned {ret}", flush=True)

            time.sleep(duration)

            ret = libcudart.cudaProfilerStop()
            print(f"[NCU_TRIGGER pid={pid}] cudaProfilerStop() returned {ret}", flush=True)
        except Exception as e:
            print(f"[NCU_TRIGGER pid={os.getpid()}] error: {e}", flush=True)

    t = threading.Thread(target=_ncu_delayed_profile, daemon=True)
    t.start()
