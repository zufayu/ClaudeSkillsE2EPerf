"""
Monkey-patch to inject roctx markers into vLLM/ATOM engine for GPU-side timing.

Works across multiprocessing boundaries via builtins.__import__ hook.
Activated by ROCTX_PATCH_ENABLED=1 environment variable.

Usage (automatic via collect_atom_trace.sh --roctx-markers):
  The script installs a .pth file in site-packages that auto-imports this module
  in every Python process, including spawned GPU workers.

What it does:
  - Wraps ModelRunner.run_model (ATOM) or .execute_model (vLLM) with roctx markers
  - Tags: decode_step_N_bs=M / prefill_step_N_bs=M
  - Visible in Kineto traces for GPU-side vs client-side timing comparison

Requires: torch with ROCm (roctx is the ROCm equivalent of NVTX)
"""
import builtins
import functools
import os
import sys


_step_counter = 0
_TARGETS = frozenset({'atom.model_engine.model_runner', 'vllm.worker.model_runner'})


def _log(msg):
    print(msg, file=sys.stderr)


def _wrap_model_fn(original_fn):
    @functools.wraps(original_fn)
    def wrapper(self, *args, **kwargs):
        global _step_counter
        _step_counter += 1
        bs = 0
        is_prefill = False
        if args:
            model_input = args[0]
            for attr in ('input_tokens', 'input_ids'):
                t = getattr(model_input, attr, None)
                if t is not None and hasattr(t, 'shape'):
                    bs = t.shape[0]
                    break
            if bs == 0:
                meta = getattr(model_input, 'seq_group_metadata_list', None)
                if meta:
                    bs = len(meta)
            if hasattr(model_input, 'is_prompt'):
                is_prefill = model_input.is_prompt
            elif hasattr(model_input, 'seq_group_metadata_list'):
                meta = model_input.seq_group_metadata_list
                if meta and hasattr(meta[0], 'is_prompt'):
                    is_prefill = meta[0].is_prompt

        import torch
        phase = 'prefill' if is_prefill else 'decode'
        tag = f"{phase}_step_{_step_counter}_bs={bs}"
        torch.cuda.nvtx.range_push(tag)
        try:
            return original_fn(self, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
    return wrapper


def _patch_class(cls, method_name):
    fn = getattr(cls, method_name, None)
    if fn is None or getattr(fn, '_roctx_patched', False):
        return False
    setattr(cls, method_name, _wrap_model_fn(fn))
    getattr(cls, method_name)._roctx_patched = True
    return True


_original_import = builtins.__import__


def _patched_import(name, *args, **kwargs):
    result = _original_import(name, *args, **kwargs)
    try:
        if 'model_runner' in name:
            for target in _TARGETS:
                mod = sys.modules.get(target)
                if mod:
                    MR = getattr(mod, 'ModelRunner', None)
                    if MR:
                        method = 'run_model' if 'atom' in target else 'execute_model'
                        if _patch_class(MR, method):
                            _log(f"[roctx_patch] Patched {target}.ModelRunner.{method} (pid={os.getpid()})")
    except Exception as e:
        _log(f"[roctx_patch] WARNING: patch failed for {name}: {e}")
    return result


def activate():
    """Install __import__ hook to patch ModelRunner when imported."""
    builtins.__import__ = _patched_import
    _log(f"[roctx_patch] __import__ hook installed (pid={os.getpid()})")


# Auto-activate when ROCTX_PATCH_ENABLED=1
if os.environ.get('ROCTX_PATCH_ENABLED') == '1':
    activate()
