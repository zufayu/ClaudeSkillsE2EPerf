"""
Monkey-patch to inject roctx markers into vLLM/ATOM engine for GPU-side timing.

Works across multiprocessing boundaries (spawn) via sys.meta_path import hook.
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
import functools
import importlib
import os
import sys


_step_counter = 0


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


class _RoctxImportHook:
    """Import hook: patches ModelRunner when atom/vllm model_runner is imported."""
    _active = True

    def find_module(self, name, path=None):
        if self._active and name in ('atom.model_engine.model_runner', 'vllm.worker.model_runner'):
            return self
        return None

    def load_module(self, name):
        self._active = False
        try:
            mod = importlib.import_module(name)
        finally:
            self._active = True
        MR = getattr(mod, 'ModelRunner', None)
        if MR:
            method = 'run_model' if 'atom' in name else 'execute_model'
            if _patch_class(MR, method):
                _log(f"[roctx_patch] Patched {name}.ModelRunner.{method} (hook, pid={os.getpid()})")
        return mod


def activate():
    """Activate roctx patching: try direct patch, fall back to import hook."""
    patched = False
    try:
        from atom.model_engine.model_runner import ModelRunner as AtomMR
        if _patch_class(AtomMR, 'run_model'):
            _log(f"[roctx_patch] Patched ATOM ModelRunner.run_model (direct, pid={os.getpid()})")
            patched = True
    except (ImportError, AttributeError):
        pass
    try:
        from vllm.worker.model_runner import ModelRunner
        if _patch_class(ModelRunner, 'execute_model'):
            _log(f"[roctx_patch] Patched vLLM ModelRunner.execute_model (direct, pid={os.getpid()})")
            patched = True
    except (ImportError, AttributeError):
        pass

    if not patched:
        sys.meta_path.insert(0, _RoctxImportHook())
        _log(f"[roctx_patch] Import hook installed (pid={os.getpid()})")


# Auto-activate when ROCTX_PATCH_ENABLED=1
if os.environ.get('ROCTX_PATCH_ENABLED') == '1':
    activate()
