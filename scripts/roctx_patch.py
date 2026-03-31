"""
Monkey-patch to inject roctx markers into vLLM/ATOM engine for GPU-side timing.

Usage: Import this BEFORE starting the server, or use sitecustomize.
  python3 -c "import roctx_patch" && python3 -m atom.entrypoints.openai_server ...
  OR: PYTHONPATH=/path/to/scripts python3 -m atom.entrypoints.openai_server ...

What it does:
  - Wraps ModelRunner.execute_model with roctx push/pop markers
  - Adds "decode_step" / "prefill_step" annotations visible in Kineto traces
  - Allows comparing GPU-side step duration vs client-side ITL (25ms vs 21ms gap)

Requires: torch with ROCm (roctx is the ROCm equivalent of NVTX)
"""
import functools
import torch


_step_counter = 0


def _wrap_execute_model(original_fn):
    @functools.wraps(original_fn)
    def wrapper(self, *args, **kwargs):
        global _step_counter
        _step_counter += 1
        bs = 0
        is_prefill = False
        if args:
            model_input = args[0]
            # Batch size from input tensor
            for attr in ('input_tokens', 'input_ids'):
                t = getattr(model_input, attr, None)
                if t is not None and hasattr(t, 'shape'):
                    bs = t.shape[0]
                    break
            if bs == 0:
                meta = getattr(model_input, 'seq_group_metadata_list', None)
                if meta:
                    bs = len(meta)
            # Detect prefill vs decode
            if hasattr(model_input, 'is_prompt'):
                is_prefill = model_input.is_prompt
            elif hasattr(model_input, 'seq_group_metadata_list'):
                meta = model_input.seq_group_metadata_list
                if meta and hasattr(meta[0], 'is_prompt'):
                    is_prefill = meta[0].is_prompt

        phase = 'prefill' if is_prefill else 'decode'
        tag = f"{phase}_step_{_step_counter}_bs={bs}"
        torch.cuda.nvtx.range_push(tag)
        try:
            result = original_fn(self, *args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()
        return result
    return wrapper


def patch():
    """Apply roctx patches to vLLM/ATOM ModelRunner."""
    patched = False

    # Try vLLM ModelRunner
    try:
        from vllm.worker.model_runner import ModelRunner
        if not getattr(ModelRunner.execute_model, '_roctx_patched', False):
            ModelRunner.execute_model = _wrap_execute_model(ModelRunner.execute_model)
            ModelRunner.execute_model._roctx_patched = True
            patched = True
            print("[roctx_patch] Patched vllm.worker.model_runner.ModelRunner.execute_model")
    except ImportError:
        pass

    # Try ATOM ModelRunner (atom.model_engine.model_runner)
    try:
        from atom.model_engine.model_runner import ModelRunner as AtomModelRunner
        if not getattr(AtomModelRunner.execute_model, '_roctx_patched', False):
            AtomModelRunner.execute_model = _wrap_execute_model(AtomModelRunner.execute_model)
            AtomModelRunner.execute_model._roctx_patched = True
            patched = True
            print("[roctx_patch] Patched atom.worker.model_runner.ModelRunner.execute_model")
    except ImportError:
        pass

    if not patched:
        print("[roctx_patch] WARNING: Could not find ModelRunner to patch")

    return patched


# Auto-patch on import
patch()
