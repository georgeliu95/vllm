# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-install the pinned vLLM 4o6/SVDQ adapter when explicitly enabled."""

import os
from functools import wraps
from typing import Any, Mapping

EVAL_COMPAT_ENV = "VLLM_NVFP4_EVAL_COMPAT"
ADAPTER_MODE_ENV = "VLLM_NVFP4_4O6_MODE"


def _render_huggingface_chat(
    tokenizer: Any,
    chat_history: list[dict[str, Any]],
    add_generation_prompt: bool,
) -> str:
    """Render lm-eval's JSON-chat representation for a completions API."""
    messages = [
        {key: value for key, value in message.items() if key != "type"}
        for message in chat_history
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt,
    )


def _install_no_thinking_chat_template() -> None:
    """Default Qwen chat rendering to non-thinking for fixed evals."""
    from transformers import PreTrainedTokenizerBase

    original = PreTrainedTokenizerBase.apply_chat_template
    if getattr(original, "_vllm_4o6_no_thinking", False):
        return

    @wraps(original)
    def apply_chat_template(self, *args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original(self, *args, **kwargs)

    apply_chat_template._vllm_4o6_no_thinking = True
    PreTrainedTokenizerBase.apply_chat_template = apply_chat_template


def _install_lm_eval_compatibility() -> None:
    """Bridge lm-eval 0.4.10 to the pinned vLLM 0.24 renderer API."""
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints import chat_utils

    def drop_removed_arguments(class_type) -> None:
        original_init = class_type.__init__
        if getattr(original_init, "_vllm_4o6_lm_eval_compat", False):
            return

        @wraps(original_init)
        def compatible_init(self, *args, **kwargs):
            # lm-eval 0.4.10 still forwards this removed vLLM argument.
            kwargs.pop("swap_space", None)
            return original_init(self, *args, **kwargs)

        compatible_init._vllm_4o6_lm_eval_compat = True
        class_type.__init__ = compatible_init

    drop_removed_arguments(LLM)
    drop_removed_arguments(EngineArgs)

    if not hasattr(chat_utils, "resolve_hf_chat_template"):
        from vllm.renderers.hf import resolve_chat_template

        chat_utils.resolve_hf_chat_template = resolve_chat_template

    try:
        from lm_eval.models.openai_completions import LocalCompletionsAPI
    except ImportError:
        return

    original_apply_chat_template = LocalCompletionsAPI.apply_chat_template
    if getattr(original_apply_chat_template, "_vllm_4o6_completions_chat", False):
        return

    @wraps(original_apply_chat_template)
    def completions_apply_chat_template(
        self,
        chat_history: list[dict[str, Any]],
        add_generation_prompt: bool = True,
    ):
        # lm-eval 0.4.10 otherwise serializes this as JSON when
        # tokenized_requests=False.  /v1/completions requires a rendered
        # string, while the chat-completions subclass still requires messages.
        if (
            type(self) is LocalCompletionsAPI
            and self.tokenizer_backend == "huggingface"
            and not self.tokenized_requests
        ):
            return _render_huggingface_chat(
                self.tokenizer,
                chat_history,
                add_generation_prompt,
            )
        return original_apply_chat_template(
            self,
            chat_history,
            add_generation_prompt=add_generation_prompt,
        )

    completions_apply_chat_template._vllm_4o6_completions_chat = True
    LocalCompletionsAPI.apply_chat_template = completions_apply_chat_template


def _runtime_features(environment: Mapping[str, str]) -> tuple[bool, bool]:
    """Resolve lm-eval compatibility and adaptive adapter installation."""
    adapter_enabled = bool(environment.get(ADAPTER_MODE_ENV))
    eval_compat_enabled = bool(environment.get(EVAL_COMPAT_ENV)) or adapter_enabled
    return eval_compat_enabled, adapter_enabled


eval_compat_enabled, adapter_enabled = _runtime_features(os.environ)
if eval_compat_enabled:
    if os.environ.get("VLLM_NVFP4_DISABLE_THINKING"):
        _install_no_thinking_chat_template()
    _install_lm_eval_compatibility()

if adapter_enabled:
    from vllm_adapter.install import install

    install()

    if os.environ.get("B300_TRT_QUANT") == "1" and os.environ.get(
        ADAPTER_MODE_ENV
    ) in {"4o6", "4o6-svdq"}:
        from trt_quant.dense import install as install_fused_quantization

        install_fused_quantization()
