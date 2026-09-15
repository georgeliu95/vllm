# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


SITE_CUSTOMIZE = Path(__file__).resolve().parents[1] / "runtime" / "sitecustomize.py"


def test_lm_eval_compat_drops_removed_swap_space(monkeypatch) -> None:
    received: dict[str, object] = {}

    class LLM:
        def __init__(self, **kwargs) -> None:
            received["llm"] = kwargs

    class EngineArgs:
        def __init__(self, **kwargs) -> None:
            received["engine_args"] = kwargs

    vllm = ModuleType("vllm")
    vllm.LLM = LLM
    entrypoints = ModuleType("vllm.entrypoints")
    chat_utils = ModuleType("vllm.entrypoints.chat_utils")
    entrypoints.chat_utils = chat_utils
    engine = ModuleType("vllm.engine")
    arg_utils = ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = EngineArgs
    renderer = ModuleType("vllm.renderers.hf")
    renderer.resolve_chat_template = object()
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", chat_utils)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    monkeypatch.setitem(sys.modules, "vllm.renderers.hf", renderer)

    spec = importlib.util.spec_from_file_location(
        "_vllm_4o6_sitecustomize", SITE_CUSTOMIZE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._install_lm_eval_compatibility()
    LLM(swap_space=4, dtype="auto")
    EngineArgs(swap_space=4, data_parallel_size=4)

    assert received == {
        "llm": {"dtype": "auto"},
        "engine_args": {"data_parallel_size": 4},
    }
    assert chat_utils.resolve_hf_chat_template is renderer.resolve_chat_template


def test_lm_eval_compat_renders_chat_for_local_completions(monkeypatch) -> None:
    rendered: dict[str, object] = {}

    class LLM:
        def __init__(self, **kwargs) -> None:
            pass

    class EngineArgs:
        def __init__(self, **kwargs) -> None:
            pass

    class TemplateAPI:
        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return "original"

    class LocalCompletionsAPI(TemplateAPI):
        def __init__(self, tokenizer) -> None:
            self.tokenizer = tokenizer
            self.tokenizer_backend = "huggingface"
            self.tokenized_requests = False

    class LocalChatCompletion(LocalCompletionsAPI):
        pass

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            rendered["messages"] = messages
            rendered["kwargs"] = kwargs
            return "rendered prompt"

    vllm = ModuleType("vllm")
    vllm.LLM = LLM
    entrypoints = ModuleType("vllm.entrypoints")
    chat_utils = ModuleType("vllm.entrypoints.chat_utils")
    chat_utils.resolve_hf_chat_template = object()
    entrypoints.chat_utils = chat_utils
    engine = ModuleType("vllm.engine")
    arg_utils = ModuleType("vllm.engine.arg_utils")
    arg_utils.EngineArgs = EngineArgs
    openai_completions = ModuleType("lm_eval.models.openai_completions")
    openai_completions.LocalCompletionsAPI = LocalCompletionsAPI
    lm_eval = ModuleType("lm_eval")
    lm_eval_models = ModuleType("lm_eval.models")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.chat_utils", chat_utils)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    monkeypatch.setitem(sys.modules, "lm_eval", lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.models", lm_eval_models)
    monkeypatch.setitem(
        sys.modules,
        "lm_eval.models.openai_completions",
        openai_completions,
    )

    spec = importlib.util.spec_from_file_location(
        "_vllm_4o6_sitecustomize", SITE_CUSTOMIZE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._install_lm_eval_compatibility()

    tokenizer = Tokenizer()
    completions = LocalCompletionsAPI(tokenizer)
    chat = LocalChatCompletion(tokenizer)
    history = [{"role": "user", "content": "hello", "type": "text"}]

    assert completions.apply_chat_template(history) == "rendered prompt"
    assert chat.apply_chat_template(history) == "original"
    assert rendered == {
        "messages": [{"role": "user", "content": "hello"}],
        "kwargs": {
            "tokenize": False,
            "add_generation_prompt": True,
            "continue_final_message": False,
        },
    }


def test_standard_nvfp4_enables_eval_compat_without_adapter() -> None:
    # Given a standard NVFP4 evaluation process without adaptive runtime mode.
    environment = {"VLLM_NVFP4_EVAL_COMPAT": "1"}
    spec = importlib.util.spec_from_file_location(
        "_vllm_standard_sitecustomize", SITE_CUSTOMIZE
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # When the sitecustomize boundary resolves process features.
    eval_compat_enabled, adapter_enabled = module._runtime_features(environment)

    # Then lm-eval compatibility is enabled without installing 4o6 patches.
    assert eval_compat_enabled is True
    assert adapter_enabled is False
