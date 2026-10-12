# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.rl import callbacks as rl_callbacks
from megatron.rl.callbacks import RLSetupCallback
from megatron.training import training
from megatron.training.callbacks import CallbackContext
from megatron.training.config import RLConfig


@pytest.mark.parametrize("train_iters, optimizer", [(0, None), (5, "optimizer")])
def test_setup_callback(monkeypatch, train_iters, optimizer):
    """on_data_init_start keeps the loop's model and optimizer, derives train_iters when there is
    no optimizer, builds the inference model, and loads the reference weights and reinitializes
    the GRPO calculator only when there are iterations."""
    calls = []
    args = SimpleNamespace(train_iters=train_iters)
    monkeypatch.setattr(rl_callbacks, "get_args", lambda: args)
    monkeypatch.setattr(rl_callbacks, "get_model_config", lambda chunk: f"config({chunk})")
    monkeypatch.setattr(rl_callbacks.mpu, "get_data_parallel_world_size", lambda: 4)
    monkeypatch.setattr(training, "update_train_iters", lambda a: calls.append(("iters", a)))
    monkeypatch.setattr(
        rl_callbacks,
        "build_rl_inference_model",
        lambda *a: calls.append(("inference", a)) or "inference_model",
    )
    monkeypatch.setattr(
        rl_callbacks, "load_reference_state_dict", lambda *a: calls.append(("ref", a)) or {"w": 1}
    )
    monkeypatch.setattr(
        rl_callbacks, "reinitialize_grpo_microbatch_calculator", lambda *a: calls.append(("mb", a))
    )
    cfg = RLConfig()
    callback = RLSetupCallback(cfg, model_provider="provider", model_type="type")

    callback.on_data_init_start(
        CallbackContext(model=["chunk"], optimizer=optimizer, checkpointing_context={"ckpt": 1})
    )

    expected = [("iters", args)] if optimizer is None else []
    expected.append(("inference", (cfg, args, "config(chunk)", "provider", "type")))
    if train_iters:
        expected += [("ref", (["chunk"], {"ckpt": 1}, args)), ("mb", (args, 4))]
    assert calls == expected
    assert (callback.model, callback.optimizer) == (["chunk"], optimizer)
    assert callback.inference_model == "inference_model"
    assert callback.ref_state_dict == ({"w": 1} if train_iters else None)
