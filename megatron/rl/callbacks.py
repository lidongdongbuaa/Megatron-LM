# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any, Optional

from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.rl.training_setup import (
    build_rl_inference_model,
    load_reference_state_dict,
    reinitialize_grpo_microbatch_calculator,
)
from megatron.training.callbacks import Callback, CallbackContext
from megatron.training.global_vars import get_args


class RLSetupCallback(Callback):
    """Builds the RL loop's state at ``on_data_init_start`` and keeps it for the rollout hooks.

    After the event: ``model`` and ``optimizer`` are the loop's, ``inference_model`` is the
    separately parallelized rollout model or None when inference shares the training model, and
    ``ref_state_dict`` holds the reference weights, or None in eval-only runs (no iterations).
    """

    def __init__(self, rl_config: Any, *, model_provider: Any, model_type: Any) -> None:
        self._rl_config = rl_config
        self._model_provider = model_provider
        self._model_type = model_type
        self.model: Optional[list] = None
        self.optimizer: Any = None
        self.inference_model: Optional[list] = None
        self.ref_state_dict: Optional[dict] = None

    def on_data_init_start(self, context: CallbackContext) -> None:
        from megatron.training.training import update_train_iters

        args = get_args()
        self.model = context.model
        self.optimizer = context.optimizer
        if self.optimizer is None:
            # The loop may run without an optimizer (inference-only), so derive the iteration
            # count that the scheduler build would otherwise have set.
            update_train_iters(args)
        self.inference_model = build_rl_inference_model(
            self._rl_config,
            args,
            get_model_config(self.model[0]),
            self._model_provider,
            self._model_type,
        )
        if (args.train_iters or 0) > 0:
            # GRPO loop state; eval-only runs (no iterations) evaluate the loaded weights directly.
            self.ref_state_dict = load_reference_state_dict(
                self.model, context.checkpointing_context, args
            )
            # RL runs on the plain MPU groups: the loop's data-parallel size for them.
            reinitialize_grpo_microbatch_calculator(args, mpu.get_data_parallel_world_size())
