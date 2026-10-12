# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from contextlib import nullcontext
from typing import Any, Optional

import torch

from megatron.core.inference.contexts.dynamic_context import HAVE_TORCH_MEMORY_SAVER
from megatron.core.inference.shards import build_inference_pg_collection
from megatron.core.inference.unified_memory import create_unified_mempool
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    get_num_microbatches,
    init_num_microbatches_calculator,
    update_num_microbatches,
)
from megatron.training.global_vars import get_train_state
from megatron.training.utils import print_rank_0

if HAVE_TORCH_MEMORY_SAVER:
    from torch_memory_saver import torch_memory_saver

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP  # noqa: F401

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False


def build_rl_inference_model(
    rl_config: Any, args: Any, config: Any, model_provider: Any, model_type: Any
) -> Optional[list]:
    """Build a separate inference model with custom parallelism, if requested.

    Returns ``None`` when inference shares the training model's weights and parallelism.
    """
    # RL inference doesn't support CP; when training uses CP>1, always build a
    # separate CP=1 inference model (CP ranks become extra DP replicas, dp*=cp).
    force_cp1_inference_model = args.context_parallel_size > 1
    build_separate_model = (
        rl_config.rl_inference_tensor_model_parallel_size is not None
        or rl_config.rl_inference_pipeline_model_parallel_size is not None
        or rl_config.rl_inference_expert_model_parallel_size is not None
        or rl_config.rl_inference_expert_tensor_model_parallel_size is not None
        or force_cp1_inference_model
    )
    if not build_separate_model:
        # Validate: offloading flag requires a separate inference model
        if rl_config.rl_offload_inference_model_weights:
            raise ValueError(
                "--rl-offload-inference-model-weights requires a separate inference model. "
                "This flag is only useful when doing refit since the weights are shared with "
                "the training model."
            )
        return None

    print_rank_0(
        "Building separate RL inference model with custom parallelism: "
        f"TP={rl_config.rl_inference_tensor_model_parallel_size}, "
        f"PP={rl_config.rl_inference_pipeline_model_parallel_size}, "
        f"CP={1 if force_cp1_inference_model else None}, "
        f"EP={rl_config.rl_inference_expert_model_parallel_size}, "
        f"ExptTP={rl_config.rl_inference_expert_tensor_model_parallel_size}"
    )
    inference_pg_collection = build_inference_pg_collection(
        args.world_size,
        tp_size=rl_config.rl_inference_tensor_model_parallel_size,
        pp_size=rl_config.rl_inference_pipeline_model_parallel_size,
        cp_size=1 if force_cp1_inference_model else None,
        ep_size=rl_config.rl_inference_expert_model_parallel_size,
        expt_tp_size=rl_config.rl_inference_expert_tensor_model_parallel_size,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
    )

    # Build an isolated inference config so training config remains unchanged
    inference_config = copy.deepcopy(config)
    if rl_config.rl_inference_tensor_model_parallel_size is not None:
        inference_config.tensor_model_parallel_size = (
            rl_config.rl_inference_tensor_model_parallel_size
        )
    if rl_config.rl_inference_pipeline_model_parallel_size is not None:
        inference_config.pipeline_model_parallel_size = (
            rl_config.rl_inference_pipeline_model_parallel_size
        )
    if force_cp1_inference_model:
        inference_config.context_parallel_size = 1
    if rl_config.rl_inference_expert_model_parallel_size is not None:
        inference_config.expert_model_parallel_size = (
            rl_config.rl_inference_expert_model_parallel_size
        )
    if rl_config.rl_inference_expert_tensor_model_parallel_size is not None:
        inference_config.expert_tensor_parallel_size = (
            rl_config.rl_inference_expert_tensor_model_parallel_size
        )

    # Optionally allocate the RL inference model weights from a unified virtual memory (UVM)
    # mempool so we can prefetch weights to CPU when idle while keeping CUDA-graph-safe pointers.
    # Alternatively, use torch_memory_saver to offload the weights to CPU when idle.
    uvm_mempool = None
    uvm_level = rl_config.rl_inference_model_unified_memory_level
    if uvm_level and uvm_level > 0:
        uvm_mempool = create_unified_mempool()

    # Determine which context manager to use for model allocation
    # Use torch_memory_saver if offloading is requested but UVM is not enabled
    use_torch_saver_for_inference_model = (
        rl_config.rl_offload_inference_model_weights and uvm_level == 0 and HAVE_TORCH_MEMORY_SAVER
    )
    if use_torch_saver_for_inference_model:
        # Use torch_memory_saver for offloading - allocate within a tagged region
        model_alloc_ctx = torch_memory_saver.region(
            tag="rl_inference_model", enable_cpu_backup=True
        )
    elif uvm_mempool is not None:
        model_alloc_ctx = torch.cuda.use_mem_pool(uvm_mempool)
    else:
        model_alloc_ctx = nullcontext()

    from megatron.training.training import get_model

    with model_alloc_ctx:
        inference_model = get_model(
            model_provider,
            model_type,
            wrap_with_ddp=False,
            pg_collection=inference_pg_collection,
            config=inference_config,
        )
    inference_model[0].eval()
    return inference_model


def load_reference_state_dict(model: list, checkpointing_context: dict, args: Any) -> dict:
    """Load the pretrained checkpoint's weights as the RL reference state dict.

    With --skip-train the current weights are the reference; otherwise the pretrained
    checkpoint is loaded, snapshotted to CPU, and the RL training checkpoint reloaded.
    """
    from megatron.training.checkpointing import load_checkpoint

    if args.skip_train:
        # In inference-only mode, use current weights as reference.
        print_rank_0("> RL inference-only: using current weights as reference.")
        return {k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()}

    print_rank_0("> Loading pretrained checkpoint for reference weights in RL training...")
    load, finetune, no_load_optim = args.load, args.finetune, args.no_load_optim
    args.no_load_optim = True

    # Load pretrained checkpoint
    args.load = None
    args.finetune = True
    load_checkpoint(
        model,
        None,  # Don't load optimizer state
        None,  # Don't load scheduler state
        checkpointing_context=checkpointing_context,
        restore_training_state=True,
        skip_load_to_model_and_opt=HAVE_FSDP2
        and getattr(args, "use_torch_fsdp2", False)
        and args.ckpt_format == "torch_dist",
    )
    ref_state_dict = {
        k: (v.cpu() if v is not None else v) for k, v in model[0].state_dict().items()
    }

    # Reload RL training checkpoint weights
    args.load = load
    args.finetune = finetune
    print_rank_0("> Reloading RL training checkpoint...")
    load_checkpoint(
        model,
        None,
        None,
        checkpointing_context=checkpointing_context,
        restore_training_state=True,
        skip_load_to_model_and_opt=HAVE_FSDP2
        and getattr(args, "use_torch_fsdp2", False)
        and args.ckpt_format == "torch_dist",
    )
    update_num_microbatches(consumed_samples=get_train_state().consumed_train_samples, verbose=True)

    args.no_load_optim = no_load_optim
    return ref_state_dict


def reinitialize_grpo_microbatch_calculator(args: Any, data_parallel_size: int) -> None:
    """Reinitialize the microbatch calculator with the GRPO configuration."""
    # IMPORTANT FIX: For RL training, reinitialize the microbatch calculator with the correct
    # configuration.
    print_rank_0("> Reinitializing microbatch calculator for GRPO training...")
    # First destroy the existing calculator
    destroy_num_microbatches_calculator()
    # Then initialize with the correct perform_rl_step=True context
    init_num_microbatches_calculator(
        rank=args.rank,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=data_parallel_size,
        decrease_batch_size_if_needed=args.decrease_batch_size_if_needed,
        step_batch_size_schedule=args.step_batch_size_schedule,
        seq_length=args.seq_length,
    )
    print_rank_0(f"> GRPO training: num_microbatches set to {get_num_microbatches()}")
