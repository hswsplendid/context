"""Model and tokenizer loading utilities."""

import os
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

logger = logging.getLogger(__name__)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer based on config.

    Uses device_map='auto' for multi-GPU sharding.
    Respects CUDA_VISIBLE_DEVICES for GPU selection.
    """
    dtype = DTYPE_MAP.get(config.torch_dtype, torch.bfloat16)

    # Limit visible GPUs if max_gpu_count is set
    if config.max_gpu_count is not None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible:
            gpu_ids = visible.split(",")[: config.max_gpu_count]
        else:
            gpu_ids = [str(i) for i in range(config.max_gpu_count)]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        logger.info(f"Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

    logger.info(f"Loading tokenizer from {config.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        f"Loading model from {config.model_path} "
        f"(dtype={config.torch_dtype}, device_map={config.device_map})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_path,
        torch_dtype=dtype,
        device_map=config.device_map,
        trust_remote_code=True,
    )
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {num_params / 1e9:.1f}B parameters")

    return model, tokenizer
