# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Qwen2.5-VLTS model compatible with HuggingFace weights."""

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from vllm.attention.backends.registry import _Backend
from vllm.config import VllmConfig
from vllm.distributed import parallel_state
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_and_mul_fn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargs
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from typing_extensions import Annotated

from .interfaces import (
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)
from .qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from .utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)

logger = init_logger(__name__)


# ===== Time-Series Input Schema ===== #

class Qwen2_5_VLTimeSeriesInputs(TensorSchema):
    """
    Time-series input schema for Qwen2.5-VLTS model.

    Dimensions:
        - nv: Number of time-series values (total across all streams)
        - dp: Digit pad length (1 + ts_digit_pad)
        - ns: Number of streams
    """
    type: Literal["time_series"]

    ts_digits: Annotated[
        torch.Tensor,
        TensorShape("nv", "dp"),
    ]

    ts_digit_lengths: Annotated[
        torch.Tensor,
        TensorShape("nv"),
    ]

    datetimes: Annotated[
        torch.Tensor,
        TensorShape("nv", 6),
    ]

    time_series_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("ns", 3),
    ]

    batch_ts_seq_end_idxs: Annotated[
        torch.Tensor,
        TensorShape("batch_plus_1"),
    ]


# ===== Configuration Classes ===== #

class Qwen2_5_VLTimeSeriesConfig(PretrainedConfig):
    """Configuration for the time-series encoder component."""

    model_type = "qwen2_5_vl"

    def __init__(
        self,
        depth: int = 1,
        hidden_size: int = 3584,
        hidden_act: str = "silu",
        intermediate_size: int = 18944,
        num_heads: int = 32,
        num_attention_heads: int = 28,
        in_channels: int = 3,
        patch_size: int = 4,
        tokens_per_second: int = 4,
        window_size: int = 112,
        out_hidden_size: int = 3584,
        initializer_range: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_attention_heads = num_attention_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.tokens_per_second = tokens_per_second
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.initializer_range = initializer_range
        self._attn_implementation = 'sdpa'


class Qwen2_5_VLTSConfig(Qwen2_5_VLConfig):
    """Extended Qwen2.5-VL config with time-series support."""

    model_type = "qwen2_5_vl"

    def __init__(
        self,
        time_series_config: Union[Qwen2_5_VLTimeSeriesConfig, Dict[str, Any], None] = None,
        time_series_token_id: int = 151666,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(time_series_config, dict):
            self.time_series_config = Qwen2_5_VLTimeSeriesConfig(**time_series_config)
        elif time_series_config is None:
            self.time_series_config = Qwen2_5_VLTimeSeriesConfig()
        else:
            self.time_series_config = time_series_config

        self.time_series_token_id = time_series_token_id


# ===== Time-Series Preprocessor ===== #

class Qwen2_5_TimeSeriesPreprocessor:
    """Handles time-series preprocessing configuration."""

    def __init__(
        self,
        ts_patch_size: int = 4,
        ts_merge_size: int = 1,
        ts_digit_pad: int = 12,
        ts_dec_digits: int = 4,
        ts_datetime_format: str = "%Y-%m-%d %H:%M:%S",
        **kwargs,
    ):
        self.ts_patch_size = ts_patch_size
        self.ts_merge_size = ts_merge_size
        self.ts_digit_pad = ts_digit_pad
        self.ts_dec_digits = ts_dec_digits
        self.ts_datetime_format = ts_datetime_format

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs) -> "Qwen2_5_TimeSeriesPreprocessor":
        """Load time-series preprocessor config from directory."""
        config_path = os.path.join(pretrained_path, "ts_preprocessor_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return cls(**config)
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_patch_size": self.ts_patch_size,
            "ts_merge_size": self.ts_merge_size,
            "ts_digit_pad": self.ts_digit_pad,
            "ts_dec_digits": self.ts_dec_digits,
            "ts_datetime_format": self.ts_datetime_format,
        }


# ===== Embedding Components (adapted from your model.py) ===== #

class DigitwiseFloatEmbedding(nn.Module):
    """Digitwise float embedding for time-series values."""

    def __init__(
        self,
        out_dim: int = 768,
        digit_pad: int = 12,
        max_position: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.digit_pad = digit_pad
        self.max_position = max_position

        # Embeddings for different components
        self.sign_embedding = nn.Embedding(2, out_dim)  # 0: positive, 1: negative
        self.digit_embedding = nn.Embedding(10, out_dim)  # 0-9 digits
        self.position_embedding = nn.Embedding(max_position, out_dim)
        self.pad_embedding = nn.Embedding(1, out_dim)  # for padding positions

        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, digits: torch.Tensor, digit_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            digits: (batch_size, 1 + digit_pad) - sign + digit tokens
            digit_lengths: (batch_size,) - actual number of digits per sample

        Returns:
            embeddings: (batch_size, out_dim)
        """
        batch_size = digits.size(0)
        device = digits.device

        # Extract sign and digits
        signs = digits[:, 0]  # (batch_size,)
        digit_tokens = digits[:, 1:]  # (batch_size, digit_pad)

        # Sign embedding
        sign_emb = self.sign_embedding(signs)  # (batch_size, out_dim)

        # Digit embeddings with positional encoding
        positions = torch.arange(self.digit_pad, device=device).unsqueeze(0).expand(batch_size, -1)

        # Create mask for valid digits
        digit_mask = positions < digit_lengths.unsqueeze(1)  # (batch_size, digit_pad)

        # Get embeddings
        digit_embs = self.digit_embedding(digit_tokens)  # (batch_size, digit_pad, out_dim)
        pos_embs = self.position_embedding(positions)  # (batch_size, digit_pad, out_dim)
        pad_embs = self.pad_embedding.weight[0].unsqueeze(0).unsqueeze(0)  # (1, 1, out_dim)

        # Combine digit and positional embeddings
        combined_embs = digit_embs + pos_embs  # (batch_size, digit_pad, out_dim)

        # Apply mask (use pad embedding for invalid positions)
        combined_embs = torch.where(
            digit_mask.unsqueeze(-1),
            combined_embs,
            pad_embs
        )

        # Pool over digit dimension (mean of valid digits)
        pooled = combined_embs.sum(dim=1) / digit_lengths.unsqueeze(1).float()  # (batch_size, out_dim)

        # Add sign embedding and project
        output = sign_emb + pooled
        return self.proj(output)


class DatetimeEmbedding(nn.Module):
    """Datetime embedding for time-series timestamps."""

    def __init__(
        self,
        out_dim: int = 768,
        **kwargs,
    ):
        super().__init__()
        self.out_dim = out_dim

        # Embeddings for different datetime components
        self.year_emb = nn.Embedding(3000, out_dim)  # Years 0-2999
        self.month_emb = nn.Embedding(12, out_dim)   # Months 0-11
        self.day_emb = nn.Embedding(31, out_dim)     # Days 0-30
        self.hour_emb = nn.Embedding(24, out_dim)    # Hours 0-23
        self.minute_emb = nn.Embedding(60, out_dim)  # Minutes 0-59
        self.second_emb = nn.Embedding(60, out_dim)  # Seconds 0-59

        # Projection layer
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, datetimes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            datetimes: (batch_size, 6) - [year, month, day, hour, minute, second]

        Returns:
            embeddings: (batch_size, out_dim)
        """
        # Clamp values to valid ranges and adjust for embedding indices
        years = torch.clamp(datetimes[:, 0], 0, 2999)
        months = torch.clamp(datetimes[:, 1] - 1, 0, 11)  # Convert to 0-indexed
        days = torch.clamp(datetimes[:, 2] - 1, 0, 30)    # Convert to 0-indexed
        hours = torch.clamp(datetimes[:, 3], 0, 23)
        minutes = torch.clamp(datetimes[:, 4], 0, 59)
        seconds = torch.clamp(datetimes[:, 5], 0, 59)

        # Get embeddings
        year_emb = self.year_emb(years.long())
        month_emb = self.month_emb(months.long())
        day_emb = self.day_emb(days.long())
        hour_emb = self.hour_emb(hours.long())
        minute_emb = self.minute_emb(minutes.long())
        second_emb = self.second_emb(seconds.long())

        # Combine all datetime embeddings
        combined = year_emb + month_emb + day_emb + hour_emb + minute_emb + second_emb

        return self.proj(combined)


# ===== Time-Series Patch Embedding ===== #

class Qwen2_5_TimeSeriesPatchEmbed(nn.Module):
    """Conv1d patch embedding for time-series data."""

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (N_patches, in_channels, patch_size)

        Returns:
            (N_patches, embed_dim)
        """
        x = hidden_states.to(dtype=self.proj.weight.dtype)
        x = self.proj(x)  # (N_patches, embed_dim, 1)
        x = x.view(-1, self.embed_dim)  # (N_patches, embed_dim)
        return x


# ===== Time-Series Transformer ===== #

class Qwen2_5_TimeSeriesTransformer(nn.Module):
    """Time-series encoder component."""

    def __init__(
        self,
        config: Qwen2_5_VLTimeSeriesConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_size
        self.patch_size = config.patch_size

        # Embedding components
        digitwise_out_dim = self.hidden_dim // 2  # Split embedding dimension
        datetime_out_dim = self.hidden_dim // 2

        self.number_embedding = DigitwiseFloatEmbedding(
            out_dim=digitwise_out_dim,
            digit_pad=12,  # Will be overridden by processor config
        )

        self.datetime_embedding = DatetimeEmbedding(
            out_dim=datetime_out_dim,
        )

        # Mixing projection
        self.mixing_project = nn.Sequential(
            nn.Linear(digitwise_out_dim + datetime_out_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # Layer norms
        self.pre_conv_ln = nn.LayerNorm((self.patch_size, self.hidden_dim), elementwise_affine=True)
        self.post_conv_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=True)

        # Patch embedding
        self.conv = Qwen2_5_TimeSeriesPatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.hidden_dim,
            embed_dim=self.hidden_dim,
        )

        # Transformer blocks (reuse vision blocks but adapt for time-series)
        # Create a compatible config for vision blocks
        from .qwen2_5_vl import Qwen2_5_VisionBlock

        # Create vision-compatible config from time-series config
        vision_config = type('VisionConfig', (), {
            'hidden_size': config.hidden_size,
            'num_heads': config.num_heads,
            'intermediate_size': config.intermediate_size,
            'hidden_act': config.hidden_act,
        })()

        self.blocks = nn.ModuleList([
            Qwen2_5_VisionBlock(
                dim=config.hidden_size,
                num_heads=config.num_heads,
                mlp_hidden_dim=config.intermediate_size,
                act_fn=get_act_and_mul_fn(config.hidden_act),
                quant_config=quant_config,
                prefix=f"{prefix}.blocks.{layer_idx}",
                use_data_parallel=False,  # Will be set properly later
            )
            for layer_idx in range(config.depth)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        ts_digits: torch.Tensor,
        ts_digit_lengths: torch.Tensor,
        datetimes: torch.Tensor,
        time_series_grid_thw: torch.Tensor,
        batch_ts_seq_end_idxs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass for time-series encoder."""

        # 1. Embed numbers and datetimes
        ts_proj = self.number_embedding(ts_digits, ts_digit_lengths)
        dt_proj = self.datetime_embedding(datetimes)
        mixed = self.mixing_project(torch.cat([ts_proj, dt_proj], dim=-1))

        # 2. Pad sequences to patch boundaries (simplified version)
        mixed_padded = self._pad_to_patch_boundaries(mixed, batch_ts_seq_end_idxs)

        # 3. Create patches
        N = mixed_padded.size(0)
        assert (N % self.patch_size) == 0, f"Sequence length {N} must be divisible by patch_size {self.patch_size}"

        patches = mixed_padded.view(-1, self.patch_size, self.hidden_dim)
        patches = self.pre_conv_ln(patches)

        # 4. Apply conv1d patchification
        hidden_states = self.conv(patches.view(-1, self.hidden_dim, self.patch_size))
        hidden_states = self.post_conv_ln(hidden_states)

        # 5. Apply transformer blocks (simplified - would need proper attention handling)
        hidden_states = hidden_states.unsqueeze(1)  # Add batch dim for compatibility

        for blk in self.blocks:
            # This is simplified - in reality we'd need proper cu_seqlens and position embeddings
            hidden_states = blk(
                hidden_states,
                cu_seqlens=torch.tensor([0, hidden_states.size(0)], device=hidden_states.device),
                rotary_pos_emb=None,
                max_seqlen=torch.tensor(hidden_states.size(0), device=hidden_states.device),
                seqlens=torch.tensor([hidden_states.size(0)], device=hidden_states.device),
            )

        # 6. Output normalization
        return self.output_norm(hidden_states.squeeze(1))

    def _pad_to_patch_boundaries(self, mixed: torch.Tensor, batch_ts_seq_end_idxs: torch.Tensor) -> torch.Tensor:
        """Pad sequences to patch boundaries."""
        # Simplified implementation - pad to next multiple of patch_size
        current_len = mixed.size(0)
        target_len = ((current_len + self.patch_size - 1) // self.patch_size) * self.patch_size

        if target_len > current_len:
            padding_size = target_len - current_len
            padding = torch.zeros(padding_size, mixed.size(1), dtype=mixed.dtype, device=mixed.device)
            mixed_padded = torch.cat([mixed, padding], dim=0)
        else:
            mixed_padded = mixed

        return mixed_padded

    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load time-series encoder weights."""
        loaded_params = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            # Remove ts_encoder prefix if present
            if name.startswith("ts_encoder."):
                param_name = name[len("ts_encoder."):]
            else:
                param_name = name

            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


# ===== Main Model Classes ===== #

class Qwen2_5_VLTSModel(nn.Module):
    """Qwen2.5-VLTS model with time-series support."""

    def __init__(
        self,
        config: Qwen2_5_VLTSConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vllm_config = vllm_config

        # Initialize base Qwen2.5-VL components
        # Visual encoder (disabled for time-series only model)
        self.visual = None

        # Time-series encoder
        self.ts_encoder = Qwen2_5_TimeSeriesTransformer(
            config=config.time_series_config,
            quant_config=vllm_config.model_config.quantization,
            prefix=maybe_prefix(prefix, "ts_encoder"),
        )

        # Language model
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # Initialize time-series preprocessor
        self.ts_preprocessor = None  # Will be loaded from config

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def get_time_series_features(
        self,
        ts_digits: torch.Tensor,
        ts_digit_lengths: torch.Tensor,
        datetimes: torch.Tensor,
        time_series_grid_thw: torch.Tensor,
        batch_ts_seq_end_idxs: torch.Tensor,
    ) -> torch.Tensor:
        """Process time-series inputs into embeddings."""
        return self.ts_encoder.forward(
            ts_digits=ts_digits,
            ts_digit_lengths=ts_digit_lengths,
            datetimes=datetimes,
            time_series_grid_thw=time_series_grid_thw,
            batch_ts_seq_end_idxs=batch_ts_seq_end_idxs,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass through the model."""
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states


# ===== Multimodal Processor ===== #

class Qwen2_5_VLTSProcessingInfo(Qwen2_5_VLProcessingInfo):
    """Processing info for Qwen2.5-VLTS model."""

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen2_5_VLTSConfig)

    def get_hf_processor(self, **kwargs: object):
        # This would normally get the custom processor, but for now use base
        return self.ctx.get_hf_processor(
            Qwen2_5_VLProcessor,  # TODO: Replace with Qwen2_5_VLTSProcessor
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )


class Qwen2_5_VLTSMultiModalProcessor(Qwen2_5_VLMultiModalProcessor):
    """Extended multimodal processor with time-series support."""

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs,
    ) -> Dict[str, MultiModalFieldConfig]:
        """Get multimodal field configurations including time-series."""
        base_config = super()._get_mm_fields_config(hf_inputs, hf_processor_mm_kwargs)

        # Add time-series specific field configurations
        ts_fields = {
            "ts_digits": MultiModalFieldConfig.flat("time_series", []),
            "ts_digit_lengths": MultiModalFieldConfig.flat("time_series", []),
            "datetimes": MultiModalFieldConfig.flat("time_series", []),
            "time_series_grid_thw": MultiModalFieldConfig.batched("time_series"),
            "batch_ts_seq_end_idxs": MultiModalFieldConfig.shared("time_series", 1),
        }

        return {**base_config, **ts_fields}


# ===== Main Model Class ===== #

@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLTSMultiModalProcessor,
    info=Qwen2_5_VLTSProcessingInfo,
)
class Qwen2_5_VLTSForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsQuant,
    SupportsMRoPE,
):
    """Qwen2.5-VLTS model for conditional generation with time-series support."""

    config_class = Qwen2_5_VLTSConfig

    # Weight mapping for loading HF weights
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen2_5_VLTSConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        # Initialize model components
        self.model = Qwen2_5_VLTSModel(
            config=config,
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.language_model = self.model.language_model

        # Load time-series preprocessor config
        self.ts_preprocessor = None
        if hasattr(vllm_config.model_config, 'model') and vllm_config.model_config.model:
            try:
                self.ts_preprocessor = Qwen2_5_TimeSeriesPreprocessor.from_pretrained(
                    vllm_config.model_config.model
                )
            except Exception as e:
                logger.warning(f"Could not load time-series preprocessor config: {e}")
                self.ts_preprocessor = Qwen2_5_TimeSeriesPreprocessor()

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass for Qwen2.5-VLTS model."""
        return self.model.forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def load_weights(self, weights: List[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the Qwen2.5-VLTS model."""
        # Separate time-series encoder weights from other weights
        ts_encoder_weights = []
        other_weights = []

        for name, weight in weights:
            if name.startswith("ts_encoder."):
                ts_encoder_weights.append((name, weight))
            else:
                other_weights.append((name, weight))

        loaded_params = set()

        # Load time-series encoder weights
        if hasattr(self.model, 'ts_encoder') and ts_encoder_weights:
            loaded_params.update(
                self.model.ts_encoder.load_weights(ts_encoder_weights)
            )

        # Load other weights using AutoWeightsLoader
        loader = AutoWeightsLoader(self)
        loaded_params.update(loader.load_weights(other_weights, mapper=self.hf_to_vllm_mapper))

        return loaded_params

    def get_multimodal_embeddings(self, **kwargs: object) -> List[torch.Tensor]:
        """Get multimodal embeddings including time-series."""
        # Parse time-series inputs if available
        ts_inputs = self._parse_and_validate_ts_input(**kwargs)
        if ts_inputs is None:
            return []

        # Process time-series inputs
        ts_embeddings = self.model.get_time_series_features(
            ts_digits=ts_inputs["ts_digits"],
            ts_digit_lengths=ts_inputs["ts_digit_lengths"],
            datetimes=ts_inputs["datetimes"],
            time_series_grid_thw=ts_inputs["time_series_grid_thw"],
            batch_ts_seq_end_idxs=ts_inputs["batch_ts_seq_end_idxs"],
        )

        return [ts_embeddings]

    def _parse_and_validate_ts_input(self, **kwargs: object) -> Optional[Dict[str, torch.Tensor]]:
        """Parse and validate time-series input."""
        ts_digits = kwargs.pop("ts_digits", None)
        ts_digit_lengths = kwargs.pop("ts_digit_lengths", None)
        datetimes = kwargs.pop("datetimes", None)
        time_series_grid_thw = kwargs.pop("time_series_grid_thw", None)
        batch_ts_seq_end_idxs = kwargs.pop("batch_ts_seq_end_idxs", None)

        if all(x is None for x in [ts_digits, ts_digit_lengths, datetimes]):
            return None

        if any(x is None for x in [ts_digits, ts_digit_lengths, datetimes, time_series_grid_thw, batch_ts_seq_end_idxs]):
            raise ValueError("All time-series inputs must be provided together")

        return {
            "ts_digits": ts_digits,
            "ts_digit_lengths": ts_digit_lengths,
            "datetimes": datetimes,
            "time_series_grid_thw": time_series_grid_thw,
            "batch_ts_seq_end_idxs": batch_ts_seq_end_idxs,
        }
