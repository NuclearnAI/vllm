# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Time-series processing utilities for Qwen2.5-VLTS integration."""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def float_to_digit_tokens(
    inputs: Union[torch.Tensor, List[float]],
    *,
    dec_digits: int,
    pad_digits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert floats -> [sign, digits...] with fixed decimals.

    Args:
        inputs: Float values to convert
        dec_digits: Number of decimal places
        pad_digits: Maximum number of digits to pad to

    Returns:
        tokens: (N, 1+pad_digits) long - sign + digit tokens
        digit_lengths: (N,) long - actual number of digits per value
    """
    if isinstance(inputs, list):
        values = inputs
    elif isinstance(inputs, torch.Tensor):
        assert inputs.dim() == 1, "Expected a 1D tensor"
        values = inputs.tolist()
    else:
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    N = len(values)
    tokens = torch.zeros((N, 1 + pad_digits), dtype=torch.long)
    lengths = torch.zeros((N,), dtype=torch.long)

    for i, val in enumerate(values):
        sign = 1 if val < 0 else 0
        formatted = f"{abs(val):.{dec_digits}f}"

        if dec_digits == 0:
            int_str, dec_str = formatted, ""
        else:
            int_str, dec_str = formatted.split(".")

        digits_str = int_str + dec_str
        if digits_str:
            num_keep = min(len(digits_str), pad_digits)
            digits_tensor = torch.tensor([int(c) for c in digits_str[:num_keep]], dtype=torch.long)
            tokens[i, 0] = sign
            tokens[i, 1 : 1 + num_keep] = digits_tensor
            lengths[i] = num_keep
        else:
            tokens[i, 0] = sign
            lengths[i] = 0

    return tokens, lengths


def parse_datetime_components(dt_str: str, fmt: str) -> List[int]:
    """
    Parse datetime string into [year, month, day, hour, minute, second].

    Args:
        dt_str: Datetime string to parse
        fmt: Expected datetime format

    Returns:
        List of [year, month, day, hour, minute, second]
    """
    try:
        dt = datetime.strptime(dt_str, fmt)
    except ValueError:
        # Fallback: assume date-only format, set time to 00:00:00
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Could not parse datetime string: {dt_str}")
    return [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]


def pad_and_reconcat_to_patches(
    mixed: torch.Tensor,
    split_indices: List[int],
    ts_patch_size: int,
) -> torch.Tensor:
    """
    Pad each sub-sequence of `mixed` along dim=0 to a multiple of `ts_patch_size` with zeros,
    then re-concatenate in the original order.

    Args:
        mixed: Tensor of shape [N, D] (N timesteps concatenated across streams; D = embedding dim).
        split_indices: Cumulative end indices for each stream.
        ts_patch_size: Positive integer patch size (non-overlapping conv kernel/stride).

    Returns:
        Tensor of shape [N_padded, D], where each stream is zero-padded to a multiple of ts_patch_size.
    """
    if mixed.dim() != 2:
        raise ValueError(f"`mixed` must be 2D [N, D], got shape {tuple(mixed.shape)}")

    if ts_patch_size <= 0:
        raise ValueError(f"`ts_patch_size` must be positive, got {ts_patch_size}")

    N, D = mixed.shape
    device = mixed.device
    dtype = mixed.dtype

    # Normalize and validate split indices
    if not split_indices:
        # Treat the entire tensor as a single stream
        split_indices = [N]
    else:
        split_indices = sorted(int(i) for i in split_indices)
        if split_indices[-1] != N:
            split_indices.append(N)
        if split_indices[0] <= 0 or split_indices[-1] != N:
            raise ValueError("`split_indices` must be cumulative end indices within (0, N] in ascending order.")

    # Slice, pad each chunk, and collect
    padded_chunks = []
    start = 0
    for end in split_indices:
        chunk = mixed[start:end]  # [S_i, D]
        S_i = chunk.shape[0]
        if S_i == 0:
            # Empty stream → nothing to patchify; skip adding anything
            start = end
            continue

        # Tail padding to multiple of ts_patch_size (in embedding space)
        tail = (-S_i) % ts_patch_size
        if tail > 0:
            pad = torch.zeros((tail, D), device=device, dtype=dtype)
            chunk = torch.cat([chunk, pad], dim=0)  # [S_i + tail, D]

        padded_chunks.append(chunk)
        start = end

    if not padded_chunks:
        # All streams were empty
        return mixed.new_zeros((0, D))

    return torch.cat(padded_chunks, dim=0)


def _round_up(x: int, m: int) -> int:
    """Round x up to the nearest multiple of m."""
    r = x % m
    return x if r == 0 else x + (m - r)


class TimeSeriesProcessor:
    """
    Time-series processor that handles the complex Qwen2.5-VLTS format.
    This implements the core logic from your vlts_processor.py.
    """

    def __init__(
        self,
        ts_patch_size: int = 4,
        ts_merge_size: int = 1,
        ts_digit_pad: int = 12,
        ts_dec_digits: int = 4,
        ts_datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        self.ts_patch_size = ts_patch_size
        self.ts_merge_size = ts_merge_size
        self.ts_digit_pad = ts_digit_pad
        self.ts_dec_digits = ts_dec_digits
        self.ts_datetime_format = ts_datetime_format

    def process_time_series_batch(
        self,
        time_series_values: List[List[List[float]]],  # List[B][M][S]
        time_series_datetimes: Union[List[List[List[str]]], List[List[str]]],  # List[B][M][S] or List[B][S]
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of time-series data in your nested format.

        Args:
            time_series_values: List[B][M][S] - B batches, M streams, S timesteps
            time_series_datetimes: Either List[B][M][S] or List[B][S] for broadcasting

        Returns:
            Dictionary containing processed tensors for model input
        """
        if not isinstance(time_series_values, list) or not isinstance(time_series_datetimes, list):
            raise TypeError("time_series_values and time_series_datetimes must be nested Python lists.")

        B = len(time_series_values)
        if len(time_series_datetimes) != B:
            raise ValueError("len(time_series_datetimes) must equal len(time_series_values) (batch dimension)")

        # Metadata accumulators
        ts_grid_thw_flat: List[List[int]] = []     # one row per stream: [T,1,1] (T uses PADDED length)
        ts_tokens_per_sample: List[int] = []       # per-sample sum of T (padded)
        ts_value_cu_seqlens = [0]                  # cumulative #REAL values per sample (B+1)
        ts_stream_cu_seqlens = [0]                 # cumulative #streams per sample (B+1)

        ts_batch_idx: List[int] = []               # indices for REAL values
        ts_stream_idx: List[int] = []              # stream index within sample
        ts_step_idx: List[int] = []                # timestep index within stream

        # Outputs (concatenated over all streams, all samples)
        digit_token_chunks: List[torch.Tensor] = []
        digit_len_chunks: List[torch.Tensor] = []
        datetime_chunks: List[torch.Tensor] = []
        pad_mask_chunks: List[torch.Tensor] = []

        stream_lengths: List[int] = []             # per-stream REAL lengths

        for b in range(B):
            series_b = time_series_values[b]   # List[M_b][S_{b,m}]
            dts_b = time_series_datetimes[b]   # Either List[M_b][S_{b,m}] or List[S*]

            if not series_b:
                ts_tokens_per_sample.append(0)
                ts_value_cu_seqlens.append(ts_value_cu_seqlens[-1])
                ts_stream_cu_seqlens.append(ts_stream_cu_seqlens[-1])
                continue

            M_b = len(series_b)

            # Normalize datetimes to per-value: List[M_b][S_{b,m}]
            if isinstance(dts_b, list) and dts_b and isinstance(dts_b[0], list):
                if len(dts_b) != M_b:
                    raise ValueError(f"time_series_datetimes[{b}] must have M={M_b} streams to match values.")
                for m in range(M_b):
                    if len(dts_b[m]) != len(series_b[m]):
                        raise ValueError(
                            f"Sample {b}, stream {m}: datetimes length {len(dts_b[m])} "
                            f"must match values length {len(series_b[m])}."
                        )
                dts_b_ms = dts_b
            else:
                # Broadcast: all streams share the same datetime sequence
                if not isinstance(dts_b, list) or len(dts_b) == 0:
                    raise ValueError(
                        f"time_series_datetimes[{b}] must be List[M][S_m] or a non-empty List[S*] to broadcast."
                    )
                S0 = len(series_b[0])
                if any(len(series_b[m]) != S0 for m in range(M_b)) or len(dts_b) != S0:
                    raise ValueError(
                        f"Sample {b}: cannot broadcast datetimes; streams are ragged or length mismatch."
                    )
                dts_b_ms = [dts_b[:] for _ in range(M_b)]

            # Process each stream
            token_sum_b = 0
            real_values_in_sample = 0
            ts_stream_cu_seqlens.append(ts_stream_cu_seqlens[-1] + M_b)

            for m in range(M_b):
                vals_m: List[float] = series_b[m]
                dts_m: List[str] = dts_b_ms[m]
                S_bm = len(vals_m)
                stream_lengths.append(S_bm)
                real_values_in_sample += S_bm

                # Encode digits for REAL values
                tokens_m, lens_m = float_to_digit_tokens(vals_m, dec_digits=self.ts_dec_digits, pad_digits=self.ts_digit_pad)

                # Encode datetime components for REAL values
                dt_rows_m = torch.tensor(
                    [parse_datetime_components(sdt, self.ts_datetime_format) for sdt in dts_m],
                    dtype=torch.int32
                ) if S_bm > 0 else torch.zeros((0, 6), dtype=torch.int32)

                # Compute per-stream padding requirements
                S_pad = _round_up(S_bm, self.ts_patch_size)
                pad_mask_m = torch.zeros(S_bm, dtype=torch.bool)

                # Store tensors (only real values, no materialized padding)
                digit_token_chunks.append(tokens_m)
                digit_len_chunks.append(lens_m)
                datetime_chunks.append(dt_rows_m)
                pad_mask_chunks.append(pad_mask_m)

                # Grid T uses the padded length for consistency
                T_m = S_pad // self.ts_patch_size
                ts_grid_thw_flat.append([int(T_m), 1, 1])
                token_sum_b += int(T_m)

                # Indices for REAL values only
                for s in range(S_bm):
                    ts_batch_idx.append(b)
                    ts_stream_idx.append(m)
                    ts_step_idx.append(s)

            ts_tokens_per_sample.append(token_sum_b)
            ts_value_cu_seqlens.append(ts_value_cu_seqlens[-1] + real_values_in_sample)

        # Stack/finalize flat tensors
        if digit_token_chunks:
            ts_digits = torch.cat(digit_token_chunks, dim=0)
            ts_digit_lengths = torch.cat(digit_len_chunks, dim=0)
            datetimes = torch.cat(datetime_chunks, dim=0)
            ts_pad_mask = torch.cat(pad_mask_chunks, dim=0)
        else:
            ts_digits = torch.zeros((0, 1 + self.ts_digit_pad), dtype=torch.long)
            ts_digit_lengths = torch.zeros((0,), dtype=torch.long)
            datetimes = torch.zeros((0, 6), dtype=torch.int32)
            ts_pad_mask = torch.zeros((0,), dtype=torch.bool)

        # Build final output
        result = {
            "time_series_grid_thw": torch.tensor(ts_grid_thw_flat, dtype=torch.long),
            "ts_tokens_per_sample": torch.tensor(ts_tokens_per_sample, dtype=torch.int32),
            "ts_value_cu_seqlens": torch.tensor(ts_value_cu_seqlens, dtype=torch.int32),
            "ts_stream_cu_seqlens": torch.tensor(ts_stream_cu_seqlens, dtype=torch.int32),
            "ts_lengths": torch.tensor(stream_lengths, dtype=torch.int32),
            "ts_digits": ts_digits,
            "ts_digit_lengths": ts_digit_lengths,
            "datetimes": datetimes,
            "ts_pad_mask": ts_pad_mask,
            "ts_batch_idx": torch.tensor(ts_batch_idx, dtype=torch.int32) if ts_batch_idx else torch.zeros((0,), dtype=torch.int32),
            "ts_channel_idx": torch.tensor(ts_stream_idx, dtype=torch.int32) if ts_stream_idx else torch.zeros((0,), dtype=torch.int32),
            "ts_step_idx": torch.tensor(ts_step_idx, dtype=torch.int32) if ts_step_idx else torch.zeros((0,), dtype=torch.int32),
        }

        return result

    def expand_time_series_tokens(
        self,
        text: str,
        ts_tokens_per_sample: torch.Tensor,
        ts_stream_cu_seqlens: torch.Tensor,
        time_series_grid_thw: torch.Tensor,
    ) -> str:
        """
        Expand time-series token placeholders in text.

        This handles the pattern: <|ts_start|><|ts_pad|><|ts_end|>
        -> <|ts_start|><|ts_pad|>*N<|ts_end|> for each stream

        Args:
            text: Input text with time-series placeholders
            ts_tokens_per_sample: Number of tokens per sample
            ts_stream_cu_seqlens: Cumulative stream counts
            time_series_grid_thw: Grid information per stream

        Returns:
            Text with expanded time-series tokens
        """
        ts_block_pattern = "<|ts_start|><|ts_pad|><|ts_end|>"

        if ts_block_pattern not in text:
            return text

        # Get grid information
        grid_flat = time_series_grid_thw.tolist()
        stream_cu = ts_stream_cu_seqlens.tolist()

        # For each sample, expand tokens based on streams
        # This is simplified - in full implementation we'd need batch handling
        if len(stream_cu) > 1:
            s_lo, s_hi = stream_cu[0], stream_cu[1]  # First sample
            per_stream_T = [int(grid_flat[j][0]) for j in range(s_lo, s_hi)]

            # Count occurrences in text
            occ = text.count(ts_block_pattern)

            if occ == 1:
                # Single block -> expand to ALL streams (aggregate)
                repeated_blocks = "".join(
                    "<|ts_start|>" + ("<|ts_pad|>" * Tm) + "<|ts_end|>" for Tm in per_stream_T
                )
                text = text.replace(ts_block_pattern, repeated_blocks, 1)
            elif occ > 1:
                # Multiple blocks -> consume streams one-by-one
                stream_idx = 0
                for _ in range(occ):
                    if stream_idx < len(per_stream_T):
                        Tm = per_stream_T[stream_idx]
                        block = "<|ts_start|>" + ("<|ts_pad|>" * Tm) + "<|ts_end|>"
                        stream_idx += 1
                    else:
                        # No stream left: produce an empty TS block
                        block = "<|ts_start|><|ts_end|>"
                    text = text.replace(ts_block_pattern, block, 1)

        return text


def create_time_series_field_configs(
    ts_data: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """
    Create multimodal field configurations for time-series data.

    Args:
        ts_data: Processed time-series tensor data

    Returns:
        Field configuration dictionary for vLLM multimodal framework
    """
    from vllm.multimodal.inputs import MultiModalFieldConfig

    # Determine sizes for flat field configurations
    n_values = ts_data["ts_digits"].size(0)
    n_streams = ts_data["time_series_grid_thw"].size(0)
    batch_size = len(ts_data["ts_value_cu_seqlens"]) - 1

    configs = {
        "ts_digits": MultiModalFieldConfig.flat("time_series", [slice(0, n_values)]),
        "ts_digit_lengths": MultiModalFieldConfig.flat("time_series", [slice(0, n_values)]),
        "datetimes": MultiModalFieldConfig.flat("time_series", [slice(0, n_values)]),
        "time_series_grid_thw": MultiModalFieldConfig.flat("time_series", [slice(0, n_streams)]),
        "ts_tokens_per_sample": MultiModalFieldConfig.shared("time_series", batch_size),
        "ts_value_cu_seqlens": MultiModalFieldConfig.shared("time_series", batch_size),
        "ts_stream_cu_seqlens": MultiModalFieldConfig.shared("time_series", batch_size),
        "ts_lengths": MultiModalFieldConfig.shared("time_series", batch_size),
        "ts_batch_idx": MultiModalFieldConfig.flat("time_series", [slice(0, len(ts_data["ts_batch_idx"]))]) if len(ts_data["ts_batch_idx"]) > 0 else MultiModalFieldConfig.shared("time_series", 1),
        "ts_channel_idx": MultiModalFieldConfig.flat("time_series", [slice(0, len(ts_data["ts_channel_idx"]))]) if len(ts_data["ts_channel_idx"]) > 0 else MultiModalFieldConfig.shared("time_series", 1),
        "ts_step_idx": MultiModalFieldConfig.flat("time_series", [slice(0, len(ts_data["ts_step_idx"]))]) if len(ts_data["ts_step_idx"]) > 0 else MultiModalFieldConfig.shared("time_series", 1),
    }

    return configs
