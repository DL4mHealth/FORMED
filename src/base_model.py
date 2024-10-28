# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications made to this file by Nan Huang on 2024/08/20.

"""Pytorch version of patched decoder."""

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class TimesFMConfig:
    """Config for initializing timesfm patched_decoder class."""

    # The number of blocks in the model.
    num_layers: int = 20
    # The number of attention heads used in the attention layers of the model.
    num_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_kv_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 1280
    # The dimension of the MLP representations.
    intermediate_size: int = 1280
    # The number of head dimensions.
    head_dim: int = 80
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # Patch length
    patch_len: int = 32
    # Padding value
    pad_val: float = 1123581321.0
    # Tolerance
    tolerance: float = 1e-6
    # use positional embedding
    use_positional_embedding: bool = True


def _masked_mean_std(
    inputs: torch.Tensor, padding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean and standard deviation of `inputs` across axis 1.

    It excludes values where `padding` is 1.

    Args:
      inputs: A PyTorch tensor of shape [b, n, p].
      padding: A PyTorch tensor of shape [b, n, p] with values 0 or 1.

    Returns:
      A tuple containing the mean and standard deviation.
      We return the statistics of the first patch with more than three non-padded
      values.
    """
    # Selecting the first patch with more than 3 unpadded values.
    pad_sum = torch.sum(1 - padding, dim=2)

    def _get_patch_index(arr: torch.Tensor):
        indices = torch.argmax((arr >= 3).to(torch.int32), dim=1)
        row_sum = (arr >= 3).to(torch.int32).sum(dim=1)
        return torch.where(row_sum == 0, arr.shape[1] - 1, indices)

    patch_indices = _get_patch_index(pad_sum)
    bidxs = torch.arange(inputs.shape[0])

    arr = inputs[bidxs, patch_indices, :]
    pad = padding[bidxs, patch_indices, :]

    # Create a mask where padding is 0
    mask = 1 - pad

    # Calculate the number of valid elements
    num_valid_elements = torch.sum(mask, dim=1)
    num_valid_elements = torch.where(
        num_valid_elements == 0,
        torch.tensor(
            1, dtype=num_valid_elements.dtype, device=num_valid_elements.device
        ),
        num_valid_elements,
    )

    # Calculate the masked sum and squared sum
    masked_sum = torch.sum(arr * mask, dim=1)
    masked_squared_sum = torch.sum((arr * mask) ** 2, dim=1)

    # Calculate the masked mean and standard deviation
    masked_mean = masked_sum / num_valid_elements
    masked_var = masked_squared_sum / num_valid_elements - masked_mean**2
    masked_var = torch.where(
        masked_var < 0.0,
        torch.tensor(0.0, dtype=masked_var.dtype, device=masked_var.device),
        masked_var,
    )
    masked_std = torch.sqrt(masked_var)

    return masked_mean, masked_std


def _shift_padded_seq(mask: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
    """Shifts rows of seq based on the first 0 in each row of the mask.

    Args:
      mask: mask tensor of shape [B, N]
      seq: seq tensor of shape [B, N, P]

    Returns:
      Returns the shifted sequence.
    """
    batch_size, num_seq, feature_dim = seq.shape

    new_mask = mask == 0

    # Use argmax to find the first True value in each row
    indices = new_mask.to(torch.int32).argmax(dim=1)

    # Handle rows with all zeros
    indices[~new_mask.any(dim=1)] = -1

    # Create index ranges for each sequence in the batch
    idx_range = (
        torch.arange(num_seq)
        .to(seq.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch_size, -1, feature_dim)
    )

    # Calculate shifted indices for each element in each sequence
    shifted_idx = (idx_range - indices[:, None, None]) % num_seq

    # Gather values from seq using shifted indices
    shifted_seq = seq.gather(1, shifted_idx)

    return shifted_seq


def get_large_negative_number(dtype: torch.dtype) -> torch.Tensor:
    """Returns a large negative value for the given dtype."""
    if dtype.is_floating_point:
        dtype_max = torch.finfo(dtype).max
    else:
        dtype_max = torch.iinfo(dtype).max
    return torch.tensor(-0.7 * dtype_max, dtype=dtype)


def apply_mask_to_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies a floating-point mask to a set of logits.

    Args:
        logits: A torch.Tensor of logit values.
        mask: A torch.Tensor (float32) of mask values with the encoding described
          in the function documentation.

    Returns:
        Masked logits.
    """

    min_value = get_large_negative_number(logits.dtype)

    return torch.where((mask >= min_value * 0.5), logits, min_value)


def convert_paddings_to_mask(
    paddings: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Converts binary paddings to a logit mask ready to add to attention matrix.

    Args:
        paddings: binary torch.Tensor of shape [B, T], with 1 denoting padding
          token.
        dtype: data type of the input.

    Returns:
        A torch.Tensor of shape [B, 1, 1, T] ready to add to attention logits.
    """
    attention_mask = paddings[:, None, None, :]  # Equivalent to jnp.newaxis
    attention_mask *= get_large_negative_number(dtype)
    return attention_mask


def causal_mask(input_t: torch.Tensor) -> torch.Tensor:
    """Computes and returns causal mask.

    Args:
        input_t: A torch.Tensor of shape [B, T, D].

    Returns:
        An attention_mask torch.Tensor of shape [1, 1, T, T]. Attention mask has
        already been converted to large negative values.
    """
    assert input_t.dtype.is_floating_point, input_t.dtype
    large_negative_number = get_large_negative_number(input_t.dtype)
    t = input_t.shape[1]
    col_idx = torch.arange(t).unsqueeze(0).repeat(t, 1)
    row_idx = torch.arange(t).unsqueeze(1).repeat(1, t)
    mask = (row_idx < col_idx).to(input_t.dtype) * large_negative_number
    return (
        mask.unsqueeze(0).unsqueeze(0).to(input_t.device)
    )  # Equivalent to jnp.newaxis


def merge_masks(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Merges 2 masks.

    logscale mask is expected but 0/1 mask is also fine.

    Args:
        a: torch.Tensor of shape [1|B, 1, 1|T, S].
        b: torch.Tensor of shape [1|B, 1, 1|T, S].

    Returns:
        torch.Tensor of shape [1|B, 1, 1|T, S].
    """

    def expand_t(key_mask):
        query_mask = key_mask.transpose(-1, -2)  # Equivalent of jnp.transpose
        return torch.minimum(query_mask, key_mask)

    if a.shape[2] != b.shape[2]:
        if a.shape[2] == 1:
            a = expand_t(a)
        else:
            assert b.shape[2] == 1
            b = expand_t(b)

    assert a.shape[1:] == b.shape[1:], f"a.shape={a.shape}, b.shape={b.shape}."
    return torch.minimum(a, b)  # Element-wise minimum, similar to jnp.minimum


class ResidualBlock(nn.Module):
    """TimesFM residual block."""

    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
    ):
        super(ResidualBlock, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        # Hidden Layer
        self.hidden_layer = nn.Linear(input_dims, hidden_dims)
        # Activation Function
        self.act = nn.SiLU()
        # Output Layer
        self.output_layer = nn.Linear(hidden_dims, output_dims)
        # Residual Layer
        self.residual_layer = nn.Linear(input_dims, output_dims)

    def forward(self, x):
        hidden = self.act(self.hidden_layer(x))
        output = self.output_layer(hidden)
        residual = self.residual_layer(x)
        return output + residual


class RMSNorm(nn.Module):
    """Pax rms norm in pytorch."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        if self.add_unit_offset:
            output = output * (1 + self.weight.float())
        else:
            output = output * self.weight.float()
        return output.type_as(x)


class TransformerMLP(nn.Module):
    """Pax transformer MLP in pytorch."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_size, eps=1e-6)

    def forward(self, x, paddings=None):
        gate_inp = self.layer_norm(x)
        gate = self.gate_proj(gate_inp)
        gate = F.relu(gate)
        outputs = self.down_proj(gate)
        if paddings is not None:
            outputs = outputs * (1.0 - paddings[:, :, None])
        return outputs + x


class TimesFMAttention(nn.Module):
    """Implements the attention used in TimesFM."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = nn.Parameter(
            torch.empty((self.head_dim,), dtype=torch.float32),
        )

        self.q_proj = nn.Linear(self.hidden_size, self.q_size)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

    def _per_dim_scaling(self, query: torch.Tensor) -> torch.Tensor:
        # [batch_size, n_local_heads, input_len, head_dim]
        r_softplus_0 = 1.442695041
        softplus_func = torch.nn.Softplus()
        scale = r_softplus_0 / math.sqrt(self.head_dim)
        scale = scale * softplus_func(self.scaling)
        return query * scale[None, None, None, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        kv_write_indices: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        xq = self.q_proj(hidden_states).view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        xk = self.k_proj(hidden_states).view(
            batch_size, -1, self.num_kv_heads, self.head_dim
        )
        xv = self.v_proj(hidden_states).view(
            batch_size, -1, self.num_kv_heads, self.head_dim
        )
        xq = self._per_dim_scaling(xq)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        if kv_cache is not None and kv_write_indices is not None:
            k_cache, v_cache = kv_cache
            k_cache.index_copy_(1, kv_write_indices, xk)
            v_cache.index_copy_(1, kv_write_indices, xv)

            key = k_cache
            value = v_cache
        else:
            key = xk
            value = xv
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # [batch_size, n_local_heads, input_len, max_seq_len]
        scores = torch.matmul(q, k.transpose(2, 3))
        scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)

        # [batch_size, n_local_heads, input_len, head_dim]
        output = torch.matmul(scores, v)
        # return scores, output.transpose(1, 2).contiguous()

        # [batch_size, input_len, hidden_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)

        return scores, output


class TimesFMDecoderLayer(nn.Module):
    """Transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.self_attn = TimesFMAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.mlp = TransformerMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        scores, hidden_states = self.self_attn(
            hidden_states=hidden_states,
            mask=mask,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # MLP
        hidden_states = self.mlp(hidden_states, paddings=paddings)

        return scores, hidden_states


class StackedDecoder(nn.Module):
    """Stacked transformer layer."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TimesFMDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                )
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        paddings: torch.Tensor,
        kv_write_indices: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        padding_mask = convert_paddings_to_mask(paddings, hidden_states.dtype)
        atten_mask = causal_mask(hidden_states)
        mask = merge_masks(padding_mask, atten_mask)
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            _, hidden_states = layer(
                hidden_states=hidden_states,
                mask=mask,
                paddings=paddings,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_cache,
            )
        return hidden_states


class PositionalEmbedding(nn.Module):
    """Generates position embedding for a given 1-d sequence.

    Attributes:
        min_timescale: Start of the geometric index. Determines the periodicity of
          the added signal.
        max_timescale: End of the geometric index. Determines the frequency of the
          added signal.
        embedding_dims: Dimension of the embedding to be generated.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: int = 1,
        max_timescale: int = 10_000,
    ) -> None:
        super().__init__()
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims

    def forward(self, seq_length=None, position=None):
        """Generates a Tensor of sinusoids with different frequencies.

        Args:
            seq_length: an optional Python int defining the output sequence length.
              if the `position` argument is specified.
            position:   [B, seq_length], optional position for each token in the
              sequence, only required when the sequence is packed.

        Returns:
            [B, seqlen, D] if `position` is specified, else [1, seqlen, D]
        """
        if position is None:
            assert seq_length is not None
            # [1, seqlen]
            position = torch.arange(seq_length, dtype=torch.float32).unsqueeze(0)
        else:
            assert position.ndim == 2, position.shape

        num_timescales = self.embedding_dims // 2
        log_timescale_increment = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / max(num_timescales - 1, 1)
        inv_timescales = self.min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
        )
        scaled_time = position.unsqueeze(2) * inv_timescales.unsqueeze(0).unsqueeze(0)
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        # Padding to ensure correct embedding dimension
        signal = F.pad(signal, (0, 0, 0, self.embedding_dims % 2))
        return signal


class PatchedTimeSeriesEncoder(nn.Module):
    """Patched time-series decoder."""

    def __init__(self, config: TimesFMConfig):
        super().__init__()
        self.config = config
        self.input_ff_layer = ResidualBlock(
            input_dims=2 * config.patch_len,
            output_dims=config.hidden_size,
            hidden_dims=config.intermediate_size,
        )
        self.freq_emb = nn.Embedding(num_embeddings=3, embedding_dim=config.hidden_size)
        self.stacked_transformer = StackedDecoder(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            head_dim=self.config.head_dim,
            num_layers=self.config.num_layers,
            rms_norm_eps=self.config.rms_norm_eps,
        )
        if self.config.use_positional_embedding:
            self.position_emb = PositionalEmbedding(self.config.hidden_size)

    def _forward_transform(
        self, inputs: torch.Tensor, patched_pads: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Input is of shape [B, N, P]."""
        mu, sigma = _masked_mean_std(inputs, patched_pads)
        sigma = torch.where(
            sigma < self.config.tolerance,
            torch.tensor(1.0, dtype=sigma.dtype, device=sigma.device),
            sigma,
        )

        # Normalize each patch
        outputs = (inputs - mu[:, None, None]) / sigma[:, None, None]
        outputs = torch.where(
            torch.abs(inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(
                self.config.pad_val, dtype=outputs.dtype, device=outputs.device
            ),
            outputs,
        )
        return outputs, (mu, sigma)

    def _preprocess_input(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """Preprocess input for stacked transformer."""

        # Reshape into patches (using view for efficiency)
        bsize = input_ts.shape[0]
        patched_inputs = input_ts.view(bsize, -1, self.config.patch_len)
        patched_pads = input_padding.view(bsize, -1, self.config.patch_len)

        patched_inputs = torch.where(
            torch.abs(patched_pads - 1.0) < self.config.tolerance,
            torch.tensor(0.0, dtype=patched_inputs.dtype, device=patched_inputs.device),
            patched_inputs,
        )
        patched_pads = torch.where(
            torch.abs(patched_inputs - self.config.pad_val) < self.config.tolerance,
            torch.tensor(1.0, dtype=patched_pads.dtype, device=patched_pads.device),
            patched_pads,
        )
        patched_inputs, stats = self._forward_transform(patched_inputs, patched_pads)

        # B x N x D
        patched_inputs = patched_inputs * (1.0 - patched_pads)
        concat_inputs = torch.cat([patched_inputs, patched_pads], dim=-1)
        model_input = self.input_ff_layer(concat_inputs)

        # A patch should not be padded even if there is at least one zero.
        patched_padding = torch.min(patched_pads, dim=-1)[
            0
        ]  # Get the values from the min result
        if self.config.use_positional_embedding:
            pos_emb = self.position_emb(model_input.shape[1]).to(model_input.device)
            pos_emb = torch.concat([pos_emb] * model_input.shape[0], dim=0)
            pos_emb = _shift_padded_seq(patched_padding, pos_emb)
            model_input += pos_emb

        return model_input, patched_padding, stats, patched_inputs

    def encode(
        self,
        input_ts: torch.Tensor,
        input_padding: torch.LongTensor,
        freq: torch.Tensor,
    ) -> torch.Tensor:
        model_input, patched_padding, _, _ = self._preprocess_input(
            input_ts=input_ts,
            input_padding=input_padding,
        )
        f_emb = self.freq_emb(freq)  # B x 1 x D
        model_input += f_emb
        model_output = self.stacked_transformer(model_input, patched_padding)

        return model_output

    def forward(
        self,
        input_ts: torch.Tensor,
        paddings: torch.Tensor,
        freq: torch.Tensor,
        max_len: int = 512,
        patch_per_step: int = 1,
        return_tokens_on_context: bool = False,
    ) -> torch.Tensor:
        """Auto-regressive encoding without caching.

        Args:
          input_ts: input time-series, shape B x C.
          paddings: input padding, shape B x C.
          freq: frequency shape B x 1
          max_len: maximum training context length.
          patch_per_step: number of patches to process per step
            if the input sequence is longer than max_len.
          return_tokens_on_context: whether to return tokens on context.

        Returns:
          Encoded features, shape B x N x D.
            In particular, if return_tokens_on_context is True, N will also include
            the tokens on context (i.e., the tokens before the last patch).
        """
        multi_channel = input_ts.dim() == 3
        if multi_channel:
            batch_size, num_channels, _ = input_ts.shape
            assert (
                paddings.shape == input_ts.shape
            ), "Input and padding must have the same shape."
            assert freq.shape == (batch_size, num_channels, 1), "Invalid freq shape."
            input_ts = input_ts.reshape(-1, input_ts.shape[2])
            paddings = paddings.reshape(-1, paddings.shape[2])
            freq = freq.reshape(-1, 1)
        else:
            batch_size, _ = input_ts.shape
            assert (
                paddings.shape == input_ts.shape
            ), "Input and padding must have the same shape."
            assert freq.shape == (batch_size, 1), "Invalid freq shape."

        encoded_features = []
        step_size = self.config.patch_len * patch_per_step
        num_steps = (
            1 + (max(input_ts.shape[1] - max_len, 0) + step_size - 1) // step_size
        )
        for step_index in range(num_steps):
            current_padding = paddings[
                :, step_index * step_size : step_index * step_size + max_len
            ]
            current_input = input_ts[
                :, step_index * step_size : step_index * step_size + max_len
            ]
            fprop_outputs = self.encode(current_input, current_padding, freq)
            if return_tokens_on_context and step_index == 0:
                encoded_features.append(fprop_outputs[:, :-1, :])

            encoded_features.append(fprop_outputs[:, -1:, :])

        encoded_features = torch.cat(encoded_features, dim=1)

        if multi_channel:
            encoded_features = encoded_features.reshape(
                batch_size, num_channels, -1, encoded_features.shape[-1]
            )

        return encoded_features
