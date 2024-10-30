# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_llama import LlamaConfig

# quant
try:
    from ptq.layers import QLinear, QAct, QIntSoftmax, QIntLayerNorm
    from ptq.bit_type import BIT_TYPE_DICT
except:
    from gptq_base.ptq.layers import QLinear, QAct, QIntSoftmax, QIntLayerNorm
    from gptq_base.ptq.bit_type import BIT_TYPE_DICT

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(
            self,
            config,

            bit_type_fc=None,
            calibration_mode_fc=None,
            observer_str_fc=None,
            quantizer_str=None,

            bit_type_fc_lower=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        # === quant =====
        # t1 * t2
        self.qact_t1_t2 = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.qact_t1_t2_lower = QAct(
            bit_type=BIT_TYPE_DICT['int4'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        # === quant =====

    def forward(
            self,
            x,
            new_lower_bit_idx_current,
    ):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            t1 = self.up_proj(x)

            t2 = self.act_fn(self.gate_proj(x))

            t1_t2 = t2 * t1
            # === quant =====
            t1_t2_here = t1_t2
            t1_t2 = self.qact_t1_t2(t1_t2_here)
            if new_lower_bit_idx_current is not None and not self.config.weight_quantization_mode:
                t1_t2_low_bit = self.qact_t1_t2_lower(t1_t2_here)
                t1_t2[:, new_lower_bit_idx_current, :] = t1_t2_low_bit[:, new_lower_bit_idx_current, :]
            # === quant =====

            down_proj = self.down_proj(t1_t2)

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: LlamaConfig,

            lower_bit_percent=None,

            token_remove_percent=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

        # === quant =====
        self.qact_q = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.qact_k = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.qact_v = QAct(  # mark
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.qact_softmax = QAct(  # mark
            bit_type=BIT_TYPE_DICT['int4'],
            calibration_mode='channel_wise',  # layer_wise  channel_wise
            observer_str='minmax',  # minmax     ptf_0to4
            quantizer_str='log2',  # uniform, log2
        )
        self.qact_afterkqv = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.qact_afterkqv_2 = QAct(
            bit_type=BIT_TYPE_DICT['int4'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.lower_bit_percent = lower_bit_percent
        # === quant =====

        # === token prune =====
        self.token_remove_percent = token_remove_percent
        # === token prune =====

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,

            lower_bit_idx_previous=None,
            remove_token_idx_previous=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        visualization_act = None

        if self.config.visualization_act:
            visualization_act = {}

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # === quant =====
        if self.config.visualization_act:
            visualization_act["after_layer.attn.q"] = query_states
            visualization_act["after_layer.attn.k"] = key_states
            visualization_act["after_layer.attn.v"] = value_states
        query_states = self.qact_q(query_states)
        key_states = self.qact_k(key_states)
        value_states = self.qact_v(value_states)
        # === quant =====

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        if self.config.demo_mode:
            # the generating step1: analyze according to the input sentence (tokens)
            # for generating step2: generating the next [one single] token
            if hidden_states.shape[1] == 1 and position_ids.shape[1] == 1:
                position_ids[0] = cos.shape[2] - 1

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if self.config.demo_mode:
            # the generating step1: analyze according to the input sentence (tokens)
            # for generating step2: generating the next [one single] token
            if attention_mask.shape[2] == 1 and torch.sum(attention_mask) == 0:
                if attention_mask.shape != attn_weights.shape:
                    attention_mask = attention_mask[:, :, :, :attn_weights.shape[3]]

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # xuan: we need to remove the following tensors in gpu to release more space for the float32 softmax operation
        del attention_mask
        del query_states
        del key_states

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)

        # === quant =====
        if self.config.visualization_act:
            visualization_act["after_layer.attn.softmax"] = attn_weights
        attn_weights = self.qact_softmax(attn_weights)
        # === quant =====

        attn_probs = attn_weights
        attn_probs = attn_probs.reshape(bsz * self.num_heads, q_len, kv_seq_len)
        # todo: (future work) this "token importance identification" only works for batch_size = 1
        #   if the batch_size != 1,
        #   "accumulate token importance" needs to revise,
        #   we need to get the mean (average) of the "attn_probs" for those batches
        '''
        # accumulate token importance
        # attn_probs.shape = batch_size, num_head, num_token, num_token
        attn_probs_here = torch.mean(attn_probs, dim=0)
        for head_id in range(attn_probs_here.shape[0]):
            attn_weights_head_i_here_for_use = torch.tril(attn_probs_here[head_id, :, :])
            score_token += torch.sum(attn_weights_head_i_here_for_use, dim=0)
        for token_id in range(attn_probs_here.shape[2]):
            score_token[0, token_id] = score_token[0, token_id] / (attn_probs_here.shape[1] - token_id)
        '''
        # ============== token importance identification ==============
        lower_bit_percent = self.lower_bit_percent
        token_remove_percent = self.token_remove_percent
        lower_bit_idx = None
        remove_token_idx = None

        if self.config.weight_quantization_mode:
            pass
        # if only lower bit quant is adopted
        elif lower_bit_percent > 0 and (token_remove_percent == 0 or token_remove_percent is None) \
                and not self.config.weight_quantization_mode:

            # ========================================================================================
            # accumulate token importance
            if self.config.conventional_token_importance:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([1, attn_probs_shape[2]])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = torch.tril(attn_probs[head_id, :, :])
                    score_token += torch.sum(attn_weights_head_i_here_for_use, dim=0)
                for token_id in range(attn_probs_shape[2]):
                    score_token[0, token_id] = score_token[0, token_id] / (attn_probs_shape[1] - token_id)
            # ========================================================================================

            # ========================================================================================
            # accumulate token importance   method 2
            else:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([attn_probs_shape[2], 1])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = attn_probs[head_id, :, :1]
                    score_token += attn_weights_head_i_here_for_use
                score_token = torch.reshape(score_token, (1, attn_probs_shape[2]))
            # ========================================================================================

            lower_bit_num_tokens = int(attn_probs_shape[2] * lower_bit_percent)
            _, lower_bit_idx = torch.topk(score_token, lower_bit_num_tokens, dim=1, largest=False)
            lower_bit_idx = lower_bit_idx.long()  # .long() to use this tensor as index of tensor matrix

            if self.config.lower_bit_inherit_from_previous_layers and lower_bit_idx_previous is not None:
                if remove_token_idx_previous is not None and remove_token_idx_previous.shape[1] > 0:
                    lower_bit_idx_previous_use_here = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    remove_token_idx_previous_use_here = np.sort(
                        remove_token_idx_previous.view(-1).detach().cpu().numpy())
                    new_lower_bit_idx_previous = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    point_a = 0
                    point_b = 0
                    while point_a < len(lower_bit_idx_previous_use_here) \
                            and point_b < len(remove_token_idx_previous_use_here):
                        if lower_bit_idx_previous_use_here[point_a] < remove_token_idx_previous_use_here[point_b]:
                            point_a += 1
                        elif lower_bit_idx_previous_use_here[point_a] == remove_token_idx_previous_use_here[point_b]:
                            raise ValueError("The lower bit index can not equal to token remove index")
                        else:
                            new_lower_bit_idx_previous[point_a:] = new_lower_bit_idx_previous[point_a:] - 1
                            point_b += 1

                    lower_bit_idx_previous = torch.from_numpy(
                        new_lower_bit_idx_previous
                    ).to(lower_bit_idx_previous.device).reshape(1, -1).long()

                lower_bit_idx = lower_bit_idx.view(-1)
                lower_bit_idx_previous = lower_bit_idx_previous.view(-1)
                lower_bit_idx = torch.cat((lower_bit_idx, lower_bit_idx_previous)).unique()
                lower_bit_idx = lower_bit_idx.reshape(1, -1)

        # only inherit the lower bit index from previous layer as the current lower bit index
        elif lower_bit_percent == 0 and lower_bit_idx_previous is not None \
                and (token_remove_percent == 0 or token_remove_percent is None) \
                and self.config.lower_bit_inherit_from_previous_layers \
                and not self.config.weight_quantization_mode:
            if remove_token_idx_previous is not None and remove_token_idx_previous.shape[1] > 0:
                lower_bit_idx_previous_use_here = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                remove_token_idx_previous_use_here = np.sort(remove_token_idx_previous.view(-1).detach().cpu().numpy())
                new_lower_bit_idx_previous = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                point_a = 0
                point_b = 0
                while point_a < len(lower_bit_idx_previous_use_here) \
                        and point_b < len(remove_token_idx_previous_use_here):
                    if lower_bit_idx_previous_use_here[point_a] < remove_token_idx_previous_use_here[point_b]:
                        point_a += 1
                    elif lower_bit_idx_previous_use_here[point_a] == remove_token_idx_previous_use_here[point_b]:
                        raise ValueError("The lower bit index can not equal to token remove index")
                    else:
                        new_lower_bit_idx_previous[point_a:] = new_lower_bit_idx_previous[point_a:] - 1
                        point_b += 1

                lower_bit_idx = torch.from_numpy(
                    new_lower_bit_idx_previous
                ).to(lower_bit_idx_previous.device).reshape(1, -1).long()
            else:
                lower_bit_idx = lower_bit_idx_previous
                lower_bit_idx = lower_bit_idx.reshape(1, -1).long()

        # if only token prune is adopted
        elif (lower_bit_percent == 0 or lower_bit_percent is None) and token_remove_percent > 0 \
                and not self.config.weight_quantization_mode:

            # ========================================================================================
            # accumulate token importance
            if self.config.conventional_token_importance:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([1, attn_probs_shape[2]])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = torch.tril(attn_probs[head_id, :, :])
                    score_token += torch.sum(attn_weights_head_i_here_for_use, dim=0)
                for token_id in range(attn_probs_shape[2]):
                    score_token[0, token_id] = score_token[0, token_id] / (attn_probs_shape[1] - token_id)
            # ========================================================================================

            # ========================================================================================
            # accumulate token importance   method 2
            else:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([attn_probs_shape[2], 1])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = attn_probs[head_id, :, :1]
                    score_token += attn_weights_head_i_here_for_use
                score_token = torch.reshape(score_token, (1, attn_probs_shape[2]))
            # ========================================================================================

            remove_num_tokens = int(attn_probs_shape[2] * token_remove_percent)
            _, remove_token_idx = torch.topk(score_token, remove_num_tokens, dim=1, largest=False)
            remove_token_idx = remove_token_idx.long()

            if self.config.lower_bit_inherit_from_previous_layers and lower_bit_idx_previous is not None:
                if remove_token_idx_previous is not None and remove_token_idx_previous.shape[1] > 0:
                    lower_bit_idx_previous_use_here = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    remove_token_idx_previous_use_here = np.sort(
                        remove_token_idx_previous.view(-1).detach().cpu().numpy())
                    new_lower_bit_idx_previous = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    point_a = 0
                    point_b = 0
                    while point_a < len(lower_bit_idx_previous_use_here) \
                            and point_b < len(remove_token_idx_previous_use_here):
                        if lower_bit_idx_previous_use_here[point_a] < remove_token_idx_previous_use_here[point_b]:
                            point_a += 1
                        elif lower_bit_idx_previous_use_here[point_a] == remove_token_idx_previous_use_here[point_b]:
                            raise ValueError("The lower bit index can not equal to token remove index")
                        else:
                            new_lower_bit_idx_previous[point_a:] = new_lower_bit_idx_previous[point_a:] - 1
                            point_b += 1

                    lower_bit_idx_previous = torch.from_numpy(
                        new_lower_bit_idx_previous
                    ).to(lower_bit_idx_previous.device).reshape(1, -1).long()

                # and remove the intersection of 'lower_bit_idx_previous' and 'remove_token_idx'
                # because the current 'remove_token_idx' may contain the 'lower_bit_idx_previous'
                lower_bit_idx_previous_here = set(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                remove_token_idx_use_here = set(remove_token_idx.view(-1).detach().cpu().numpy())
                need_to_remove_idx = remove_token_idx_use_here & lower_bit_idx_previous_here  # get the intersection
                lower_bit_idx = torch.from_numpy(
                    np.array(list(lower_bit_idx_previous_here - need_to_remove_idx))  # get difference set
                ).to(lower_bit_idx_previous.device).reshape(1, -1).long()

        # if both lower bit quant and token prune are adopted
        elif lower_bit_percent > 0 and token_remove_percent > 0 and not self.config.weight_quantization_mode:
            # ========================================================================================
            # accumulate token importance   method 1
            if self.config.conventional_token_importance:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([1, attn_probs_shape[2]])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = torch.tril(attn_probs[head_id, :, :])
                    score_token += torch.sum(attn_weights_head_i_here_for_use, dim=0)
                for token_id in range(attn_probs_shape[2]):
                    score_token[0, token_id] = score_token[0, token_id] / (attn_probs_shape[1] - token_id)
            # ========================================================================================

            # ========================================================================================
            # accumulate token importance   method 2
            else:
                attn_probs_shape = attn_probs.shape
                score_token = torch.zeros([attn_probs_shape[2], 1])  # token importance initialization
                score_token = score_token.to(attn_probs.device)
                for head_id in range(attn_probs_shape[0]):
                    attn_weights_head_i_here_for_use = attn_probs[head_id, :, :1]
                    score_token += attn_weights_head_i_here_for_use
                score_token = torch.reshape(score_token, (1, attn_probs_shape[2]))
            # ========================================================================================

            remove_num_tokens = int(attn_probs_shape[2] * token_remove_percent)
            _, remove_token_idx = torch.topk(score_token, remove_num_tokens, dim=1, largest=False)
            remove_token_idx = remove_token_idx.long()

            lower_bit_num_tokens = int(attn_probs_shape[2] * lower_bit_percent) + remove_num_tokens
            if lower_bit_num_tokens > attn_probs_shape[2]:  # when token prune ratio + token lower bit ratio > 1
                lower_bit_num_tokens = attn_probs_shape[2]
            _, lower_bit_idx = torch.topk(score_token, lower_bit_num_tokens, dim=1, largest=False)
            lower_bit_idx = lower_bit_idx.long()
            # the first remove_num_tokens indexes should be removed
            # lower_bit_idx = lower_bit_idx[:, remove_num_tokens:]
            lower_bit_idx_here = set(lower_bit_idx.view(-1).detach().cpu().numpy())
            remove_token_idx_here = set(remove_token_idx.view(-1).detach().cpu().numpy())
            lower_bit_idx = torch.from_numpy(
                np.array(list(lower_bit_idx_here - remove_token_idx_here))
            ).to(lower_bit_idx.device).reshape(1, -1).long()

            # update the previous index of lower bit part
            # according to the previous token remove index
            # to merge the new index here
            if self.config.lower_bit_inherit_from_previous_layers and \
                    lower_bit_idx_previous is not None and lower_bit_idx_previous.shape[1] > 0:
                if remove_token_idx_previous is not None and remove_token_idx_previous.shape[1] > 0:
                    lower_bit_idx_previous_use_here = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    remove_token_idx_previous_use_here = np.sort(
                        remove_token_idx_previous.view(-1).detach().cpu().numpy())
                    new_lower_bit_idx_previous = np.sort(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                    point_a = 0
                    point_b = 0
                    while point_a < len(lower_bit_idx_previous_use_here) \
                            and point_b < len(remove_token_idx_previous_use_here):
                        if lower_bit_idx_previous_use_here[point_a] < remove_token_idx_previous_use_here[point_b]:
                            point_a += 1
                        elif lower_bit_idx_previous_use_here[point_a] == remove_token_idx_previous_use_here[point_b]:
                            raise ValueError("The lower bit index can not equal to token remove index")
                        else:
                            new_lower_bit_idx_previous[point_a:] = new_lower_bit_idx_previous[point_a:] - 1
                            point_b += 1

                    lower_bit_idx_previous = torch.from_numpy(
                        new_lower_bit_idx_previous
                    ).to(lower_bit_idx_previous.device).reshape(1, -1)

                # merge 'new_lower_bit_idx_previous' and current 'lower_bit_idx'
                lower_bit_idx_here = set(lower_bit_idx.view(-1).detach().cpu().numpy())
                lower_bit_idx_previous_here = set(lower_bit_idx_previous.view(-1).detach().cpu().numpy())
                lower_bit_idx_here = lower_bit_idx_here | lower_bit_idx_previous_here  # get the union
                # and remove the intersection of 'lower_bit_idx' and 'remove_token_idx'
                # because the current 'remove_token_idx' may contain the 'lower_bit_idx_previous'
                remove_token_idx_use_here = set(remove_token_idx.view(-1).detach().cpu().numpy())
                need_to_remove_idx = remove_token_idx_use_here & lower_bit_idx_previous_here  # get the intersection
                lower_bit_idx = torch.from_numpy(
                    np.array(list(lower_bit_idx_here - need_to_remove_idx))  # get difference set
                ).to(lower_bit_idx.device).reshape(1, -1).long()
        # =============================================================

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if attn_output.shape[1] == 1:
            # only one token there, we do not use low bit and pruning.
            remove_token_idx = None
            lower_bit_idx = None

        # === quant =====
        if self.config.visualization_act:
            visualization_act["after_layer.attn.qact_afterkqv"] = attn_output
        attn_output_here = attn_output
        attn_output = self.qact_afterkqv(attn_output_here)
        if lower_bit_idx is not None and attn_output.shape[1] > 1 and not self.config.weight_quantization_mode:
            attn_output_low_bit = self.qact_afterkqv_2(attn_output_here)
            lower_bit_idx = lower_bit_idx.to(attn_output_low_bit.device)
            attn_output[:, lower_bit_idx, :] = attn_output_low_bit[:, lower_bit_idx, :]
        # === quant =====

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, \
               visualization_act, \
               lower_bit_idx, remove_token_idx


class LlamaDecoderLayer(nn.Module):
    def __init__(
            self,
            config: LlamaConfig,

            lower_bit_percent=None,

            token_remove_percent=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(
            config=config,

            lower_bit_percent=lower_bit_percent,

            token_remove_percent=token_remove_percent,
        )

        self.mlp = LlamaMLP(config)

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # === quant =====
        # input layernorm
        self.input_layernorm_qact = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        # post attention layernorm
        self.post_attention_layernorm_qact = QAct(
            bit_type=BIT_TYPE_DICT['int8'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.post_attention_layernorm_qact_lower = QAct(
            bit_type=BIT_TYPE_DICT['int4'],
            calibration_mode='channel_wise',
            observer_str='minmax',
            quantizer_str='uniform'
        )
        self.lower_bit_percent = lower_bit_percent
        # === quant =====

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,

            lower_bit_idx_previous=None,
            remove_token_idx_previous=None,

    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        visualization_act = None

        if self.config.visualization_act:
            visualization_act = {}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # === quant =====
        if self.config.visualization_act:
            visualization_act["input_layernorm"] = hidden_states
        hidden_states = self.input_layernorm_qact(hidden_states)
        # === quant =====

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, \
        attn_visualization_act, \
        lower_bit_idx_current, remove_token_idx_current = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,

            lower_bit_idx_previous=lower_bit_idx_previous,
            remove_token_idx_previous=remove_token_idx_previous,
        )
        if self.config.visualization_act:
            visualization_act["after_layer.attn"] = attn_visualization_act

        hidden_states = residual + hidden_states

        # ====================================================================================
        # token remove here (we remove token after the residual link)
        if remove_token_idx_current is not None and remove_token_idx_current.shape[1] > 0 \
                and not self.config.weight_quantization_mode:
            remove_token_idx_current_use_here = set(remove_token_idx_current.view(-1).detach().cpu().numpy())
            all_idx = set(np.arange(hidden_states.shape[1]))
            remain_token_idx_current = torch.from_numpy(
                np.array(list(all_idx - remove_token_idx_current_use_here))  # get difference set
            ).to(remove_token_idx_current.device).view(-1)

            hidden_states = hidden_states[:, remain_token_idx_current, :]

        # merge lower bit index
        new_lower_bit_idx_current = None
        if lower_bit_idx_current is not None \
                and not self.config.weight_quantization_mode:
            if remove_token_idx_current is not None and remove_token_idx_current.shape[1] > 0:
                lower_bit_idx_current_use_here = np.sort(lower_bit_idx_current.view(-1).detach().cpu().numpy())
                remove_token_idx_current_use_here = np.sort(remove_token_idx_current.view(-1).detach().cpu().numpy())
                new_lower_bit_idx_current = np.sort(lower_bit_idx_current.view(-1).detach().cpu().numpy())
                point_a = 0
                point_b = 0
                while point_a < len(lower_bit_idx_current_use_here) and point_b < len(
                        remove_token_idx_current_use_here):
                    if lower_bit_idx_current_use_here[point_a] < remove_token_idx_current_use_here[point_b]:
                        point_a += 1
                    elif lower_bit_idx_current_use_here[point_a] == remove_token_idx_current_use_here[point_b]:
                        raise ValueError("The lower bit index can not equal to token remove index")
                    else:
                        new_lower_bit_idx_current[point_a:] = new_lower_bit_idx_current[point_a:] - 1
                        point_b += 1
                new_lower_bit_idx_current = torch.from_numpy(
                    new_lower_bit_idx_current
                ).to(lower_bit_idx_current.device).view(-1).long()
            else:
                new_lower_bit_idx_current = lower_bit_idx_current
        # ====================================================================================

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # === quant =====
        if self.config.visualization_act:
            visualization_act["post_layernorm"] = hidden_states
        hidden_states_here = hidden_states
        hidden_states = self.post_attention_layernorm_qact(hidden_states_here)
        if new_lower_bit_idx_current is not None and not self.config.weight_quantization_mode:
            hidden_states_lower = self.post_attention_layernorm_qact_lower(hidden_states_here)
            hidden_states[:, new_lower_bit_idx_current, :] = hidden_states_lower[:, new_lower_bit_idx_current, :]
        # === quant =====
        hidden_states = self.mlp(hidden_states, new_lower_bit_idx_current)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, visualization_act, lower_bit_idx_current, remove_token_idx_current


LLAMA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value


LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = []
        for i in range(config.num_hidden_layers):
            self.layers.append(
                LlamaDecoderLayer(
                    config,

                    lower_bit_percent=config.lower_bit_percent_total[i],

                    token_remove_percent=config.token_remove_percent_total[i],
                )
            )
        self.layers = nn.ModuleList(self.layers)

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        visualization_act = None
        layer_entropy_record = None

        if self.config.visualization_act:
            visualization_act = {}

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if self.config.token_entropy_computation_mode:
            token_record = []

        # initialize the index
        lower_bit_idx_previous = None
        lower_bit_idx_record = []
        remove_token_idx_previous = None
        token_prune_idx_all = []

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                # if there is token prune
                if hidden_states.shape[1] != attention_mask.shape[2] and not self.config.weight_quantization_mode:
                    attention_mask = torch.ones(
                        (1, hidden_states.shape[1]),
                        dtype=torch.bool, device=inputs_embeds.device
                    )
                    attention_mask = self._prepare_decoder_attention_mask(
                        attention_mask, torch.tensor([batch_size, hidden_states.shape[1]]),
                        inputs_embeds, past_key_values_length
                    )

                    device = input_ids.device if input_ids is not None else inputs_embeds.device
                    position_ids = torch.arange(
                        past_key_values_length, hidden_states.shape[1] + past_key_values_length,
                        dtype=torch.long, device=device
                    )
                    position_ids = position_ids.unsqueeze(0).view(-1, hidden_states.shape[1])

                layer_outputs, \
                visualization_act_layer, \
                lower_bit_idx_current, remove_token_idx_current = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,

                    lower_bit_idx_previous=lower_bit_idx_previous,
                    remove_token_idx_previous=remove_token_idx_previous,
                )
                if self.config.visualization_act:
                    visualization_act["after_layer_{}".format(idx)] = visualization_act_layer

                lower_bit_idx_previous = lower_bit_idx_current
                lower_bit_idx_record.append(lower_bit_idx_current)
                remove_token_idx_previous = remove_token_idx_current
                if remove_token_idx_current is not None:
                    token_prune_idx_all.append(remove_token_idx_current)

            hidden_states = layer_outputs[0]

            # here we get the tokens at each layer and save them
            if self.config.token_entropy_computation_mode:
                token_record.append(hidden_states)

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # we compute the entropy after recording all tokens at each layer
        if self.config.token_entropy_computation_mode:
            layer_entropy_record = []
            for i_th, tokens in enumerate(token_record):
                # in i_th layer
                layer_entropy = 0
                lower_bit_idx_current_layer = lower_bit_idx_record[i_th]
                for j_th in range(tokens.shape[1]):
                    # get the j_th token

                    # get the bit number for this token
                    n_bit_current_layer = self.config.BIT_TYPE_A_total[i_th]
                    if lower_bit_idx_current_layer is not None \
                            and lower_bit_idx_current_layer.shape[
                        1] > 0 and j_th in lower_bit_idx_current_layer:
                        n_bit_current_layer = self.config.BIT_TYPE_A_Lower_total[i_th]
                    n_bit_current_layer = self.config.bit_str2int[n_bit_current_layer]
                    # if want to calculate original model, use 16 here directly     # mark
                    # n_bit_current_layer = 16

                    current_token = tokens[:, j_th, :]
                    # compute the mean of entropy for all batch according to the standard deviation
                    variance_here = torch.std(current_token, dim=1)
                    variance_here = variance_here * n_bit_current_layer ** 2
                    current_entropy = torch.mean(torch.log(variance_here))
                    layer_entropy += current_entropy.to(torch.float32)  # will over the float16 here
                layer_entropy_record.append(layer_entropy)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), visualization_act, layer_entropy_record, token_prune_idx_all, lower_bit_idx_record


class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, \
        visualization_act_model, layer_entropy_record, token_prune_idx_all, lower_bit_idx_record = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        # xuan: for the new evaluation and visualization here
        if not self.config.demo_mode:
            return logits, visualization_act_model, layer_entropy_record, token_prune_idx_all, lower_bit_idx_record

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    LLAMA_START_DOCSTRING,
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
