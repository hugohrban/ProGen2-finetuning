# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

# Modified configuration implementation based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/gptj/configuration_gptj.py

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ProGenConfig(PretrainedConfig):
    model_type = "progen"

    def __init__(
        self,
        vocab_size_emb=32,
        vocab_size_lm_head=32,
        n_positions=1024,
        embed_dim=1024,
        n_layer=12,
        n_head=16,
        rotary_dim=32,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size_emb = vocab_size_emb                # input vocab size
        self.vocab_size_lm_head = vocab_size_lm_head        # output vocab size
        self.n_positions = n_positions                      # context window size
        self.embed_dim = embed_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner                              # inner dimension of the MLP
        self.rotary_dim = rotary_dim
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

