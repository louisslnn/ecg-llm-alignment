# This file stores ChatNT and all associated layers and configs

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812



@dataclass
class PerceiverResamplerConfig:
    """
    Parameters to initialize an PerceiverResampler model.
    Args:
        emb_layer_norm_before: Whether to use layer norm before the first attention
            layer.
        attention_heads: Number of attention heads.
        key_size: The dimension of the query, key, and values within each attention
            head, if not specified, it is set to attention_heads//embed_dim.
            It can be useful to set a custom key size if we want to impose the size of
            the query, key and value tensor ( for example, tensors shaped with
            power of 2 are more efficiently handled on TPUs ).
            Note: Parametrizing the model with a custom key size has been done in :
            Brown, Tom, et al. "Language models are few-shot learners."
            Advances in neural information processing systems 33 (2020): 1877-1901.
        embed_dim: Embedding dimension.
        ffn_embed_dim: Feed forward embedding dimension.
        num_layers: Number of attention blocks.
        ffn_activation_name: Activation function to be used in FFN block. Supported
            names are "gelu", "relu", "swish".
        use_glu_in_ffn: Whether to use Gated Linear Unit (GLU) in Feed
            Forward Network (FFN) block. To do a swiGLU (gated-swish) put this arg
            to True and use swish as ffn_activation_name.
            Same principle for a gated-relu. To keep the same number of parameters in
            the FFN block, one should multiply by 2/3 the ffn_embed_dim when using GLU.
            See https://arxiv.org/pdf/2002.05202.pdf for more details.
        resampled_length: length of the resampled output of the module
        use_gradient_checkpointing: Whether to use gradient checkpointing (checkpoint
            gradients in the forward pass to reduce the computation in the backward).
    """

    # architecture
    emb_layer_norm_before: bool = False
    attention_heads: int = 20
    key_size: Optional[int] = None
    embed_dim: int = 1280
    ffn_embed_dim: int = 5120
    num_layers: int = 24
    add_bias_kv: bool = False
    add_bias_ffn: bool = True
    ffn_activation_name: str = "gelu-no-approx"
    use_glu_in_ffn: bool = False
    resampled_length: int = 64

    # performance
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        """
        Checks that the given values are compatible.
        """

        if self.key_size is None:
            if not self.embed_dim % self.attention_heads == 0:
                raise ValueError(
                    f"When no key size is provided, the embedding dimension should be "
                    f"divisible by the number of heads, however provided embedding "
                    f"dimension is {self.embed_dim} and the number of heads is "
                    f"{self.attention_heads}."
                )
            self.key_size = self.embed_dim // self.attention_heads


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        key_size: int,
        # rotary_embedding_config: Optional[RotaryEmbeddingConfigBis] = None,
        add_bias_kv: bool = False,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        if not model_size:
            model_size = key_size * num_heads
        if not value_size:
            value_size = key_size
        self.model_size = model_size
        self.key_size = key_size
        self.value_size = value_size
        self.add_bias_kv = add_bias_kv
        self.name = name
        self.num_heads = num_heads
        # self._rotary_embedding_config = rotary_embedding_config

        self.w_k = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_q = nn.Linear(self.model_size, self.num_heads * self.key_size)
        self.w_v = nn.Linear(self.model_size, self.num_heads * self.value_size)
        self.output = nn.Linear(self.num_heads * self.value_size, self.model_size)
        # if self._rotary_embedding_config:
        #     self._rotary_embedding = RotaryEmbeddingBis(
        #         self.key_size, self._rotary_embedding_config
        #     )

    def apply_rotary_embeddings(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ """
        query, key = self._rotary_embedding(query, key)
        return query, key

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weight_bias: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dictionary containing attention weights
            and outputs.
        """
        key_heads = self.w_k(key).reshape(
            (*key.shape[:-1], self.num_heads, self.key_size)
        )
        query_heads = self.w_q(query).reshape(
            (*query.shape[:-1], self.num_heads, self.key_size)
        )
        value_heads = self.w_v(value).reshape(
            (*value.shape[:-1], self.num_heads, self.value_size)
        )
        # if self._rotary_embedding_config:
        #     query_heads, key_heads = self.apply_rotary_embeddings(
        #         query_heads, key_heads
        #     )
        attention_weights = torch.einsum(
            "...thd, ...Thd -> ...htT", query_heads, key_heads
        )
        sqrt_key_size = np.sqrt(self.key_size)
        attention_weights = attention_weights / sqrt_key_size
        if attention_mask is not None:
            attention_weights = torch.where(attention_mask, attention_weights, -1e30)

        attention_weights = attention_weights.to(value_heads.dtype)

        if attention_weight_bias is not None:
            attention_weights = F.softmax(
                attention_weights + attention_weight_bias, dim=-1
            )
        else:
            attention_weights = F.softmax(attention_weights, dim=-1)

        value_out = torch.einsum(
            "...htT, ...Thd->...thd", attention_weights, value_heads
        )
        value_out = value_out.reshape((*value_out.shape[:-2], -1))
        embeddings = self.output(value_out)

        return {"attention_weights": attention_weights, "embeddings": embeddings}


class TorchMultiModalPerceiverResamplerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        key_size: Optional[int] = None,
        add_bias_kv: bool = False,
        add_bias_ffn: bool = True,
        ffn_activation_name: str = "gelu",
        use_glu_in_ffn: bool = False,
    ):
        super().__init__()

        if key_size is None:
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"Embedding dimension {embed_dim} should be divisible by "
                    f"num_heads {num_heads}."
                )
            key_size = embed_dim // num_heads

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim * 2 if use_glu_in_ffn else ffn_embed_dim
        self.use_glu_in_ffn = use_glu_in_ffn

        self.cross_attention_1 = MultiHeadAttention(
            num_heads=num_heads, key_size=key_size, add_bias_kv=add_bias_kv
        )
        self.cross_attention_2 = MultiHeadAttention(
            num_heads=num_heads, key_size=key_size, add_bias_kv=add_bias_kv
        )

        self.norm_cross_attention_1 = nn.LayerNorm(embed_dim)
        self.norm_cross_attention_2 = nn.LayerNorm(embed_dim)
        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, self.ffn_embed_dim, bias=add_bias_ffn)
        self.fc2 = nn.Linear(self.ffn_embed_dim, embed_dim, bias=add_bias_ffn)

        self.activation_fn = getattr(
            nn.functional, ffn_activation_name, nn.functional.gelu
        )

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_mlp(x)
        if self.use_glu_in_ffn:
            x1, x2 = torch.chunk(self.fc1(x), 2, dim=-1)
            x = self.activation_fn(x1) * x2
        else:
            x = self.activation_fn(self.fc1(x))
        return self.fc2(x)

    def forward(
        self,
        x: torch.Tensor,
        cross_attention_embeddings_1: torch.Tensor,
        cross_attention_embeddings_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        res = x
        x = self.norm_cross_attention_1(x)

        attn_output = self.cross_attention_1(
            query=x,
            key=cross_attention_embeddings_1,
            value=cross_attention_embeddings_1,
            attention_mask=attention_mask_1,
        )["embeddings"]
        x = res + attn_output

        res = x
        x = self.norm_cross_attention_2(x)
        attn_output = self.cross_attention_2(
            query=x,
            key=cross_attention_embeddings_2,
            value=cross_attention_embeddings_2,
            attention_mask=attention_mask_2,
        )["embeddings"]
        x = res + attn_output

        x = x + self.mlp(x)

        return {"embeddings": x}


class TorchMultiModalPerceiverResampler(nn.Module):
    """
    Perceiver Resampler model, made of successive PerceiverResamplerBlocks.
    """

    def __init__(
        self,
        config: PerceiverResamplerConfig,
        name: Optional[str] = None,
    ):
        """
        Initialize a Perceiver Resampler model.
        Args:
            config: Dataclass containing model hyperparameters.
            name: Name for module (custom will break weight loading).
        """
        super().__init__()
        self.config = config
        self.name = name
        self.layers = nn.ModuleList(
            [
                TorchMultiModalPerceiverResamplerBlock(
                    num_heads=self.config.attention_heads,
                    embed_dim=self.config.embed_dim,
                    key_size=self.config.key_size,
                    ffn_embed_dim=self.config.ffn_embed_dim,
                    add_bias_kv=self.config.add_bias_kv,
                    add_bias_ffn=self.config.add_bias_ffn,
                    ffn_activation_name=self.config.ffn_activation_name,
                    use_glu_in_ffn=self.config.use_glu_in_ffn,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.latent_queries = torch.nn.Parameter(
            torch.randn(self.config.resampled_length, self.config.embed_dim)
            * (
                1.0
                / torch.sqrt(torch.tensor(self.config.embed_dim, dtype=torch.float32))
            )
        )

    def apply_attention_blocks(
        self,
        x: torch.Tensor,
        xf_1: torch.Tensor,
        xf_2: torch.Tensor,
        outs: Dict[str, torch.Tensor],
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Create the blocks of attention layers and applies them.
        """
        for layer in self.layers:
            concat_input_1 = torch.cat([xf_1, x], dim=1)
            concat_input_2 = torch.cat([xf_2, x], dim=1)

            output = layer(
                x=x,
                cross_attention_embeddings_1=concat_input_1,
                cross_attention_embeddings_2=concat_input_2,
                attention_mask_1=attention_mask_1,
                attention_mask_2=attention_mask_2,
            )
            x = output["embeddings"]

        return x, outs

    def forward(
        self,
        input_embeddings_1: torch.Tensor,
        input_embeddings_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the embeddings based on the input tokens.
        """
        assert (
            input_embeddings_1.shape[-1] == self.config.embed_dim
        ), "The input embedding dim should match the model embed dim"
        assert (
            input_embeddings_2.shape[-1] == self.config.embed_dim
        ), "The input embedding dim should match the model embed dim"

        batch_size = input_embeddings_1.shape[0]

        latent_queries = self.latent_queries.unsqueeze(0).repeat(batch_size, 1, 1)

        outs: Dict[str, torch.Tensor] = {}
        x = latent_queries

        x, outs = self.apply_attention_blocks(
            x=x,
            xf_1=input_embeddings_1,
            xf_2=input_embeddings_2,
            outs=outs,
            attention_mask_1=attention_mask_1,
            attention_mask_2=attention_mask_2,
        )

        outs["embeddings"] = x

        return outs


class TorchMultiModalPerceiverResamplerProjection(nn.Module):
    def __init__(
        self,
        perceiver_resampler_config: PerceiverResamplerConfig,
        input_embed_dim: int,
        embed_dim: int,
        bio_pad_token_id: int,
        english_pad_token_id: int,
        english_vocab_size: int,
    ):
        super().__init__()
        self.config = perceiver_resampler_config
        self.input_embed_dim = input_embed_dim
        self.embed_dim = embed_dim
        self.bio_pad_token_id = bio_pad_token_id
        self.english_pad_token_id = english_pad_token_id
        self.english_vocab_size = english_vocab_size

        self.bio_projection = nn.Linear(input_embed_dim, embed_dim)
        #self.bio_projection = nn.Linear(input_embed_dim, num_bio_tokens * embed_dim)  
        #self.token_embedding = nn.Embedding(english_vocab_size, embed_dim)
        self.perceiver_resampler = TorchMultiModalPerceiverResampler(config=self.config)

    def forward(
        self,
        bio_token_ids: torch.Tensor,
        bio_embeddings: torch.Tensor,
        english_token_ids: torch.Tensor,
        english_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bio_token_ids (torch.Tensor):
                Shape (batch_size, num_bio_tokens)
            bio_embeddings (torch.Tensor):
                Shape (batch_size, num_bio_tokens, embed_dim)
            english_token_ids (torch.Tensor):
                Shape (batch_size, num_english_tokens)
        """

        #print(f"bio_embeddings shape: {bio_embeddings.shape}")
        projected_bio_embeddings = self.bio_projection(bio_embeddings)
        #print(f"projected_bio_embeddings shape: {projected_bio_embeddings.shape}")

        # B, T, _ = projected_bio_embeddings.shape
        # projected_bio_embeddings = projected_bio_embeddings.view(
        #     B, T * self.num_bio_tokens, self.embed_dim
        # ) 
        #english_embeddings = self.token_embedding(english_token_ids)

        bio_attention_mask = build_perceiver_padding_attention_mask(
            bio_token_ids, self.config.resampled_length, self.bio_pad_token_id
        )
        english_attention_mask = build_perceiver_padding_attention_mask(
            english_token_ids, self.config.resampled_length, self.english_pad_token_id
        )

        projected_embeddings = self.perceiver_resampler(
            input_embeddings_1=projected_bio_embeddings,
            attention_mask_1=bio_attention_mask,
            input_embeddings_2=english_embeddings,
            attention_mask_2=english_attention_mask,
        )["embeddings"]

        return projected_embeddings


def build_perceiver_padding_attention_mask(
    tokens: torch.Tensor, resampled_length: int, pad_token_id: int
) -> torch.Tensor:
    batch_size, seq_len = tokens.shape
    padding_mask = tokens != pad_token_id  # (batch_size, seq_len)

    padding_mask = torch.cat(
        [
            padding_mask,
            torch.ones(
                (batch_size, resampled_length), dtype=torch.bool, device=tokens.device
            ),
        ],
        dim=1,
    )  # (batch_size, seq_len + resampled_length)

    padding_mask = padding_mask[:, None, None, :]
    padding_mask = padding_mask.repeat(1, 1, resampled_length, 1)  # noqa
    return padding_mask