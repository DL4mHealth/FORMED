import dataclasses

import torch
from einops import rearrange, repeat
from torch import nn

from .base_model import RMSNorm, PositionalEmbedding


@dataclasses.dataclass
class ClassifierConfig:
    hidden_size: int = 1280
    """The hidden size of the encoder model"""
    intermediate_size: int = 1280
    """The intermediate size of the mlp layer"""
    num_heads: int = 16
    """The number of heads for the multihead attention"""
    dataset_settings_map: dict[str, dict[str, int]] = dataclasses.field(
        default_factory=dict
    )
    use_positional_embedding: bool = True
    """use positional embedding"""
    use_channel_embedding: bool = True
    """use channel embedding"""
    random_scale: float = 1.0
    """scale of random initialization"""


class DotProductDecoder(nn.Module):
    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.mha = nn.MultiheadAttention(
            config.hidden_size, config.num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, 1),
        )

    def forward(self, query: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_class, hidden_size]
        assert len(query.shape) == 3, f"Query shape must be 3D, got {query.shape}"
        # [batch_size, num_features, hidden_size]
        assert (
            len(hidden_state.shape) == 3
        ), f"Hidden state shape must be 3D, got {hidden_state.shape}"
        # Pre-normalization
        hidden_state = self.input_layernorm(hidden_state)
        query = self.input_layernorm(query)
        # MHA with residual connection
        output, _ = self.mha(query, hidden_state, hidden_state, need_weights=False)
        output = output + query
        # MLP for logits
        logits = self.mlp(output).squeeze(-1)
        # [batch_size, num_class]
        return logits


class Classifier(nn.Module):
    def __init__(self, config: ClassifierConfig):
        super().__init__()
        assert len(config.dataset_settings_map) > 0, "Dataset settings map is empty"
        for dataset, settings in config.dataset_settings_map.items():
            assert (
                settings["num_classes"] > 0
            ), f"Number of classes must be greater than 0, got {settings['num_classes']}"
            assert (
                settings["num_channels"] > 0
            ), f"Number of channels must be greater than 0, got {settings['num_channels']}"
        self.config = config

        if config.use_positional_embedding:
            self.positional_embedding = PositionalEmbedding(config.hidden_size)

        if config.use_channel_embedding:
            self.channel_embedding = nn.ParameterDict(
                {
                    dataset: nn.Parameter(
                        torch.randn(settings["num_channels"], config.hidden_size)
                        * config.random_scale
                    )
                    for dataset, settings in config.dataset_settings_map.items()
                }
            )

        self.class_query = nn.ParameterDict(
            {
                dataset: nn.Parameter(
                    torch.randn(settings["num_classes"], config.hidden_size)
                    * config.random_scale
                )
                for dataset, settings in config.dataset_settings_map.items()
            }
        )

        self.decoder = DotProductDecoder(config)

    def forward(self, x: torch.Tensor, dataset: str | None = None) -> torch.Tensor:
        # [batch_size, num_channels, num_features, hidden_size]
        assert len(x.shape) == 4, f"Input shape must be 4D, got {x.shape}"
        batch_size, num_channels, num_features, _ = x.shape
        if dataset is None:
            if len(self.config.dataset_settings_map) == 1:
                dataset = next(iter(self.config.dataset_settings_map))
            else:
                raise ValueError(
                    "Dataset name must be provided if there are multiple datasets"
                )
        if self.config.use_positional_embedding:
            x = x + repeat(
                self.positional_embedding(x.shape[2]),
                "1 n h -> b c n h",
                b=batch_size,
                c=num_channels,
            ).to(x.device)
        if self.config.use_channel_embedding:
            x = x + repeat(
                self.channel_embedding[dataset],
                "c h -> b c n h",
                b=batch_size,
                n=num_features,
            ).to(x.device)
        # [batch_size, num_channels * num_features, hidden_size]
        x = rearrange(x, "b c n h -> b (c n) h")
        # [batch_size, num_classes, hidden_size]
        class_query = repeat(
            self.class_query[dataset], "c h -> b c h", b=batch_size
        ).to(x.device)
        # [batch_size, num_classes]
        logits = self.decoder(class_query, x)
        return logits
