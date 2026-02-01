# -*- coding: utf-8 -*-
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn



class Activation(str, Enum):
    SquaredReLU = "squared_relu"
    GeLU = "gelu"
    LeakyReLU = "leaky_relu"
    ReLU = "relu"
    SmeLU = "smelu"
    StarReLU = "star_relu"


# For unit testing / parity comparisons, probably not the fastest way
class SquaredReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.relu(x)
        return x_ * x_


class StarReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = torch.nn.functional.relu(x)
        return 0.8944 * x_ * x_ - 0.4472


class SmeLU(nn.Module):
    def __init__(self, beta: float = 2.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu = torch.where(
            x >= self.beta,
            x,
            torch.tensor([0.0], device=x.device, dtype=x.dtype),
        )
        return torch.where(
            torch.abs(x) <= self.beta,
            ((x + self.beta) ** 2).type_as(x) / (4.0 * self.beta),
            relu,
        )


def build_activation(activation: Optional[Activation]):
    if not activation:
        return nn.Identity()

    return {
        Activation.ReLU: nn.ReLU,
        Activation.GeLU: nn.GELU,
        Activation.LeakyReLU: nn.LeakyReLU,
        Activation.SquaredReLU: SquaredReLU,
        Activation.StarReLU: StarReLU,
        Activation.SmeLU: SmeLU,
    }[activation]()


## End of xformers part


def get_linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
    activation: Activation = None,
    dropout: float = 0.0,
):
    """
    Build a linear layer with optional activation and dropout.
    """
    layers = [nn.Linear(in_features=in_features, out_features=out_features, bias=bias)]
    if activation:
        layers.append(build_activation(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def get_feedforward(
    hidden_dim: int,
    activation: Activation,
    hidden_layer_multiplier: int,
    bias: bool = False,
    dropout: float = 0.0,
):
    """
    Build a feedforward layer with optional activation function.
    """
    return nn.Sequential(
        get_linear(
            in_features=hidden_dim,
            out_features=hidden_dim * hidden_layer_multiplier,
            bias=bias,
            activation=activation,
            dropout=dropout,
        ),
        get_linear(
            in_features=hidden_dim * hidden_layer_multiplier,
            out_features=hidden_dim,
            bias=bias,
            activation=None,
            dropout=dropout,
        ),
    )


def no_weight_decay(model):
    # no weight decay on layer norms and embeddings
    no_wd_list = []
    named_parameters_list = [name for name, _ in model.named_parameters()]
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm, nn.RMSNorm)):
            for parameter_name, _ in module.named_parameters():
                if isinstance(module, torch.nn.Linear):
                    if "weight" in parameter_name:
                        continue
                global_parameter_name = module_name + "." + parameter_name
                assert global_parameter_name in named_parameters_list
                no_wd_list.append(global_parameter_name)
    return set(no_wd_list)


def init_linear_weights(module, gain=1.0):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.zero_()




class GraniteRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GraniteRMSNorm is equivalent to T5LayerNorm
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

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class NormalizationType(str, Enum):
    LayerNorm = "layernorm"
    Skip = "skip"
    RMSNorm = "rmsnorm"


class Skip(nn.Module):
    def __init__(self, *_, **__) -> None:
        super().__init__()

    def forward(self, x, **_):
        return x


class LayerNormGraph(nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.node_norm = nn.LayerNorm(hidden_size, **kwargs)
        self.edge_norm = nn.LayerNorm(hidden_size, **kwargs)

    def forward(self, node_features, edge_features):
        node_features = self.node_norm(node_features)
        edge_features = self.edge_norm(edge_features)
        return node_features, edge_features


class RMSNormGraph(nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.node_norm = GraniteRMSNorm(hidden_size, **kwargs)
        self.edge_norm = GraniteRMSNorm(hidden_size, **kwargs)

    def forward(self, node_features, edge_features):
        node_features = self.node_norm(node_features)
        edge_features = self.edge_norm(edge_features)
        return node_features, edge_features


def get_normalization_layer(normalization_type: NormalizationType, is_graph=True):
    return {
        NormalizationType.LayerNorm: LayerNormGraph if is_graph else nn.LayerNorm,
        NormalizationType.Skip: Skip,
        NormalizationType.RMSNorm: RMSNormGraph if is_graph else GraniteRMSNorm,
    }[normalization_type]
