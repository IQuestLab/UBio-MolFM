# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from e3nn import o3
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode


# Reference:
@compile_mode("unsupported")
class EquivariantLayerNorm(torch.nn.Module):
    NORM_CLAMP = 2**-24  # Minimum positive subnormal for FP16

    def __init__(self, irreps_in, eps=1e-5):
        super().__init__()
        self.irreps_in = irreps_in
        self.eps = eps
        self.layer_norms = []

        for idx, (mul, ir) in enumerate(self.irreps_in):
            self.layer_norms.append(torch.nn.LayerNorm(mul, eps))
        self.layer_norms = torch.nn.ModuleList(self.layer_norms)


    def forward(self, f_in, **kwargs):
        """
        Assume `f_in` is of shape [N, C].
        """
        f_out = []
        channel_idx = 0
        N = f_in.shape[0]
        for degree_idx, (mul, ir) in enumerate(self.irreps_in):
            feat = f_in[:, channel_idx : (channel_idx + mul * ir.dim)]
            feat = feat.reshape(N, mul, ir.dim)
            norm = feat.norm(dim=-1).clamp(min=self.NORM_CLAMP)
            new_norm = self.layer_norms[degree_idx](norm)


            norm = norm.reshape(N, mul, 1)
            new_norm = new_norm.reshape(N, mul, 1)
            feat = feat * new_norm / norm
            feat = feat.reshape(N, -1)
            f_out.append(feat)

            channel_idx += mul * ir.dim

        f_out = torch.cat(f_out, dim=-1)
        return f_out

    def __repr__(self):
        return "{}({}, eps={})".format(
            self.__class__.__name__, self.irreps_in, self.eps
        )


class EquivariantLayerNormV2(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]
        ori_shape = node_input.shape
        N = node_input.shape[:-1].numel()
        node_input = node_input.reshape(N, -1)
        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean

            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output.reshape(ori_shape)


class EquivariantLayerNormV3(nn.Module):
    """
    V2 + Centering for vectors of all degrees
    """

    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, 1, repr]
            field = field - field_mean

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class EquivariantLayerNormV4(nn.Module):
    """
    V3 + Learnable mean shift
    """

    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        mean_shift = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                mean_shift.append(torch.ones(1, mul, 1))
            else:
                mean_shift.append(torch.zeros(1, mul, 1))
        mean_shift = torch.cat(mean_shift, dim=1)
        self.mean_shift = nn.Parameter(mean_shift)

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0
        i_mean_shift = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, 1, repr]
            field_mean = field_mean.expand(-1, mul, -1)
            mean_shift = self.mean_shift.narrow(1, i_mean_shift, mul)
            field = field - field_mean * mean_shift
            i_mean_shift += mul

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


import torch
from torch import nn


class RMS_Norm_SH(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        irreps_in,
        eps=1e-5,
        affine=True,
        normalization="component",
        centering=True,
        std_balance_degrees=True,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.lmax = irreps_in.lmax
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        self.affine_weight_list = nn.ParameterList()
        if self.affine:
            for idx, (mul, ir) in enumerate(self.irreps_in):
                self.affine_weight_list.append(nn.Parameter(torch.ones(1, mul, 1)))
                if ir.l == 0:
                    if self.centering:
                        self.affine_bias = nn.Parameter(torch.zeros(1, mul, 1))
                    else:
                        self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, node_input, batch=None):
        """
        Assume input is of shape [B, L, (l_max+1)^2*C]
        """
        shape = list(node_input.shape[:-1])
        last_dim = node_input.shape[-1]
        num = node_input.shape[:-1].numel()
        node_input = node_input.reshape(num, -1)

        features = []
        start_idx = 0
        feature_norm_record = self.eps
        for idx, (mul, ir) in enumerate(self.irreps_in):
            l = ir.l
            feature = node_input[:, start_idx : start_idx + mul * (2 * l + 1)].reshape(
                -1, mul, (2 * l + 1)
            )
            start_idx += mul * (2 * l + 1)

            if self.centering and ir.l == 0:
                feature = feature - feature.mean(dim=1, keepdim=True)

            if self.normalization == "norm":
                assert not self.std_balance_degrees
                feature_norm = feature.pow(2).sum(dim=2, keepdim=True) * (self.lmax + 1)
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2).sum(dim=2, keepdim=True) / (2 * l + 1)
                else:
                    feature_norm = feature.pow(2).sum(dim=2, keepdim=True) / (
                        self.lmax + 1
                    )  # /(self.lmax+1)**2 * (self.lmax+1)

            feature_norm_record += (
                torch.sum(feature_norm, dim=1) / self.irreps_in.num_irreps
            )

            if self.affine:
                if l == 0 and self.centering:
                    feature = self.affine_weight_list[idx] * feature + self.affine_bias
                else:
                    feature = self.affine_weight_list[idx] * feature

            features.append(feature.reshape(num, -1))

        features = torch.cat(features, dim=-1) * feature_norm_record.pow(-0.5)

        features = features.reshape(shape + [last_dim])
        return features


def get_norm_layer(norm_type):
    if norm_type == "layer":
        return EquivariantLayerNormV2
    elif norm_type == "rms_norm_sh":
        return RMS_Norm_SH
    elif norm_type is None:
        return None
    else:
        raise ValueError(
            "Norm type {} not supported. Only None or layer:EquivariantLayerNormV2 is supported here.".format(
                norm_type
            )
        )


if __name__ == "__main__":
    torch.manual_seed(10)

    irreps_in = o3.Irreps("4x0e+2x1o+1x2e")
    ln = EquivariantLayerNorm(irreps_in, eps=1e-5)
    print(ln)

    inputs = irreps_in.randn(10, -1)
    ln.train()
    outputs = ln(inputs)

    # Check equivariant
    rot = -o3.rand_matrix()
    D = irreps_in.D_from_matrix(rot)

    outputs_before = ln(inputs @ D.T)
    outputs_after = ln(inputs) @ D.T

    print(torch.max(torch.abs(outputs_after - outputs_before)))

    ln2 = EquivariantLayerNormV4(irreps_in)
    outputs2 = ln2(inputs)


# -*- coding: utf-8 -*-
"""
    1. Normalize features of shape (N, sphere_basis, C),
    with sphere_basis = (lmax + 1) ** 2.

    2. The difference from `layer_norm.py` is that all type-L vectors have
    the same number of channels and input features are of shape (N, sphere_basis, C).
"""

import math

import torch
import torch.nn as nn


def get_l_to_all_m_expand_index(lmax):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for l in range(lmax + 1):
        start_idx = l**2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
    return expand_index


class EquivariantLayerNormArray(nn.Module):
    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        for l in range(self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1

            feature = node_input.narrow(1, start_idx, length)

            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean

            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_norm = feature_norm * weight  # [N, 1, C]

            feature = feature * feature_norm

            if self.affine and l == 0:
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias

            out.append(feature)

        out = torch.cat(out, dim=1)

        return out


class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize over L = 0.
    2. Normalize across all m components from degrees L > 0.
    3. Do not normalize separately for different L (L > 0).
    """

    def __init__(
        self,
        lmax,
        num_channels,
        eps=1e-5,
        affine=True,
        normalization="component",
        std_balance_degrees=True,
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees

        self.norm_l0 = torch.nn.LayerNorm(
            self.num_channels, eps=self.eps, elementwise_affine=self.affine
        )

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l**2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
            elif self.normalization == "component":
                if self.std_balance_degrees:
                    feature_norm = feature.pow(
                        2
                    )  # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum(
                        "nic, ia -> nac", feature_norm, self.balance_degree_weight
                    )  # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l**2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)  # [1, C]
                    weight = weight.view(1, 1, -1)  # [1, 1, C]
                    feature_scale = feature_norm * weight  # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)

        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    """

    def __init__(
        self, lmax, num_channels, eps=1e-5, affine=True, normalization="component"
    ):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
        else:
            self.register_parameter("affine_weight", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"

    def forward(self, node_input):
        """
        Assume input is of shape [N, sphere_basis, C]
        """

        out = []

        feature = node_input
        if self.normalization == "norm":
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l**2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)  # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)  # [1, C]
                weight = weight.view(1, 1, -1)  # [1, 1, C]
                feature_scale = feature_norm * weight  # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)

        out = torch.cat(out, dim=1)
        return out
