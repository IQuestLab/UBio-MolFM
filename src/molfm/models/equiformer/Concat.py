# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class IrrepsConcat(nn.Module):
    """Scatter for irreps
    ----------
    irreps : `Irreps`
        representation
        concat all irreps of the same shape as irreps together
    """

    def __init__(self, irreps):
        super().__init__()

        self.irreps = Irreps(irreps)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps})"

    def forward(self, node_input, **kwargs):
        """evaluate
        Parameters
        ----------
        node_input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        # the node_input batch slices this into separate graphs
        dim = node_input[0].shape[-1]

        fields = []
        ix = 0
        out_irreps = ""
        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = [input.narrow(1, ix, mul * d) for input in node_input]
            ix += mul * d

            field = [f.reshape(-1, mul, d) for f in field]
            out_field = torch.cat(field, dim=1)
            fields.append(
                out_field.reshape(-1, len(node_input) * mul * d)
            )  # [batch * sample, mul * repr]
            out_irreps += f"{len(node_input) * mul}x{ir} + "
        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output, Irreps(out_irreps[:-3])


if __name__ == "__main__":
    irreps = "128x0e + 128x1e"
    concat = IrrepsConcat(irreps)
    input = [torch.randn(2, 512), torch.randn(2, 512)]
    output, out_irreps = concat(input)
    print(output.shape)
    print(out_irreps)
    print(output)
    print(input)
