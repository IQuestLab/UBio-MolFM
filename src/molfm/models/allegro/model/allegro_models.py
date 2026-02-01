import math
import torch
from e3nn import o3

from nequip.data import AtomicDataDict
from nequip.data.AtomicDataDict import batched_from_list
from nequip.model import model_builder
from nequip.model.utils import get_current_compile_mode #, _COMPILE_TIME_AOTINDUCTOR_KEY
from nequip.nn import (
    SequentialGraphNetwork,
    ScalarMLP,
    AtomwiseReduce,
    PerTypeScaleShift,
    ForceStressOutput,
)

from nequip.nn.embedding import (
    EdgeLengthNormalizer,
    AddRadialCutoffToData,
    PolynomialCutoff,
)
from allegro.nn import (
    TwoBodySphericalHarmonicTensorEmbed,
    EdgewiseReduce,
    Allegro_Module,
)
from nequip.utils import RankedLogger
from nequip.utils.versions import _TORCH_GE_2_6

from hydra.utils import instantiate
from typing import Sequence, Union, Optional, Dict, Final


logger = RankedLogger(__name__, rank_zero_only=True)


@model_builder
def AllegroEnergyModel(
    l_max: int,
    parity: bool = True,
    **kwargs,
):
    irreps_edge_sh = repr(o3.Irreps.spherical_harmonics(l_max, p=-1))
    # set tensor_track_allowed_irreps
    # note that it is treated as a set, so order doesn't really matter
    if parity:
        # we want all irreps up to lmax
        tensor_track_allowed_irreps = o3.Irreps(
            [(1, (this_l, p)) for this_l in range(l_max + 1) for p in (1, -1)]
        )
    else:
        # we want only irreps that show up in the original SH
        tensor_track_allowed_irreps = irreps_edge_sh

    return FullAllegroEnergyModel(
        irreps_edge_sh=irreps_edge_sh,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        **kwargs,
    )


@model_builder
def AllegroModel(**kwargs):
    r"""Allegro model that predicts energies and forces (and stresses if cell is provided).

    Args:
        seed (int): seed for reproducibility
        model_dtype (str): ``float32`` or ``float64``
        r_max (float): cutoff radius
        per_edge_type_cutoff (Dict): one can optionally specify cutoffs for each edge type [must be smaller than ``r_max``] (default ``None``)
        type_names (Sequence[str]): list of atom type names
        l_max (int): maximum order :math:`\ell` to use in spherical harmonics embedding, 1 is baseline (fast), 2 is more accurate, but slower, 3 highly accurate but slow
        parity (bool): whether to include features with odd mirror parity (default ``True``)
        radial_chemical_embed: an Allegro-compatible two-body radial-chemical embedding module, e.g. ``allegro.nn.TwoBodyBesselScalarEmbed``
        two_body_mlp_hidden_layers_depth (int): number of hidden layers of two-body MLP (default ``1``)
        two_body_mlp_hidden_layers_width (int): depth of hidden layers of two-body MLP
        two_body_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        scalar_embed_output_dim (int): output dimension of the scalar embedding module (default ``None`` will use ``num_scalar_features``)
        num_layers (int): number of Allegro layers
        num_scalar_features (int): multiplicity of scalar features in the Allegro layers
        num_tensor_features (int): multiplicity of tensor features in the Allegro layers
        allegro_mlp_hidden_layers_depth (int): number of hidden layers in the Allegro scalar MLPs (default ``1``)
        allegro_mlp_hidden_layers_width (int): width of hidden layers in the Allegro scalar MLPs (reasonable to set it to be the same as ``num_scalar_features``)
        allegro_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        tp_path_channel_coupling (bool): whether Allegro tensor product weights couple the paths with the channels or not, ``True`` is expected to be more expressive than ``False`` (default ``True``)
        readout_mlp_hidden_layers_depth (int): number of hidden layers in the readout MLP (default ``1``)
        readout_mlp_hidden_layers_width (int): width of hidden layers in the readout MLP (reasonable to set it to be the same as ``num_scalar_features``)
        readout_mlp_nonlinearity (str): ``silu``, ``mish``, ``gelu``, or ``None`` (default ``silu``)
        avg_num_neighbors (float): used to normalize edge sums for better numerics (default ``None``)
        per_type_energy_scales (float/List[float]): per-atom energy scales, which could be derived from the force RMS of the data (default ``None``)
        per_type_energy_shifts (float/List[float]): per-atom energy shifts, which should generally be isolated atom reference energies or estimated from average pre-atom energies of the data (default ``None``)
        per_type_energy_scales_trainable (bool): whether the per-atom energy scales are trainable (default ``False``)
        per_type_energy_shifts_trainable (bool): whether the per-atom energy shifts are trainable (default ``False``)
        pair_potential (torch.nn.Module): additional pair potential term, e.g. ``nequip.nn.pair_potential.ZBL`` (default ``None``)
    """
    return ForceStressOutput(AllegroEnergyModel(**kwargs))


@model_builder
def FullAllegroEnergyModel(
    r_max: float,
    type_names: Sequence[str],
    # irreps
    irreps_edge_sh: Union[int, str, o3.Irreps],
    tensor_track_allowed_irreps: Union[str, o3.Irreps],
    # scalar embed
    radial_chemical_embed: Dict,
    radial_chemical_embed_dim: Optional[int] = None,
    per_edge_type_cutoff: Optional[Dict[str, Union[float, Dict[str, float]]]] = None,
    # scalar embed MLP
    scalar_embed_mlp_hidden_layers_depth: int = 1,
    scalar_embed_mlp_hidden_layers_width: int = 64,
    scalar_embed_mlp_nonlinearity: int = "silu",
    # allegro layers
    num_layers: int = 2,
    num_scalar_features: int = 64,
    num_tensor_features: int = 16,
    allegro_mlp_hidden_layers_depth: int = 1,
    allegro_mlp_hidden_layers_width: int = 64,
    allegro_mlp_nonlinearity: Optional[str] = "silu",
    tp_path_channel_coupling: bool = True,
    # readout
    readout_mlp_hidden_layers_depth: int = 1,
    readout_mlp_hidden_layers_width: int = 32,
    readout_mlp_nonlinearity: Optional[str] = "silu",
    # edge sum normalization
    avg_num_neighbors: Optional[float] = None,
    # allegro layers defaults
    weight_individual_irreps: bool = True,
    scatter_features: bool = False,
    # per atom energy params
    per_type_energy_scales: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_shifts: Optional[Union[float, Sequence[float]]] = None,
    per_type_energy_scales_trainable: Optional[bool] = False,
    per_type_energy_shifts_trainable: Optional[bool] = False,
    pair_potential: Optional[Dict] = None,
    # weight initialization and normalization
    forward_normalize: bool = True,
):
    edge_norm = EdgeLengthNormalizer(
        r_max=r_max,
        type_names=type_names,
        per_edge_type_cutoff=per_edge_type_cutoff,
    )
    radial_chemical_embed_module = instantiate(
        radial_chemical_embed,
        type_names=type_names,
        module_output_dim=(
            num_scalar_features
            if radial_chemical_embed_dim is None
            else radial_chemical_embed_dim
        ),
        forward_weight_init=forward_normalize,
        scalar_embed_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=edge_norm.irreps_out,
    )

    scalar_embed_mlp = ScalarMLP(
        output_dim=num_scalar_features,
        hidden_layers_depth=scalar_embed_mlp_hidden_layers_depth,
        hidden_layers_width=scalar_embed_mlp_hidden_layers_width,
        nonlinearity=scalar_embed_mlp_nonlinearity,
        bias=False,
        forward_weight_init=forward_normalize,
        field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=radial_chemical_embed_module.irreps_out,
    )

    tensor_embed = TwoBodySphericalHarmonicTensorEmbed(
        irreps_edge_sh=irreps_edge_sh,
        num_tensor_features=num_tensor_features,
        forward_weight_init=forward_normalize,
        scalar_embedding_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        tensor_basis_out_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_embedding_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=scalar_embed_mlp.irreps_out,
    )

    use_custom_kernel: Final[bool] = (
        False #get_current_compile_mode() == _COMPILE_TIME_AOTINDUCTOR_KEY
        and _TORCH_GE_2_6
        and torch.cuda.is_available()
    )
    if use_custom_kernel:
        logger.info(
            "Allegro model will be built to use custom TP kernels wherever possible (i.e. when specific shape conditions are met and running on GPUs)."
        )

    allegro_kwargs = dict(
        num_layers=num_layers,
        num_scalar_features=num_scalar_features,
        num_tensor_features=num_tensor_features,
        tensor_track_allowed_irreps=tensor_track_allowed_irreps,
        avg_num_neighbors=avg_num_neighbors,
        # MLP
        latent_kwargs={
            "hidden_layers_depth": allegro_mlp_hidden_layers_depth,
            "hidden_layers_width": allegro_mlp_hidden_layers_width,
            "nonlinearity": allegro_mlp_nonlinearity,
            "bias": False,
            "forward_weight_init": forward_normalize,
        },
        tp_path_channel_coupling=tp_path_channel_coupling,
        tp_use_custom_kernels=use_custom_kernel,
        weight_individual_irreps=weight_individual_irreps,
        scatter_features=scatter_features,
        # fields
        tensor_basis_in_field=AtomicDataDict.EDGE_ATTRS_KEY,
        tensor_features_in_field=AtomicDataDict.EDGE_FEATURES_KEY,
        scalar_in_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
        scalar_out_field=AtomicDataDict.EDGE_FEATURES_KEY,
        irreps_in=tensor_embed.irreps_out,
    )
    while True:
        try:
            allegro = Allegro_Module(**allegro_kwargs)
            break
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg:
                key = msg.split("'")[1]
                allegro_kwargs.pop(key, None)
                continue
            raise

    modules = {
        "edge_norm": edge_norm,
        "radial_chemical_embed": radial_chemical_embed_module,
        "scalar_embed_mlp": scalar_embed_mlp,
        "tensor_embed": tensor_embed,
        "allegro": allegro,
    }

    edge_readout = ScalarMLP(
        output_dim=1,
        hidden_layers_depth=readout_mlp_hidden_layers_depth,
        hidden_layers_width=readout_mlp_hidden_layers_width,
        nonlinearity=readout_mlp_nonlinearity,
        bias=False,
        forward_weight_init=forward_normalize,
        field=AtomicDataDict.EDGE_FEATURES_KEY,
        out_field=AtomicDataDict.EDGE_ENERGY_KEY,
        irreps_in=allegro.irreps_out,
    )
    edge_eng_sum = EdgewiseReduce(
        field=AtomicDataDict.EDGE_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        factor=1.0 / math.sqrt(2 * avg_num_neighbors),
        irreps_in=edge_readout.irreps_out,
    )

    per_type_energy_scale_shift = PerTypeScaleShift(
        type_names=type_names,
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        scales=per_type_energy_scales,
        shifts=per_type_energy_shifts,
        scales_trainable=per_type_energy_scales_trainable,
        shifts_trainable=per_type_energy_shifts_trainable,
        irreps_in=edge_eng_sum.irreps_out,
    )

    modules.update(
        {
            "edge_readout": edge_readout,
            "edge_eng_sum": edge_eng_sum,
            "per_type_energy_scale_shift": per_type_energy_scale_shift,
        }
    )

    prev_irreps_out = per_type_energy_scale_shift.irreps_out
    if pair_potential is not None:

        # case where model doesn't have edge cutoffs up to this point, but pair potential required
        if AtomicDataDict.EDGE_CUTOFF_KEY not in prev_irreps_out:
            cutoff = AddRadialCutoffToData(
                cutoff=PolynomialCutoff(6),
                irreps_in=prev_irreps_out,
            )
            prev_irreps_out = cutoff.irreps_out
            modules.update({"cutoff": cutoff})

        pair_potential = instantiate(
            pair_potential,
            type_names=type_names,
            irreps_in=prev_irreps_out,
        )
        prev_irreps_out = pair_potential.irreps_out
        modules.update({"pair_potential": pair_potential})

    total_energy_sum = AtomwiseReduce(
        irreps_in=prev_irreps_out,
        reduce="sum",
        field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
        out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
    )
    modules.update({"total_energy_sum": total_energy_sum})

    return SequentialGraphNetwork(modules)


@model_builder
def FullAllegroModel(**kwargs):
    return ForceStressOutput(FullAllegroEnergyModel(**kwargs))


from nequip.utils.global_state import set_global_state
from torch import nn
from nequip.data.transforms import (
    ChemicalSpeciesToAtomTypeMapper,
    NeighborListTransform,
)
from nequip.data.AtomicDataDict import batched_from_list


class Allegro_Wrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        set_global_state(allow_tf32=False) 
        allegro_params = {
            "seed": 456,                      # seed for reproducibility
            "model_dtype": "float32",         # “float32” or “float64”

            "r_max": 5.0,                     # cutoff radius
            # "per_edge_type_cutoff": None,   

            "type_names": ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'],    # list of atom type names

            "l_max": 1,                       # maximum spherical-harmonic order
            # "parity": True,                   # whether to include odd-parity features

            # Two-body radial-chemical embedding
            "radial_chemical_embed": {
                "_target_": "allegro.nn.TwoBodyBesselScalarEmbed",
                "num_bessels": 8,
                "bessel_trainable": False,
                "polynomial_cutoff_p": 6,
            },

            # Two-body MLP
            # "two_body_mlp_hidden_layers_depth": 1,
            # "two_body_mlp_hidden_layers_width": 64,
            # "two_body_mlp_nonlinearity": "silu",

            # "scalar_embed_output_dim": 64,

            "num_layers": 2,
            "num_scalar_features": 64,
            "num_tensor_features": 32,

            "allegro_mlp_hidden_layers_depth": 1,
            "allegro_mlp_hidden_layers_width": 64,
            "allegro_mlp_nonlinearity": "silu",

            "tp_path_channel_coupling": True,

            "readout_mlp_hidden_layers_depth": 1,
            "readout_mlp_hidden_layers_width": 64,
            "readout_mlp_nonlinearity": "silu",

            "avg_num_neighbors": 20,

            "per_type_energy_scales": 1.0,
            "per_type_energy_shifts": 0.0,
            "per_type_energy_scales_trainable": False,
            "per_type_energy_shifts_trainable": False,

            "pair_potential": {
                "_target_": "nequip.nn.pair_potential.ZBL",
                "units": "real",
                "chemical_species": ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi'],
            },
        }
        self.model = AllegroModel(**allegro_params)

        type_names = allegro_params["type_names"]
        r_max     = allegro_params["r_max"]

        self.transforms = [
            ChemicalSpeciesToAtomTypeMapper(chemical_symbols=type_names),
            NeighborListTransform(r_max=r_max),
        ] 

    def forward(self, batched_data):
        allegro_data = self.transform_data(batched_data)
        
        decoder_output= self.model(allegro_data)
    
        result = {
            "pred_energy_per_atom": decoder_output["atomic_energy"],
            "pred_energy": decoder_output["total_energy"],
            "pred_forces": decoder_output["forces"],
            "node_irrepsBxN": None,
            "node_featuresBxN": None,
            "node_vec_featuresBxN": None,
        }

        return result
    
    def transform_data(self, batched_data):
        # This is a placeholder function and should be implemented based on the model's requirements
        B, N, _ = batched_data.pos.shape
        per_frame = []
        atom_mask = ~batched_data.non_atom_mask
        for i in range(B):
            frame = {
                # nodes
                "pos":             batched_data.pos[i][atom_mask[i]],                         # [N,3]
                "atomic_numbers":  (batched_data.node_attr[i].squeeze(-1).long())[atom_mask[i]],# [N]
                "total_energy":          batched_data.energy[i].unsqueeze(0),         # [1]
                "forces":          (batched_data.forces[i])[atom_mask[i]],                     # [N,3]
            }
            per_frame.append(frame)

        # batch the list of dicts into one AtomicDataDict
        atomic_batch = batched_from_list(per_frame)

        # apply the same transforms as on your dataset
        self.transforms[0].lookup_table = self.transforms[0].lookup_table.to(atomic_batch["pos"].device)
        for tf in self.transforms:
                atomic_batch = tf(atomic_batch)

        return atomic_batch
