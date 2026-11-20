# -*- coding: utf-8 -*-
import os
from typing import Dict

import e3nn
import torch
import torch.nn as nn

from molfm.models.e2former.mr_routing import resolve_mr_is_light

kcalmol_to_ev = 0.0433634

class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """

    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="silu",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        act_class_mapping = {
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        act_class = act_class_mapping[activation]
        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            act_class(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)

    def forward(self, x, v):
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(-2) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class PureLinearForce(nn.Module):
    def __init__(self, embedding_dim,lmax = 2, activation="silu"):
        super(PureLinearForce, self).__init__()
        # self.output_network = nn.ModuleList(
        #     [
        #         GatedEquivariantBlock(
        #             embedding_dim,
        #             embedding_dim // 2,
        #             activation=activation,
        #             scalar_activation=True,
        #         ),
        #         GatedEquivariantBlock(embedding_dim // 2, 1, activation=activation),
        #     ]
        # )
        self.sing_linear = nn.Linear(embedding_dim,1,bias = False)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     for layer in self.output_network:
    #         layer.reset_parameters()

    def forward(self, x, v, batched_data,pred_energy,**kwargs):
        v = self.sing_linear(v)
        # for layer in self.output_network:
        #     x, v = layer(x, v)
        return v.squeeze(-1)


class PAINNLinearForce(nn.Module):
    def __init__(self, embedding_dim, lmax = 2,activation="silu",*args,**kwargs):
        super(PAINNLinearForce, self).__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    embedding_dim,
                    embedding_dim // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(embedding_dim // 2, 1, activation=activation),
            ]
        )
        # self.sing_linear = nn.Linear(embedding_dim,1,bias = False)
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     for layer in self.output_network:
    #         layer.reset_parameters()

    def forward(self, x, v, batched_data,pred_energy,**kwargs):
        # v = self.sing_linear(v)
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze(-1)
class GradientHead(torch.nn.Module):
    def __init__(
        self,
    ):
        super(GradientHead, self).__init__()



    def forward(
        self,
        x,v,batched_data,pred_energy,**kwargs
    ):
        grad_outputs = [torch.ones_like(pred_energy)]

        grad_pos = torch.autograd.grad(
            outputs=pred_energy,
            inputs=batched_data["pos"],
            grad_outputs=grad_outputs,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if grad_pos is None:
            pred_forces = torch.zeros_like(batched_data["pos"])
        else:
            pred_forces = -grad_pos
        # if self.training == False:
        #     pred_forces = pred_forces.detach()
        return pred_forces

class MLP_energy_head(nn.Module):
    def __init__(self, embedding_dim, lmax=2, final_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # final_bias=False drops the last layer's additive per-atom constant.
        # Used by the LIGHT head of cross_mode="interleaved_flash", where the
        # energy is a sum over two heads on two DISJOINT atom sets: a bias on
        # each of them, on top of the per-element reference energies, is three
        # ways to spell the same constant, and the redundancy shows up as a slow
        # drift between the two heads rather than as a training failure.
        # Default True -> bit-exact and shape-identical for every existing head.
        self.mlp = nn.Sequential(
                            nn.Linear(embedding_dim, embedding_dim, bias=True),
                            nn.SiLU(),
                            nn.Linear(embedding_dim, embedding_dim, bias=True),
                            nn.SiLU(),
                            nn.Linear(embedding_dim, 1, bias=bool(final_bias)),
                        )
    def forward(self,embedding,irreps):
        return self.mlp(embedding)



class MDEnergyForceMultiHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_fn = args.loss_fn.lower()
        self.energy_loss_weight = args.energy_loss_weight
        self.force_loss_weight = args.force_loss_weight
        self.auto_grad = args.AutoGradForce
        self.with_moe = args.with_moe
        if self.with_moe == "escn":
            raise NotImplementedError(
                "with_moe='escn' is not supported in this release; use with_moe="
                "'painn' or leave it disabled."
            )
        if args.backbone_config["name"] == "e2former":
            embedding_dim = e3nn.o3.Irreps(
                args.backbone_config["irreps_node_embedding"]
            )[0][0]
            lmax = len(e3nn.o3.Irreps(args.backbone_config["irreps_node_embedding"]))-1
        else:
            embedding_dim = 256

        # The backbone can concatenate the outputs of selected blocks along the
        # channel axis (backbone_config.cat_layer_output). Its output channel dim
        # is then n_slots * embedding_dim, so widen the heads by the same factor;
        # both sides call cat_layer_slot_indices so they cannot drift apart.
        self.cat_layer_output = bool(
            args.backbone_config.get("cat_layer_output", False)
        )
        if self.cat_layer_output:
            _cat_mr_cfg = args.backbone_config.get("mixed_resolution", {}) or {}
            if _cat_mr_cfg.get("enabled", False):
                from molfm.models.e2former.e2former_mixed_resolution import (
                    cat_layer_slot_indices,
                )
                n_slots = len(cat_layer_slot_indices(
                    args.backbone_config.get("num_layers", 6),
                    True,
                    args.backbone_config.get("cat_layer_indices", None),
                ))
            else:
                raise NotImplementedError(
                    "cat_layer_output is only supported for the mixed-resolution "
                    "backbone in this release."
                )
            embedding_dim = embedding_dim * n_slots

        # Mixed-resolution energy head configuration. In "distinct" mode the
        # head uses a separate energy head for light atoms (those whose atomic
        # number is in mixed_resolution.lf_atoms, typically H). This mirrors
        # the training-time setup so checkpoints trained with
        # energy_head="distinct" load and run correctly here.
        if args.backbone_config.get("name") == "e2former":
            mr_cfg = args.backbone_config.get("mixed_resolution", {}) or {}
        else:
            mr_cfg = {}
        self._energy_head_mode = mr_cfg.get("energy_head", "standard")
        self._mr_enabled = bool(mr_cfg.get("enabled", False))
        self._lf_atoms = list(mr_cfg.get("lf_atoms", [1]))

        # cross_mode="interleaved_flash": the backbone carries TWO persistent node
        # states and never projects the light one into the heavy irreps tensor, so
        # that tensor's LIGHT ROWS ARE STRUCTURALLY ZERO. The light energy head
        # therefore reads a different tensor (published by the backbone as
        # batched_data["node_irreps_lightBxN"]) at the LIGHT dims, not the heavy
        # ones.
        self._flash = bool(
            self._mr_enabled
            and mr_cfg.get("cross_mode", "") == "interleaved_flash"
        )
        assert not self.cat_layer_output or not self._flash, (
            "cat_layer_output is not supported by "
            "cross_mode='interleaved_flash' (a flash block writes exactly one "
            "of the two streams, so \"the output of block k\" is not one "
            "object across the atom set)."
        )
        self._light_embedding_dim = None
        self._light_lmax = None
        if self._flash:
            _light_irreps = e3nn.o3.Irreps(
                mr_cfg.get("irreps_light_embedding", "32x0e")
            )
            self._light_embedding_dim = _light_irreps[0][0]
            self._light_lmax = _light_irreps[-1][1][0]
            # Not defensive programming -- each of these fails SILENTLY.
            assert self._energy_head_mode == "distinct", (
                "cross_mode='interleaved_flash' requires "
                "mixed_resolution.energy_head='distinct'; a single energy head "
                "would read the structurally-zero light rows of the heavy irreps "
                "tensor and give every light atom the same constant energy."
            )
            # AutoGradForce=False is SUPPORTED, but only through a second force
            # head. A single direct head (PureLinearForce / PAINNLinearForce)
            # reads the heavy irreps tensor's l=1 block, whose light rows are
            # structurally zero here, so it would predict EXACTLY 0 eV/A on every
            # light atom -- silently. Below, a light force head is built at the LIGHT
            # dims and fed the light stream; the two are selected by is_light_2d,
            # never summed. That needs a light
            # state with a vector part: lmax_light >= 1. The backbone already
            # asserts this unconditionally under flash, so this is a restatement
            # at the point of use rather than a new constraint.
            if not self.auto_grad:
                assert self._light_lmax >= 1, (
                    "cross_mode='interleaved_flash' with AutoGradForce=False needs "
                    "mixed_resolution.irreps_light_embedding to carry lmax_light >= "
                    f"1, got {self._light_lmax}. The light force head predicts a "
                    "VECTOR from the light stream's l=1 block; a scalar light state "
                    "has none, and the heavy tensor's light rows are structurally "
                    "zero, so every light atom would get exactly 0 eV/A."
                )
        # Initialize multiple energy and force heads based on dataset_name_list
        self.energy_heads = nn.ModuleDict()
        self.force_heads = nn.ModuleDict()
        # Populated only when MR is enabled with energy_head="distinct".
        self.energy_heads_light = nn.ModuleDict()
        # Populated ONLY under cross_mode="interleaved_flash" with a DIRECT force
        # head (AutoGradForce=False). Empty everywhere else, so no existing
        # checkpoint gains or loses a key.
        self.force_heads_light = nn.ModuleDict()

        for dataset_name in args.dataset_name_list.split(","):
            head_name = dataset_name
            if head_name not in self.energy_heads:
                self.energy_heads[head_name] = MLP_energy_head(embedding_dim, lmax)
                if self.auto_grad:
                    self.force_heads[head_name] = GradientHead()
                elif self.with_moe == 'painn':
                    self.force_heads[head_name] = PAINNLinearForce(embedding_dim, lmax)
                else:
                    self.force_heads[head_name] = PureLinearForce(embedding_dim, lmax)

                # "distinct" mode: separate energy head for light atoms with the
                # same architecture as the heavy head.
                if self._energy_head_mode == "distinct" and self._mr_enabled:
                    if self._flash:
                        # Sized at the LIGHT dims: under flash this head is fed the
                        # light stream, not the heavy one. No final bias -- see
                        # MLP_energy_head's final_bias docstring.
                        self.energy_heads_light[head_name] = MLP_energy_head(
                            self._light_embedding_dim, self._light_lmax,
                            final_bias=False,
                        )
                    else:
                        self.energy_heads_light[head_name] = MLP_energy_head(embedding_dim, lmax)

                # cross_mode="interleaved_flash" + DIRECT force: a second force
                # head at the LIGHT dims, same class as the heavy one so the two
                # streams are read the same way. It is fed the light stream
                # (node_irreps_lightBxN), never the heavy tensor, and its output
                # is selected -- not added -- by is_light_2d in forward().
                if self._flash and not self.auto_grad:
                    if self.with_moe == 'painn':
                        self.force_heads_light[head_name] = PAINNLinearForce(
                            self._light_embedding_dim, self._light_lmax
                        )
                    else:
                        self.force_heads_light[head_name] = PureLinearForce(
                            self._light_embedding_dim, self._light_lmax
                        )

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor],
                        batched_data: Dict[str, torch.Tensor], 
                        need_force = True):
        if "pred_energy" in result_dict:return result_dict

        decoder_x_output = result_dict["node_featuresBxN"]
        decoder_vec_output = result_dict["node_vec_featuresBxN"]
        decoder_irreps_output = result_dict["node_irrepsBxN"]
        
        dataset_name = batched_data["data_name"][0]
        head_name = dataset_name

        # ---- cross_mode="interleaved_flash": pick up the SECOND node state -------
        # The backbone publishes the light stream and the mask the two streams were
        # actually split by on the container it returns as "data" (so the mask can
        # never disagree with the routing). Hoisted out of the energy branch because
        # the DIRECT force path below needs exactly the same two tensors: under flash
        # both heads -- energy and force -- are two-headed on two disjoint atom sets.
        _flash_light_irreps = None
        _flash_is_light = None
        if self._flash:
            _flash_data = result_dict.get("data", None)
            assert (
                _flash_data is not None
                and "node_irreps_lightBxN" in _flash_data
                and "is_light_2d" in _flash_data
            ), (
                "cross_mode='interleaved_flash' but the backbone output carries no "
                "'node_irreps_lightBxN'. The light energy/force heads have nothing to "
                "read, and the heavy irreps tensor's light rows are structurally zero "
                "-- falling back to it would silently give every light atom the same "
                "constant energy and exactly 0 eV/A of force. Check that the backbone "
                "really ran the flash branch (E2FormerBackbone.forward must return "
                "'data')."
            )
            _flash_light_irreps = _flash_data["node_irreps_lightBxN"]
            _flash_is_light = _flash_data["is_light_2d"].bool()

        if (
            self._energy_head_mode == "distinct"
            and self._mr_enabled
            and head_name in self.energy_heads_light
        ):
            # Flash reuses the mask the backbone published for its own light
            # stream. Every other mode goes through the canonical resolver: an
            # explicit batched_data["is_light"] (which the wrapper sets, carrying
            # the HOH-O mask) wins over a fresh atomic_numbers + hoh_o_mask
            # recomputation. Recomputing here would quietly re-route to the heavy
            # stream any water oxygen whose routing depends on neighbours the
            # recomputation does not see, and the head would then charge its
            # energy to the heavy head.
            if self._flash:
                is_light = _flash_is_light
            else:
                lf_atoms_t = torch.tensor(
                    self._lf_atoms,
                    device=decoder_x_output.device,
                    dtype=batched_data["atomic_numbers"].dtype,
                )
                is_light = resolve_mr_is_light(
                    batched_data, lf_atoms_t, where="MDEnergyForceMultiHead.update_loss",
                )
            atom_energy_heavy = self.energy_heads[head_name](
                decoder_x_output, decoder_irreps_output
            ).squeeze(-1)
            if self._flash:
                # The two heads no longer accept the same tensor, so the light one
                # is fed the light stream: [B, L, (lmax_light+1)^2, C_L], zero on
                # heavy rows. Both heads are still evaluated everywhere and
                # selected by torch.where rather than index-scattered: the scatter
                # needs a nonzero() (a host sync and a graph break under
                # torch.compile) on every forward, and the wasted work is the light
                # MLP at C_L=64 against a heavy one at C_H=256 -- cheap. The
                # selection is what makes it correct: no atom is ever charged both
                # heads, so there is no double-counted bias.
                _light_irreps_bl = _flash_light_irreps
                atom_energy_light = self.energy_heads_light[head_name](
                    _light_irreps_bl[:, :, 0], _light_irreps_bl
                ).squeeze(-1)
            else:
                atom_energy_light = self.energy_heads_light[head_name](
                    decoder_x_output, decoder_irreps_output
                ).squeeze(-1)
            atom_energy = torch.where(is_light, atom_energy_light, atom_energy_heavy)
        else:
            atom_energy = self.energy_heads[head_name](
                decoder_x_output, decoder_irreps_output
            ).squeeze(-1)
        atom_energy = atom_energy.masked_fill(~batched_data["atom_masks"], 0.0)
        pred_energy = atom_energy.sum(dim=-1)
        pred_energy_per_atom = pred_energy / torch.sum(batched_data["atom_masks"],dim = -1).reshape(-1)
        

        # If dataset_name is not "deshaw_400" or "deshaw_650", process force head
        # if dataset_name not in ["deshaw_400", "deshaw_650"]:

        pred_forces = None
        if need_force:
            pred_forces = self.force_heads[head_name](
                decoder_x_output, decoder_vec_output,batched_data,pred_energy,irreps = decoder_irreps_output
            )
            if self._flash and not self.auto_grad:
                # DIRECT force under flash. The call above ran the HEAVY head on
                # the heavy irreps tensor, whose light rows are structurally zero,
                # so its light entries are exactly 0 and meaningless. Run the light
                # head on the light stream's l=1 block and SELECT -- torch.where,
                # not a sum: the two heads own disjoint atom sets, and both are
                # evaluated everywhere on purpose (an index_put_ would need a
                # nonzero(), i.e. a host sync and a graph break under
                # torch.compile, to save the light MLP at C_L against a heavy one
                # at C_H). The light head returns identically 0 on heavy rows
                # anyway -- its input there is zero and every path to the output
                # vector is linear in v -- but the selection is what makes it
                # correct, not that coincidence.
                _f_light = self.force_heads_light[head_name](
                    _flash_light_irreps[:, :, 0],
                    _flash_light_irreps[:, :, 1:4],
                    batched_data,
                    pred_energy,
                    irreps=_flash_light_irreps,
                )
                pred_forces = torch.where(
                    _flash_is_light.unsqueeze(-1), _f_light, pred_forces
                )

        # print(f"f_pred{pred_forces} decoder_vec_output{torch.mean(torch.abs(decoder_vec_output))}")
        result_dict.update({
                "pred_energy":pred_energy,
                "pred_forces":pred_forces,
                "pred_energy_per_atom":pred_energy_per_atom})
        return result_dict
        # {
        #         "f_force_weight":result_dict["f_force_weight"],
        #         "pred_energy":pred_energy,
        #         "pred_forces":pred_forces,
        #         "pred_energy_per_atom":pred_energy_per_atom}

    def update_loss(self, loss, log_ouput, model_output, batched_data):
        # Extract information from batched data
        data_name = batched_data["data_name"][0]
        e_pred = model_output["pred_energy"].reshape(-1)
        f_pred = model_output.get("pred_forces", None)
        e_pred_peratom = model_output["pred_energy_per_atom"].reshape(-1)

        e_true = batched_data["energy"].reshape(-1)
        f_true = batched_data.get("forces", None)
        e_true_peratom = batched_data["energy_per_atom"].reshape(-1)

        # Unit conversion if needed
        if self.args.loss_unit == "kcal/mol":
            e_true /= kcalmol_to_ev
            e_true_peratom /= kcalmol_to_ev
            if f_true is not None:
                f_true /= kcalmol_to_ev
        elif (self.args.loss_unit).lower() == "mev":
            e_true /= 0.001
            e_true_peratom /= 0.001
            if f_true is not None:
                f_true /= 0.001
        elif (self.args.loss_unit).lower() == "normev":
            e_true /= 1.5
            e_true_peratom /= 1.5
            if f_true is not None:
                f_true /= 1.5

        
        
        # Calculate energy loss

        if self.loss_fn == "mse":
            # e_loss = torch.mean((e_pred - e_true) ** 2)
            e_loss = torch.mean((e_pred_peratom - e_true_peratom) ** 2)
        elif "atom" in self.loss_fn:
            # e_loss = torch.mean(torch.abs(e_pred - e_true))
            e_loss = torch.mean(torch.abs(e_pred_peratom - e_true_peratom))
        else:
            e_loss = torch.mean(torch.abs(e_pred - e_true))

        loss = self.energy_loss_weight * e_loss #e_loss
        log_ouput = {}
        # Batch size
        bs = e_true.shape[0]
        if f_pred is not None and f_true is not None:
            masks = batched_data["atom_masks"]&(~(batched_data["fixed"].bool()))
            # ---- light / heavy split for force MAE logging (MR only) ----
            # Uses the canonical is_light mask built by E2FormerBackbone, which already OR's
            # in hoh_o_mask when lf_O_in_HOH=True. It arrives via model_output because under
            # DDP the input containers are scattered before forward, so the copy the backbone
            # writes onto never reaches this function; batched_data is kept as a fallback for
            # callers that hand the mask in directly.
            is_light_2d = model_output.get("is_light", None)
            if is_light_2d is None:
                is_light_2d = batched_data.get("is_light", None)
            if is_light_2d is not None:
                is_light_2d = is_light_2d.bool()
                light_masks = masks & is_light_2d
                heavy_masks = masks & (~is_light_2d)
                if torch.sum(light_masks) > 0:
                    log_ouput[f"force_loss_light_{data_name}"] = torch.mean(
                        torch.abs(f_pred[light_masks] - f_true[light_masks])
                    )
                if torch.sum(heavy_masks) > 0:
                    log_ouput[f"force_loss_heavy_{data_name}"] = torch.mean(
                        torch.abs(f_pred[heavy_masks] - f_true[heavy_masks])
                    )
            # ---- per-subset (data_src) force/energy MAE for SPICE ----
            # SPICE's published leaderboard (E2Former arXiv:2501.19216 Table 3; MACE-OFF23
            # Table 1) reports seven per-subset Energy/Force MAE pairs plus an "All" that is a
            # STRUCTURE-COUNT-WEIGHTED mean of those per-subset MAEs. The global
            # force_loss_spice below is an atom-pooled mean over all components instead, and
            # the two bases are not interchangeable (verified on the identical MACE-OFF-L
            # reference: 12.260 structure-weighted vs 13.389 atom-pooled, 1.092x). Emitting
            # the per-subset pieces lets an external tool rebuild the published basis; a bare
            # global mean cannot be re-weighted after the fact.
            #
            # WHY SUM *AND* COUNT, AND WHY THE RATIO IS EXACT. `LogAccumulator.averge_log`
            # np.mean's each key over the batches that wrote it, and DDP's `_allreduceAVG`
            # then takes an elementwise MEAN over ranks. Both normalizations are unknown from
            # the log (they differ per subset, because a subset is absent from many batches).
            # So a logged MEAN is a batch-mean, not the pooled mean we want. A logged SUM,
            # divided by the COUNT that shares its exact normalizer, cancels it: for one
            # subset, logged_sum/logged_count == sum|dF| / sum(components) over every sample
            # in the test set, i.e. the exact pooled per-subset MAE, independent of batching,
            # rank count and the accumulator's averaging. That ratio -- not the logged mean --
            # is what the aggregation script may consume. (`force_loss_src*` is written too,
            # for a quick read, but it is the batch-caliber value and must not be pooled.)
            #
            # VALIDATION-ONLY. The per-subset subtraction would pay a full-size
            # allocation on every training step for a metric training never reads.
            # No effect on loss: this only writes into log_ouput.
            #
            # `data_src` arrives as a 1-D per-sample long tensor (SPICExyzDataset.__getitem__
            # emits one id per frame, from its subset_name2id table: PubChem 0, DES370K
            # Monomers 1, DES370K Dimers 2, Dipeptides 3, Solvated Amino Acids 4, water 5,
            # QMugs 6). If a caller does not supply it we log nothing, because silently
            # omitting the per-subset metrics is exactly the failure this block prevents.
            if data_name == "spice" and not self.training:
                data_src = batched_data.get("data_src", None)
                atom_masks = batched_data.get("atom_masks", None)
                # collate_fn deliberately EXCLUDES 'data_src' from its torch.stack() (it
                # ships as a python list of per-sample (1,) tensors), so it arrives here as
                # a list, not a stacked tensor. Normalise both shapes rather than assuming.
                if isinstance(data_src, (list, tuple)):
                    try:
                        data_src = torch.cat(
                            [torch.as_tensor(t).reshape(-1) for t in data_src]).to(f_pred.device)
                    except (RuntimeError, ValueError, TypeError):
                        data_src = None
                elif data_src is not None and torch.is_tensor(data_src):
                    data_src = data_src.reshape(-1).to(f_pred.device)
                if (
                    data_src is not None
                    and atom_masks is not None
                    and f_pred.shape[0] > 0
                    # Under DDP the collate pads to N_max, so a sample-major reshape is only
                    # valid when the padded shape and the predicted shape agree. If they do
                    # not, we skip rather than broadcast a subset id onto the wrong atoms.
                    and tuple(atom_masks.shape[:2]) == tuple(f_pred.shape[:2])
                    and data_src.shape[0] == f_pred.shape[0]
                ):
                    src = data_src
                    atom_ok = atom_masks.bool().to(f_pred.device)
                    src2d = src.reshape(-1, 1).expand_as(atom_ok)
                    e_perat_pred = model_output["pred_energy_per_atom"].reshape(-1).to(f_pred.device)
                    e_perat_true = batched_data["energy_per_atom"].reshape(-1).to(f_pred.device)
                    e_shape_ok = e_perat_pred.shape[0] == src.shape[0] == e_perat_true.shape[0]
                    for sid in torch.unique(src2d[atom_ok]).tolist():
                        sel2d = atom_ok & (src2d == sid)
                        if not bool(sel2d.any()):
                            continue
                        dabs = torch.abs(f_pred[sel2d] - f_true[sel2d])
                        log_ouput[f"force_loss_src{int(sid)}_{data_name}"] = dabs.mean()
                        log_ouput[f"force_abssum_src{int(sid)}_{data_name}"] = dabs.sum()
                        # float, not int: the accumulator and _allreduceAVG both average.
                        log_ouput[f"force_ncomp_src{int(sid)}_{data_name}"] = torch.tensor(
                            float(dabs.numel()), device=f_pred.device)
                        # ---- PER-ATOM VECTOR-L2, the convention every published SPICE
                        # force number uses. SPICE's leaderboard (MACE-OFF23 Table 1,
                        # E2Former Table 3) reports the mean L2 norm of each atom's force-
                        # error VECTOR, not the mean absolute error per Cartesian component;
                        # the two differ by a factor near 2 (verified against the test
                        # file's own `MACE_forces` column: its structure-weighted All is
                        # 28.88 per-atom L2 against a published MACE-OFF-Small 29.76, ratio
                        # 1.031, whereas the per-component reading of the same column is
                        # 14.44 and matches no published row). Our paper's own convention is
                        # per-component throughout, so both are logged and neither column may
                        # be silently read as the other.
                        dl2 = torch.linalg.vector_norm(f_pred[sel2d] - f_true[sel2d], dim=-1)
                        log_ouput[f"force_l2_abssum_src{int(sid)}_{data_name}"] = dl2.sum()
                        log_ouput[f"force_l2_natom_src{int(sid)}_{data_name}"] = torch.tensor(
                            float(dl2.numel()), device=f_pred.device)
                        # Energy is per-sample, so select whole samples rather than atoms.
                        if e_shape_ok:
                            e_sel = src == sid
                            if bool(e_sel.any()):
                                ede = torch.abs(e_perat_pred[e_sel] - e_perat_true[e_sel])
                                log_ouput[f"energy_peratom_loss_src{int(sid)}_{data_name}"] = ede.mean()
                                log_ouput[f"energy_abssum_src{int(sid)}_{data_name}"] = ede.sum()
                                log_ouput[f"energy_nsample_src{int(sid)}_{data_name}"] = torch.tensor(
                                    float(ede.numel()), device=f_pred.device)
            f_pred = f_pred[masks]
            f_true = f_true[masks]
            # mask_weight = batched_data["atom_masks"].numel() / \
            #         torch.sum(batched_data["atom_masks"].float())

            f_mae = torch.mean(torch.abs(f_pred - f_true))
            if self.loss_fn == "mae":
                f_loss = torch.mean(torch.abs(f_pred - f_true))
            elif self.loss_fn == "mse":
                f_loss = torch.mean((f_pred - f_true) ** 2)
            elif self.loss_fn == "l2mae":
                f_loss = torch.linalg.vector_norm(f_pred - f_true, ord=2, dim=-1)
                f_loss = torch.sum(f_loss)/bs # follow fairchem
            elif self.loss_fn == "atoml2mae":
                f_loss = torch.linalg.vector_norm(f_pred - f_true, ord=2, dim=-1)
                f_loss = torch.mean(f_loss) # follow fairchem
            elif self.loss_fn == "huber":
                f_loss = torch.linalg.vector_norm(f_pred - f_true, ord=2, dim=-1)
                f_loss = torch.nn.functional.huber_loss(f_loss, torch.zeros_like(f_loss), delta=1.0, reduction="mean")

            loss += self.force_loss_weight * f_loss

            # Calculate force magnitudes
            # Compute magnitudes of each force vector and find the maximum for each sample
            f_loss_magnitudes = torch.norm(torch.abs(f_pred - f_true), dim=-1)  # Magnitude of each (x, y, z) force vector
            max_f_loss = torch.max(f_loss_magnitudes) 


            # cos{(0,0,0),(0,0,0)} = 0, thus div mask_weight
            forces_cosine_sim = torch.mean(
                torch.cosine_similarity(f_pred.detach(), f_true, dim=-1, eps=1e-8)
            )
            log_ouput["force_loss"] = f_mae #, size)
            log_ouput["max_f_loss"] = max_f_loss

            log_ouput[f"force_loss_{data_name}"] = f_mae
            # log_ouput[f"max_f_loss{data_name}"] = (
            #     max_f_loss,
            #     size,
            # )  # Log the mean of the maximum forces
            log_ouput[f"f_cos_{data_name}"] = forces_cosine_sim  # Log the mean of the maximum forces

        e_mae = torch.mean(torch.abs(e_pred - e_true))
        e_peratom_mae = torch.mean(torch.abs(e_pred_peratom.detach() - e_true_peratom))
        # Update logging output
        log_ouput.update(
            {
                "loss": loss.detach(),
                f"loss_{data_name}": loss.detach(),
                "energy_loss": e_mae,
                "energy_peratom_loss": e_peratom_mae,
                f"e_peratom_loss_{data_name}":e_peratom_mae,
            }
        )
        
        return loss, log_ouput
