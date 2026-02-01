# -*- coding: utf-8 -*-
from pathlib import Path
from socket import IP_RECVRETOPTS
from typing import Dict, Union

import e3nn
import torch
import torch.nn as nn
from loguru import logger

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
        self.sing_linear = nn.Linear(embedding_dim,1,bias = False)

    def forward(self, x, v, batched_data,pred_energy,**kwargs):
        v = self.sing_linear(v)
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

    def forward(self, x, v, batched_data,pred_energy,**kwargs):
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

        pred_forces = -torch.autograd.grad(
            outputs=pred_energy,
            inputs=batched_data["pos"],
            grad_outputs=grad_outputs,
            create_graph=self.training,
            retain_graph=self.training,
            only_inputs=True,
        )[0]

        return pred_forces

class MLP_energy_head(nn.Module):
    def __init__(self,embedding_dim,lmax=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
                            nn.Linear(embedding_dim, embedding_dim, bias=True),
                            nn.SiLU(),
                            nn.Linear(embedding_dim, embedding_dim, bias=True),
                            nn.SiLU(),
                            nn.Linear(embedding_dim, 1, bias=True),
                        )
    def forward(self,embedding,irreps):
        return self.mlp(embedding)



import math

def CalcSpherePoints(num_points: int, device: str = "cpu"):
    goldenRatio = (1 + 5**0.5) / 2
    i = torch.arange(num_points, device=device).view(-1, 1)
    theta = 2 * math.pi * i / goldenRatio
    phi = torch.arccos(1 - 2 * (i + 0.5) / num_points)
    points = torch.cat(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    )

    # weight the points by their density
    pt_cross = points.view(1, -1, 3) - points.view(-1, 1, 3)
    pt_cross = torch.sum(pt_cross**2, dim=2)
    pt_cross = torch.exp(-pt_cross / (0.5 * 0.3))
    scalar = 1.0 / torch.sum(pt_cross, dim=1)
    scalar = num_points * scalar / torch.sum(scalar)
    return points * (scalar.view(-1, 1))
from e3nn import o3
class eSCN_EnergyBlock(nn.Module):
    """
    Energy Block: Output block computing the energy

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self,embedding_dim,lmax = 2 ):
        super().__init__()
        self.lmax = lmax
        self.num_channels = embedding_dim
        self.num_sphere_samples = 18  # num_sphere_samples
        
        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        self.sphharm_weights = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )
        self.act = nn.SiLU()

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

    def compiled_forward(self, x):
        out_shape = x.shape[:-2]
        x = x.reshape((-1,)+x.shape[-2:])

        x_pt = torch.einsum(
            "abc, pb->apc",
            x,
            self.sphharm_weights,
        ).contiguous()
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        return torch.sum(x_pt, dim=1).reshape(out_shape+(1,)) / self.num_sphere_samples

    def forward(self, embedding,irreps):
        return self.compiled_forward(
            irreps,
        )


class eSCN_ForceBlock(nn.Module):
    """
    Force Block: Output block computing the per atom forces

    Args:
        num_channels (int):         Number of channels
        num_sphere_samples (int):   Number of samples used to approximate the integral on the sphere
        act (function):             Non-linear activation function
    """

    def __init__(self,num_channels,lmax = 2):
        super().__init__()

        self.num_channels = num_channels
        self.num_sphere_samples = 18  # num_sphere_samples

        self.fc1 = nn.Linear(self.num_channels, self.num_channels)
        self.fc2 = nn.Linear(self.num_channels, self.num_channels)
        self.fc3 = nn.Linear(self.num_channels, 1, bias=False)

        self.sphere_points = nn.Parameter(
            CalcSpherePoints(self.num_sphere_samples), requires_grad=False
        )

        self.sphharm_weights = nn.Parameter(
            o3.spherical_harmonics(
                torch.arange(0, lmax + 1).tolist(),
                self.sphere_points,
                False,
            ),
            requires_grad=False,
        )
        self.act = nn.SiLU()

    def compiled_forward(self, x):
        out_shape = x.shape[:-2]
        x = x.reshape((-1,)+x.shape[-2:])
        x_pt = torch.einsum(
            "abc, pb->apc",
            x,
            self.sphharm_weights,
        ).contiguous()
        # x_pt are the values of the channels sampled at different points on the sphere
        x_pt = self.act(self.fc1(x_pt))
        x_pt = self.act(self.fc2(x_pt))
        x_pt = self.fc3(x_pt)
        x_pt = x_pt.view(-1, self.num_sphere_samples, 1)
        forces = x_pt * self.sphere_points.view(1, self.num_sphere_samples, 3)
        return torch.sum(forces, dim=1).reshape(out_shape+(3,))  / self.num_sphere_samples

    def forward(self,decoder_x_output, decoder_vec_output,batched_data,pred_energy,irreps):
        return self.compiled_forward(irreps)  # *0.001



class MDEnergyForceMultiHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_fn = args.loss_fn.lower()
        self.energy_loss_weight = args.energy_loss_weight
        self.force_loss_weight = args.force_loss_weight
        self.auto_grad = args.AutoGradForce
        if args.backbone_config["name"] == "e2former":
            embedding_dim = e3nn.o3.Irreps(
                args.backbone_config["irreps_node_embedding"]
            )[0][0]
            lmax = len(e3nn.o3.Irreps(args.backbone_config["irreps_node_embedding"]))-1
        else:
            embedding_dim = 256

        # Initialize multiple energy and force heads based on dataset_name_list
        self.energy_heads = nn.ModuleDict()
        self.force_heads = nn.ModuleDict()

        head_cfg_str = getattr(args,"head_cfg_str","") #"reweight;share"



        self.threshold = float(head_cfg_str.split("+")[-1]) if "." in head_cfg_str.split("+")[-1] else 0.7

        for dataset_name in args.dataset_name_list.split(","):

            if dataset_name not in self.energy_heads:
                self.energy_heads[dataset_name] = MLP_energy_head(embedding_dim,lmax)
                self.force_heads[dataset_name] = PureLinearForce(embedding_dim,lmax)

    
                # If dataset_name is not "deshaw_400" or "deshaw_650", create force head
                if self.auto_grad:
                    self.force_heads[dataset_name] = GradientHead()

    def update_batched_data(self, samples, batched_data):
        return batched_data

    def forward(self, result_dict: Dict[str, torch.Tensor],
                        batched_data: Dict[str, torch.Tensor]):
        if "pred_energy" in result_dict:return result_dict

        decoder_x_output = result_dict["node_featuresBxN"]
        decoder_vec_output = result_dict["node_vec_featuresBxN"]
        decoder_irreps_output = result_dict["node_irrepsBxN"]
        
        dataset_name = batched_data["data_name"][0]
        # Use the correct energy head based on dataset_name
        head_name = (
            dataset_name
            if dataset_name not in self.dataset_rename
            else self.dataset_rename[dataset_name]
        )
        bs = decoder_x_output.shape[0]

        atom_energy = self.energy_heads[head_name](decoder_x_output,decoder_irreps_output).squeeze(-1)
        
        atom_energy = atom_energy.masked_fill(~batched_data["atom_masks"], 0.0)
        pred_energy = atom_energy.sum(dim=-1)
        pred_energy_per_atom = pred_energy / torch.sum(batched_data["atom_masks"],dim = -1).reshape(-1)
        

        # If dataset_name is not "deshaw_400" or "deshaw_650", process force head

        pred_forces = self.force_heads[head_name](
            decoder_x_output, decoder_vec_output,batched_data,pred_energy,irreps = decoder_irreps_output
        )

        result_dict.update({
                "pred_energy":pred_energy,
                "pred_forces":pred_forces,
                "pred_energy_per_atom":pred_energy_per_atom})
        return result_dict
        #         "pred_energy":pred_energy,
        #         "pred_forces":pred_forces,

    def update_loss(self, loss, log_ouput, model_output, batched_data):
        data_name = batched_data["data_name"][0]
        e_pred = model_output["pred_energy"].reshape(-1)
        f_pred = model_output.get("pred_forces", None)
        e_pred_peratom = model_output["pred_energy_per_atom"].reshape(-1)

        e_true = batched_data["energy"].reshape(-1)
        f_true = batched_data.get("forces", None)
        e_true_peratom = batched_data["energy_per_atom"].reshape(-1)

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
            e_loss = torch.mean((e_pred_peratom - e_true_peratom) ** 2)
        elif "atom" in self.loss_fn:
            e_loss = torch.mean(torch.abs(e_pred_peratom - e_true_peratom))
        else:
            e_loss = torch.mean(torch.abs(e_pred - e_true))


        loss = self.energy_loss_weight * e_loss #e_loss
        log_ouput = {}
        # Batch size
        bs = e_true.shape[0]
        if f_pred is not None and f_true is not None:
            N_max = f_true.shape[1]
            masks = batched_data["atom_masks"]&(~(batched_data["fixed"].bool()))
            f_pred = f_pred[masks]
            f_true = f_true[masks]
                
            f_mae = torch.mean(torch.abs(f_pred - f_true))
            if data_name == "spice":
                log_ouput.update(
                                            loss_by_Z(torch.mean(torch.abs(f_pred - f_true),dim = -1).view(-1),
                                            batched_data["data_src"].reshape(-1,1).repeat(1,N_max)[masks],
                                            prefill = "force")

                )
                log_ouput.update(
                                            loss_by_Z(torch.abs(e_pred_peratom - e_true_peratom).view(-1),
                                            batched_data["data_src"].view(-1),
                                            prefill = "energy_peratom")
                )

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
            f_loss_magnitudes = torch.norm(torch.abs(f_pred - f_true), dim=-1)  # Magnitude of each (x, y, z) force vector
            max_f_loss = torch.max(f_loss_magnitudes) 

            forces_cosine_sim = torch.mean(
                torch.cosine_similarity(f_pred.detach(), f_true, dim=-1, eps=1e-8)
            )
            log_ouput["force_loss"] = f_mae #, size)
            log_ouput["max_f_loss"] = max_f_loss

            log_ouput[f"force_loss_{data_name}"] = f_mae

            log_ouput[f"f_cos_{data_name}"] = forces_cosine_sim  # Log the mean of the maximum forces

        e_mae = torch.mean(torch.abs(e_pred - e_true))
        e_peratom_mae = torch.mean(torch.abs(e_pred_peratom.detach() - e_true_peratom))
        # Update logging output
        log_ouput.update(
            {
                "loss": loss,
                "energy_loss": e_mae,
                "energy_peratom_loss": e_peratom_mae,
                f"e_peratom_loss_{data_name}":e_peratom_mae,
            }
        )
        
        return loss, log_ouput

    
def loss_by_Z(any_loss: torch.Tensor,
                    atomic_numbers: torch.Tensor,
                    prefill = "force"):

    
    force_flat = any_loss.reshape(-1).float()
    Z_flat = atomic_numbers.reshape(-1).long()

    
    unique_Z, inv = torch.unique(Z_flat, return_inverse=True)  

    
    sum_loss_per_Z = torch.bincount(inv, weights=force_flat, minlength=unique_Z.numel())
    count_per_Z    = torch.bincount(inv, minlength=unique_Z.numel()).clamp_min(1)

    mean_loss_per_Z = sum_loss_per_Z / count_per_Z
    logout = {}
    for i in range(len(unique_Z)):
        logout[f"{prefill}_{unique_Z[i].item()}"] = mean_loss_per_Z[i].item()
    return logout
