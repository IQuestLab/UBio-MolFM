import torch
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus
import torch.nn as nn

from torch_geometric.utils import softmax
import copy
import math
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_cluster import radius_graph


def _norm(vec, dim = 1, keepdim=False):
    return torch.square(vec + 1e-6).sum(dim=dim, keepdim=keepdim).sqrt() + 1e-6

def vec_layernorm(vec, norm):
    dist = torch.norm(vec, dim=1, keepdim=True)
    if (dist == 0).all():
        return torch.zeros_like(vec)
    tmp_dist = torch.where(dist == 0, torch.ones_like(dist), dist)
    direct = vec / tmp_dist 
    return F.relu(norm(dist)) * direct
    
def max_min_norm(dist):
    max_val, _ = torch.max(dist, dim=-1)
    min_val, _ = torch.min(dist, dim=-1)
    delta = (max_val - min_val).view(-1)
    delta = torch.where(delta == 0, torch.ones_like(delta), delta)
    dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
    return dist

class NormBasedOperator(nn.Module):
    def __init__(self, num_features):
        super(NormBasedOperator, self).__init__()
        self.norm_operator = None
    
    def forward(self, x):
        '''
        x: (num_nodes, 3, num_features)
        '''
        norm = _norm(x, dim=1, keepdim = True)  # (num_nodes, 1, n_features)
        norm_bn = self.norm_operator(norm) # (num_nodes, 1, n_features)
        x = x / norm * norm_bn
        return x
    

class VNLayerNorm(NormBasedOperator):
    '''
    Vec Batch Norm Layer 
    @parms num_features: number of features
    '''
    def __init__(self, num_features):
        super(VNLayerNorm, self).__init__(num_features)
        self.norm_operator = nn.LayerNorm(num_features)
    
class NormSiLu(NormBasedOperator):
    '''
    Vec Batch Norm Layer 
    @parms num_features: number of features
    '''
    def __init__(self, num_features):
        super(NormSiLu, self).__init__(num_features)
        self.norm_operator = nn.Sequential(nn.Linear(num_features, num_features),
            nn.SiLU(), nn.LayerNorm(num_features))

        
    
class VecLinear2(nn.Module):
    def __init__(self, hidden_channels):
        super(VecLinear2, self).__init__()
        self.hidden_channels = hidden_channels
        self.linear_1 = nn.Linear(hidden_channels, 2 * hidden_channels, bias = False)
        self.linear_2 = nn.Linear(2 * hidden_channels, hidden_channels, bias = False)
        
    def forward(self, v):
        v1, v2 = torch.split(self.linear_1(v), self.hidden_channels, dim = 2)
        v3 = v1.cross(v2, dim = 1)
        v_out = self.linear_2(torch.cat([v, v3], dim = 2))
        return v_out, (v1 * v2).sum(dim = 1)
        
class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower, cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W
    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)
    
    def aggregate(self, features, index):
        # no aggregate
        return features

class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        start_value = torch.exp(
            torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower)
        )
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas
            * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (
                torch.cos(
                    math.pi
                    * (
                        2
                        * (distances - self.cutoff_lower)
                        / (self.cutoff_upper - self.cutoff_lower)
                        + 1.0
                    )
                )
                + 1.0
            )
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(
        self,
        cutoff_lower,
        cutoff_upper,
        max_num_neighbors=32,
        return_vecs=False,
        loop=False,
    ):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        self.return_vecs = return_vecs
        self.loop = loop

    def forward(self, pos, batch, edge_index = None):
        if edge_index is None:
            edge_index = radius_graph(
                pos,
                r=self.cutoff_upper,
                batch=batch,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors,
            )
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:, lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vecs:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        return edge_index, edge_weight, None

class Sphere(nn.Module):
    
    def __init__(self, l=2):
        
        super(Sphere, self).__init__()
        
        self.l = l
        
    def forward(self, edge_vec):
        
        edge_sh = _spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        
        return edge_sh
        
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    if lmax == 1:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2
        ], dim=-1)

    sh_2_0 = math.sqrt(3.0) * x * z
    sh_2_1 = math.sqrt(3.0) * x * y
    y2 = y.pow(2)
    x2z2 = x.pow(2) + z.pow(2)
    sh_2_2 = y2 - 0.5 * x2z2
    sh_2_3 = math.sqrt(3.0) * y * z
    sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

    if lmax == 2:
        return torch.stack([
            sh_1_0, sh_1_1, sh_1_2,
            sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
        ], dim=-1)
      
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
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v
    
rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}

class LongShortIneractModel_distance(MessagePassing):
    r'''
    This is the long term model to capture the relationship b/w center node and long term groups
    First, perform a pointTransformer within each group to obtain the representation of each group.
    Second, MP is performed on a bipartite graph of size #nodes * #groups.
    Below is an disection of the architecture of the model

    in_channels_node ---- (PointTransformer) ----- > in channels group

    in_channels_group --- linear --- num_filters -|    |-- > out_channels
                                                  | MP |
    in_channels_node  --- linear --- num_filters -| -> |-- > out_channels
                                                  |    
    num_gaussians     --- linear --- num_filters -|    

    @param in_channels_node: The size of node features
    @param in_chnnels_right: The size of group features
    @param out_channels: The size of output node features and group features (unified output size)
    @param num_filters: number of filters 
    @param num_gaussians: number of gaussians used to expand edge weight
    @param cutoff: long term cutoff distance
    ''' 
    def __init__(self, hidden_channels,num_gaussians, cutoff ,max_group_num = 3,act = "silu",**kwargs):
        super().__init__()

        self.act1 = nn.SiLU()
        self.cutoff = cutoff
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
        self.max_group_num  = max_group_num
        
        self.act = nn.SiLU()
        self.mlp_1 = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels, hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        pass
        # Get the center position
        
        
        #     raise NotImplementedError

    def forward(self, edge_index, node_embedding, node_pos,group_embedding,group_pos,**kwargs):
        '''
        grouping_graph_edge_idx = data.grouping_graph # Grouping graph (intra group complete graph; inter group disconnected)
        edge_idx = data.interaction_graph # Bipartite graph
        
        '''

        nodes = edge_index[0]
        groups = edge_index[1]

        # Distance Expansion
        edge_weight = _norm((node_pos[nodes] - group_pos[groups]), dim = -1)
        edge_attr = self.distance_expansion(edge_weight)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        C = 0.5 * (torch.cos(edge_weight * torch.pi / self.cutoff) + 1.0) #cutoff function
        W = self.mlp_1(edge_attr) * C.view(-1, 1)

        node_embedding = self.lin1(node_embedding)
        group_embedding = self.lin2(group_embedding)
        node_embedding = self.propagate(edge_index.flip(0), size = (num_groups, num_nodes), x=(group_embedding, node_embedding), W=W)/self.max_group_num
        node_embedding = self.lin3(node_embedding)
        
        return node_embedding,None
        
    def message(self, x_j, W):
        return x_j * W
    
    
class Bipartite_Edge_Feat_Init(nn.Module):
    def __init__(self,
                rbf_type="expnorm",
                num_rbf = 50,
                trainable_rbf = True,
                hidden_channels = 128,
                cutoff_lower = 0,
                cutoff_upper = 10):
    
        super().__init__()
        if rbf_type == "expnorm":
            rbf = ExpNormalSmearing
        elif rbf_type == "":
            rbf = GaussianSmearing
        else:
            assert(False)
        self.distance_encoder=rbf(cutoff_lower=cutoff_lower, cutoff_upper=cutoff_upper, num_rbf=num_rbf, trainable=trainable_rbf)
        self.rbf_linear = nn.Linear(num_rbf,hidden_channels)

    def forward(self, edge_index, node_pos, group_pos, *args, **kwargs):
        edge_vec = node_pos[edge_index[0]] - group_pos[edge_index[1]]
        edge_weight = _norm(edge_vec, dim=1)
        edge_vec = edge_vec / edge_weight.unsqueeze(1)
        edge_attr = self.distance_encoder(edge_weight)
        edge_attr = self.rbf_linear(edge_attr)
        return edge_index, edge_weight, edge_attr, edge_vec   
    
class LongShortIneractModel_dis_direct(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        self.model_1 = None
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            group_embedding = self.layernorm_group(group_embedding)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim = -1) / math.sqrt(self.attn_channels)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        if mode == 'node_to_group':
            model = self.model_1
        else:
            model = self.model_2
        

        
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec    


class LongShortIneractModel_dis_direct_vector(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node




class LongShortIneractModel_dis_direct_vector2(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    


class LongShortIneractModel_dis_direct_vector2_drop(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8, p =0.1,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        self.dropout_s = nn.Dropout(p)
        self.dropout_v = nn.Dropout(p)
        self.p = p
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
            group_embedding = self.layernorm_group(group_embedding)
            group_vec = vec_layernorm(group_vec, max_min_norm)
        if self.p>0:
            group_embedding = self.dropout_s(group_embedding)
            group_vec = self.dropout_v(group_vec)
        
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    
        
class LongShortIneractModel_dis_direct_vector3(LongShortIneractModel_dis_direct):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(hidden_channels, num_gaussians, cutoff, norm,act,num_heads) # currently only node embedding is computed and updated, only group message flow to node
        
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)

        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]

        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x=(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')    
        

        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    






class LongShortIneractModel_dis_direct_two_way(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias=False) for _ in range(6)])    
        })
        self.model_1 = nn.ModuleDict({
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar': nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
        })
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        m_s_group, m_v_group = self.propagate(edge_index,
                                size = (num_nodes, num_groups,),
                                x =(node_embedding, group_embedding, ), 
                                v = node_vec[edge_index[0]],
                                u_ij = edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = None, 
                                val = None,
                                mode = 'node_to_group')
        
        group_embedding = group_embedding + m_s_group
        group_vec = group_vec + m_v_group 
        
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu")
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][2](node_vec)
        v_node_2 = self.model_2['linears'][3](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears'][4](m_s_node) + self.model_2['linears'][5](m_s_node)
        dv_node = m_v_node + self.model_2['linears'][0](m_s_node).unsqueeze(1) * self.model_2['linears'][1](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        attn = attn.sum(dim = -1) / math.sqrt(self.attn_channels)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        
        
        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec
    

class LongShortIneractModel_dis_direct_new(MessagePassing):
    def __init__(self, hidden_channels, num_gaussians, cutoff, norm = False,
                 act = "silu",num_heads=8,**kwargs):
        super().__init__(aggr='add', node_dim = 0) # currently only node embedding is computed and updated, only group message flow to node
        self.act = act_class_mapping[act]()
        self.norm = norm
        self.layernorm_node = nn.LayerNorm(hidden_channels)
        self.layernorm_group = nn.LayerNorm(hidden_channels)
        self.layernorm_node_vec = nn.LayerNorm(hidden_channels)
        self.layernorm_group_vec = nn.LayerNorm(hidden_channels)
        self.model_2 = nn.ModuleDict({
            'q': nn.Linear(hidden_channels, hidden_channels),
            'k': nn.Linear(hidden_channels, hidden_channels),
            'val': nn.Linear(hidden_channels, hidden_channels),
            'mlp_scalar': nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels)),
            'mlp_scalar_pos': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'mlp_scalar_vec': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                self.act,
                nn.Linear(hidden_channels, hidden_channels),),
            'linears': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, bias = False) for _ in range(3)]),
            'linears_scalar': nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(3)]),
            'vec_linear': VecLinear2(hidden_channels)    
        })
        self.cutoff_func = CosineCutoff(cutoff)
        self.model_1 = None
        self.num_heads = num_heads
        self.attn_channels = hidden_channels // num_heads
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Create Xavier Unifrom for all linear modules
        '''
        self.layernorm_node.reset_parameters()
        self.layernorm_group.reset_parameters()
        for model in [self.model_1, self.model_2]:
            if model is None:continue
            for _, value in model.items():
                if isinstance(value, nn.ModuleList):
                    for m in value.modules():
                        if isinstance(m, nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(value, nn.Linear):
                    torch.nn.init.xavier_uniform_(value.weight)
                    value.bias.data.fill_(0)
                else:
                    pass
            
    
    def forward(self, edge_index, node_embedding, 
                node_pos, node_vec, group_embedding, 
                group_pos, group_vec, edge_attr, edge_weight, edge_vec):
        """_summary_

        Args:
            edge_index (_type_): 2*edge_num, [0] is node id,[1] is group id
            x_node (_type_): _description_
            x_group (_type_): _description_
            v_node (_type_): _description_
            v_group (_type_): _description_
            r_node (_type_): _description_
            r_group (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.norm:
            node_embedding = self.layernorm_node(node_embedding)
            node_vec = vec_layernorm(node_vec, max_min_norm)
        num_nodes = node_embedding.shape[0]
        num_groups = group_embedding.shape[0]
        
        attn_2, val_2 = self.calculate_attention(node_embedding, group_embedding, edge_index[0], edge_index[1], edge_attr, self.model_2, "silu", edge_weight)
        m_s_node, m_v_node = self.propagate(edge_index.flip(0),
                                size = (num_groups, num_nodes),
                                x =(group_embedding, node_embedding), 
                                v = group_vec[edge_index[1]],
                                u_ij = -edge_vec, 
                                d_ij = edge_weight, 
                                attn_score = attn_2, 
                                val = val_2[edge_index[1]],
                                mode = 'group_to_node')       
        
        v_node_1 = self.model_2['linears'][0](node_vec)
        v_node_2 = self.model_2['linears'][1](node_vec)
        dx_node =  (v_node_1 * v_node_2).sum(dim = 1) * self.model_2['linears_scalar'][0](m_s_node) + self.model_2['linears_scalar'][1](m_s_node)
        dv_node = m_v_node + self.model_2['linears_scalar'][2](m_s_node).unsqueeze(1) * self.model_2['linears'][2](node_vec)
        return dx_node, dv_node
    
    def calculate_attention(self,x_1, x_2, x1_index, x2_index, expanded_edge_weight, model, attn_type, edge_weight = None):
        r'''
        Calculate attention value for each edge.
        x_1: embedding for query. target node embedding.
        x_2: embedding for key value. source node embedding.
        edge_index: graph to calculate attention
        expanded_edge_weight: the expanded edge weight
        model: the model for calculating attention, be a dictionary with keys: q, k, val, mlp_edge_attr, mlp_scalar_pos, mlp_scalar_vec
        '''
        __supported_attn__ = ['softmax', 'silu']
        q = model['q'](x_1).reshape(-1, self.num_heads, self.attn_channels) # num_groups x num_heads x attn_channels
        k = model['k'](x_2).reshape(-1, self.num_heads, self.attn_channels) # num_nodes x num_heads x attn_channels
        val = model['val'](x_2).reshape(-1, self.num_heads, self.attn_channels) 

        q_i = q[x1_index]
        k_j = k[x2_index]

        expanded_edge_weight = expanded_edge_weight.reshape(-1, self.num_heads, self.attn_channels)
        attn = q_i * k_j * expanded_edge_weight
        if edge_weight is not None:
            attn = attn.sum(dim = -1) 
        else:
            attn = attn.sum(dim = -1)

            
        if attn_type == 'softmax':
            attn = softmax(attn, x1_index, dim = 0) * self.cutoff_func(edge_weight).unsqueeze(1)
        elif attn_type == 'silu':
            attn = act_class_mapping['silu']()(attn) * self.cutoff_func(edge_weight).unsqueeze(1)
        else:
            raise NotImplementedError(f'Attention type {attn_type} is not supported, supported types are {__supported_attn__}')
        return attn, val      

    def message(self, x_i, x_j, v, u_ij, d_ij, attn_score, val, mode):
        '''
        Calculate the message from node j to node i
        Return: Scalar message from node j to node i, Vector message from node j to node i
        @param x_j: Node feature of node j
        @param x_i: Node feature of node i
        @param smeared_distance: Smearing distance between node i and node j
        @param v: Embedding of the group th
        @param mode: 'node_to_group' or 'group_to_node'
        '''

        
        if mode == 'node_to_group':
            model = self.model_1
            m_s_ij = model['mlp_scalar'](torch.cat([x_i, x_j], dim = -1))
            m_v_ij = model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * v + \
            model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1)
            return m_s_ij, m_v_ij
        else:
            model = self.model_2
        

        

        m_s_ij = val * attn_score.unsqueeze(-1) # scalar message
        m_s_ij = m_s_ij.reshape(-1, self.num_heads * self.attn_channels)
        v, s = model['vec_linear'](v)        
        m_s_ij = model['mlp_scalar'](torch.cat([s, m_s_ij], dim = -1))
        m_v_ij = model['mlp_scalar_pos'](m_s_ij).unsqueeze(1) * u_ij.unsqueeze(-1) \
        + model['mlp_scalar_vec'](m_s_ij).unsqueeze(1) * (v) # vector message
        return  m_s_ij, m_v_ij
        
    def aggregate(self, features, index, ptr, dim_size):
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec



class Long_Range(torch.nn.Module):
    def  __init__(self,hidden,cutoff_upper,layers):
        super().__init__() # currently only node embedding is computed and updated, only group message flow to node

        self.hidden = hidden
        self.layers = layers
        self.bipartite_edge_fea_init = Bipartite_Edge_Feat_Init(rbf_type = "expnorm",
                        num_rbf = 50,
                        trainable_rbf = True,
                        hidden_channels = hidden,
                        cutoff_lower = 0,
                        cutoff_upper = cutoff_upper)


        self.long_range_model = nn.ModuleList([
            LongShortIneractModel_dis_direct_vector2_drop(hidden, num_gaussians=50, 
                                                                cutoff=cutoff_upper,norm=True, max_group_num = 8,act = "silu",num_heads=8,
                                                                p = 0.1) for _ in range(layers)])


    def forward(self,node_pos,
                node_irreps_input,
                edge_dis,
                edge_vec,
                attn_weight,  # e.g. rbf(|r_ij|) or relative pos in sequence
                atomic_numbers,
                poly_dist=None,
                attn_mask=None,  # non-adj is True
                batch=None,
                batched_data=None,
                cluster_pos=None,
                cluster_irreps_input=None,):

        topK = attn_weight.shape[1]
        f_N1 = node_pos.shape[0]

        pos = node_pos.reshape((-1,3))
        node_embedding_long = node_irreps_input[:,0]
        node_vec_long = node_irreps_input[:,1:4]
        group_pos = cluster_pos.reshape((-1,3))
        group_embedding = cluster_irreps_input[:,0]
        group_vec = cluster_irreps_input[:,1:4]

        interaction_graph=torch.stack([
                        torch.arange(f_N1).reshape(f_N1, -1).repeat(1, topK).reshape(-1).cuda(),
                        batched_data["f_sparse_idx_expnode"].reshape(-1),
                    ])[:,~attn_mask.view(-1)]
        edge_index_bipartite, edge_weight_bipartite, edge_attr_bipartite, edge_vec_bipartite = \
            self.bipartite_edge_fea_init(interaction_graph, pos, group_pos)

        for i in range(self.layers):
            delta_node_embedding_long, delta_node_vec_long = self.long_range_model[i](edge_index = edge_index_bipartite, 
                                                                                     node_embedding = node_embedding_long, node_pos = pos,node_vec = node_vec_long,
                                                                                     group_embedding=group_embedding, group_pos = group_pos,
                                                                                     group_vec = group_vec, edge_attr = edge_attr_bipartite,
                                                                                     edge_weight = edge_weight_bipartite, edge_vec = edge_vec_bipartite)

            node_embedding_long = node_embedding_long + delta_node_embedding_long
            node_vec_long = node_vec_long + delta_node_vec_long

        node_irreps_output =  torch.cat([
                            node_embedding_long.reshape((f_N1,1,-1)),
                            node_vec_long.reshape((f_N1,3,-1)),
                            node_irreps_input[:,4:]],dim = 1)

        return node_irreps_output,None