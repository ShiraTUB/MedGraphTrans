import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, Optional, Union
from torch_geometric.nn.conv import HGTConv, HEATConv, GCNConv
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter

from src.medical_hgt.ml_utils import compute_weight
from src.medical_hgt.ml_utils import construct_bipartite_edge_index


class WeightedHGTConv(HGTConv):
    def __init__(
            self,
            in_channels: Union[int, Dict[str, int]],
            out_channels: int,
            metadata: Metadata,
            heads: int = 1,
            **kwargs,
    ):
        super(WeightedHGTConv, self).__init__(in_channels, out_channels, metadata, heads, **kwargs)
        # self.lin_edge_list = torch.nn.ModuleList()
        #
        # for edge_type in self.edge_types_map.values():
        #     edge_lin = Linear(1, self.heads * self.out_channels, bias=False, weight_initializer='glorot')
        #     self.lin_edge_list.append(edge_lin)
        #
        # self.e = Parameter(
        #     torch.empty(self.heads * self.out_channels,
        #                 self.heads * out_channels // heads))
        self._alpha = None

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)
        # glorot(self.lin_edge)
        # glorot(self.e)

    def forward(
            self,
            x_dict: Dict[NodeType, Tensor],
            edge_index_dict: Dict[EdgeType, Adj],
            edge_weights_dict: Dict[EdgeType, Tensor] = None  # new edge_weights dict
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        # for edge_type, edge_weights_indices in edge_weights_dict.items():
        #     relevant_weights = self.edge_weights_dict[edge_type][0][edge_weights_indices]
        #     self.relevant_weights_dict[edge_type] = torch.nn.Parameter(torch.stack((relevant_weights, relevant_weights), dim=1), requires_grad=True)

        edge_index, edge_attr, edge_weight, edge_offset_dict = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict=self.p_rel,
            edge_weights_dict=edge_weights_dict,
        )

        out = self.propagate(edge_index, k=k, q=q, v=v,
                             edge_attr=edge_attr,
                             edge_weight=edge_weight,
                             edge_offset_dict=edge_offset_dict,
                             size=None)

        message_alpha = self._alpha
        self._alpha = None

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]
                for node_index, node_embeddings in enumerate(out_dict[node_type]):
                    weight = compute_weight(node_type, node_index, edge_index_dict, edge_weights_dict)
                    out_dict[node_type][node_index] *= weight

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
                torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int],
                edge_weight: Optional[Tensor] = None,  # New argument for edge weights
                edge_offset_dict: Dict[EdgeType, int] = None,
                ) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr

        alpha_edge = 0
        # if edge_weight is not None:
        #     if edge_weight.dim() == 1:
        #         edge_weight = edge_weight.view(-1, 1)
        #
        #     edge_weights = self.lin_edge(edge_weight).view(
        #         -1, self.heads * self.out_channels)
        #     # if edge_weights.size(0) != edge_weight.size(0):
        #     #     edge_weights = torch.index_select(edge_weights, 0,
        #     #                                          edge_type)
        #     alpha_edge = torch.matmul(edge_weights, self.e)
        #
        #     alpha = alpha * alpha_edge  # Apply edge weights

        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        out = v_j * alpha.view(-1, self.heads, 1)

        return out.view(-1, self.out_channels)


def __repr__(self) -> str:
    return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
            f'heads={self.heads})')
