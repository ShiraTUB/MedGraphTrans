import torch
import numpy as np
import torch_geometric.transforms as T
from torch.nn.functional import pad
from torch_geometric.data import HeteroData


class Graph(object):
    def __init__(self):
        self.subgraphs = []
        self.max_person_padding = []
        self.max_message_padding = []
        self.max_know_padding = []

    def insert_subgraph(self, subgraph):
        # track max padding person nodes
        if len(self.max_person_padding) > 0:

            max_person_padding = self.max_person_padding[-1]
            max = np.max((max_person_padding, subgraph.person_padding))
            self.max_person_padding.append(max)

        else:
            self.max_person_padding.append(subgraph.person_padding)
        # track max padding message nodes
        if len(self.max_message_padding) > 0:
            max_message_padding = self.max_message_padding[-1]
            max = np.max((max_message_padding, subgraph.message_padding))
            self.max_message_padding.append(max)

        else:
            self.max_message_padding.append(subgraph.message_padding)
        # track max padding knowledge nodes
        if len(self.max_know_padding) > 0:
            max_know_padding = self.max_know_padding[-1]
            max = np.max((max_know_padding, subgraph.know_padding))
            self.max_know_padding.append(max)

        else:
            self.max_know_padding.append(subgraph.know_padding)
        # insert subgraph to subgraph's list
        self.subgraphs.append(subgraph)

    def get_tgt(self, idx):
        return self.subgraphs[idx].message_nodes

    def pad_nodes(self, nodes, pad_size):
        if type(nodes) != type(None):
            p = (0, pad_size)
            nodes = pad(nodes, p)
            return nodes
        else:
            return torch.zeros(size=(1, pad_size))

    def get_src_subgraph(self, idx):

        # extract max padding for the nodes
        max_person_padding = self.max_person_padding[idx]
        max_message_padding = self.max_message_padding[idx]
        max_know_padding = self.max_know_padding[idx]

        # concatenate and pad all subgraphs
        for i, sub in enumerate(self.subgraphs[:idx]):

            if sub.person_padding < max_person_padding:
                pad_size = max_person_padding - sub.person_padding
                sub.person_nodes = self.pad_nodes(sub.person_nodes, pad_size)

            if sub.message_padding < max_message_padding:
                pad_size = max_message_padding - sub.message_padding
                sub.message_nodes = self.pad_nodes(sub.message_nodes, pad_size)

            if sub.know_padding < max_know_padding:
                pad_size = max_know_padding - sub.know_padding
                sub.know_nodes = self.pad_nodes(sub.know_nodes, pad_size)

            if i == 0:
                person_nodes = sub.person_nodes
                message_nodes = sub.message_nodes
                know_nodes = sub.know_nodes
                person_msg = sub.person_msg
                msg_know = sub.msg_know
                know_know = sub.know_know
            else:
                person_nodes = torch.cat((person_nodes, sub.person_nodes), 0)
                message_nodes = torch.cat((message_nodes, sub.message_nodes), 0)
                know_nodes = torch.cat((know_nodes, sub.know_nodes), 0)

                person_msg = torch.cat((person_msg, sub.person_msg), 1)
                msg_know = torch.cat((msg_know, sub.msg_know), 1)
                if type(sub.know_know) != type(None) and type(know_know) != type(None):
                    know_know = torch.cat((know_know, sub.know_know), 1)
                elif type(sub.know_know) != type(None):
                    know_know = sub.know_know

        # generate the Hetero Graph Data
        data = HeteroData()
        data['person'].x = person_nodes.type(torch.int64)
        data['message'].x = message_nodes.type(torch.int64)
        data['know'].x = know_nodes.type(torch.int64)

        data['person', 'speak by', 'message'].edge_index = person_msg.type(torch.int64)
        data['message', 'know by', 'know'].edge_index = msg_know.type(torch.int64)
        data['know', 'relatet to', 'know'].edge_index = know_know.type(torch.int64)

        data = T.ToUndirected()(data)

        return data
