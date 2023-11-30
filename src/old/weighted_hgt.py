import pickle
import torch
import tqdm
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.metrics import roc_auc_score
from torch_geometric.nn import Linear
from torch_geometric.loader import LinkNeighborLoader
from src.utils import node_types, metadata
from torch_geometric.data import HeteroData

from src.old.weighted_hgt_conv import WeightedHGTConv

data_path = '../../datasets/merged_hetero_dataset/processed_graph_1000_train_val_masked_with_edge_uids.pickle'

with open(data_path, 'rb') as f:
    hetero_data = pickle.load(f)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    is_undirected=True,
    disjoint_train_ratio=0.2,
    neg_sampling_ratio=3.0,
    add_negative_train_samples=False,
    edge_types=("question", "question_answer", "answer"),
    rev_edge_types=("answer", "rev_question_answer", "question"),
)

train_data, val_data, test_data = transform(hetero_data)

# Perform mini batching
edge_label_index = train_data["question", "question_answer", "answer"].edge_label_index
edge_label = train_data["question", "question_answer", "answer"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data.contiguous(),
    num_neighbors=[4, 3, 2, 10, 10, 3],
    neg_sampling_ratio=3.0,
    edge_label_index=(("question", "question_answer", "answer"), edge_label_index),
    edge_label=edge_label,
    batch_size=32,
    shuffle=True
)

edge_label_index = val_data["question", "question_answer", "answer"].edge_label_index
edge_label = val_data["question", "question_answer", "answer"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data.contiguous(),
    num_neighbors=[4, 3, 2, 10, 10, 3],
    edge_label_index=(("question", "question_answer", "answer"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)

edge_label_index = test_data["question", "question_answer", "answer"].edge_label_index
edge_label = test_data["question", "question_answer", "answer"].edge_label
test_loader = LinkNeighborLoader(
    data=val_data.contiguous(),
    num_neighbors=[4, 3, 2, 10, 10, 3],
    edge_label_index=(("question", "question_answer", "answer"), edge_label_index),
    edge_label=edge_label,
    batch_size=3 * 128,
    shuffle=False,
)


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = WeightedHGTConv(hidden_channels, hidden_channels, metadata, num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data, edge_weights_dict):
        x_dict = {node_type: self.lin_dict[node_type](x).relu_() for node_type, x in data.x_dict.items()}

        # # include edges weights before passing to hgt: apply weight to all source nodes for message passing
        # weighted_x_dict = {node_type: feature_tensor.clone() for node_type, feature_tensor in x_dict.items()}
        # for edge_type, edge_indices in data.edge_index_dict.items():
        #     source_node_type = edge_type[0]
        #     source_nodes_indices = edge_indices[0]
        #
        #     # apply a sigmoid function to keep weights in (0, 1)
        #     # edge_weights[edge_type[1]].data = torch.sigmoid(edge_weights[edge_type[1]].data)
        #
        #     edge_type_weights = edge_weights[edge_type[1]].squeeze()
        #     weighted_x_dict[source_node_type][source_nodes_indices] *= edge_type_weights.unsqueeze(1).repeat(1, weighted_x_dict[source_node_type].size(1))

        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict, edge_weights_dict)

        return x_dict


# Our decoder applies the dot-product between source and destination node embeddings to derive edge-level predictions:
class Decoder(torch.nn.Module):
    def forward(self, x_question: torch.Tensor, x_answer: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_question = x_question[edge_label_index[0]]
        edge_feat_answer = x_answer[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_question * edge_feat_answer).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, edge_index_dict):
        super().__init__()
        self.hgt = HGT(hidden_channels=64, out_channels=64, num_heads=2, num_layers=1)
        self.decoder = Decoder()

        # Initialize learnable edge weights
        self.edge_weights_dict = torch.nn.ParameterDict()

        for edge_type, edge_indices in edge_index_dict.items():
            edge_type = '__'.join(edge_type)
            parameter_tensor = torch.nn.Parameter(torch.randn((1, edge_indices.size(1))), requires_grad=True)
            self.edge_weights_dict[edge_type] = F.sigmoid(parameter_tensor)

    def forward(self, complete_graph_data, batch_data: HeteroData) -> (torch.Tensor, dict):
        relevant_edge_weights_dict = {}

        # find the relevant indices in the models weights dict
        for edge_type in batch_data.edge_types:
            relevant_indices = complete_graph_data[edge_type].edge_uid[batch_data[edge_type].edge_uid]
            edge_type = '__'.join(edge_type)
            relevant_edge_weights_dict[edge_type] = self.edge_weights_dict[edge_type][0][relevant_indices]

        weighted_z_dict = self.hgt(batch_data, relevant_edge_weights_dict)

        pred = self.decoder(
            weighted_z_dict["question"],
            weighted_z_dict["answer"],
            batch_data["question", "question_answer", "answer"].edge_label_index,
        )

        pred = F.sigmoid(pred)

        # Make sure edges weights are between 0 and 1
        # for edge_type in self.edge_weights_dict.keys():
        #     self.edge_weights_dict[edge_type].data = self.sigmoid(self.edge_weights_dict[edge_type].data)

        return pred


model = Model(hetero_data.edge_index_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, complete_graph_data, train_loader):
    model.train()
    total_loss = total_examples = 0

    # Mini-Batching
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(complete_graph_data, sampled_data)
        ground_truth = sampled_data["question", "question_answer", "answer"].edge_label

        link_prediction_loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        link_prediction_loss.backward()
        optimizer.step()

        total_loss += float(link_prediction_loss) * pred.numel()
        total_examples += pred.numel()

    return total_loss / total_examples, model.edge_weights_dict


@torch.no_grad()
def test(model, complete_graph_data, test_loader):
    model.eval()
    preds = []
    ground_truths = []

    # Mini-Batching
    for sampled_data in tqdm.tqdm(test_loader):
        with torch.no_grad():
            sampled_data.to(device)
            pred = model(complete_graph_data, sampled_data)
            preds.append(pred)
            ground_truths.append(sampled_data["question", "question_answer", "answer"].edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    return auc, model.edge_weights_dict


def weights_decoder(hetero_data, edge_weights):
    top_10_edge_per_type = {}

    for edge_type in edge_weights.keys():
        weights = edge_weights[edge_type].data
        top_10_indices = torch.topk(weights, 10).indices.squeeze()
        edge_type_tuple = edge_type.split('__')
        edge_type_tuple = (edge_type_tuple[0], edge_type_tuple[1], edge_type_tuple[2])
        top_10_edge_per_type[edge_type] = hetero_data[edge_type_tuple].edge_index[:, top_10_indices]

        # todo: continue

    return 0


for epoch in range(1, 20):
    loss, _ = train(model, hetero_data, train_loader)
    val_auc, _ = test(model, hetero_data, val_loader)
    print(f"\nEpoch: {epoch:03d}, Train Loss: {loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

test_auc, edge_weights = test(model, hetero_data, test_loader)
most_relevant_nodes = weights_decoder(hetero_data, edge_weights)
print(f"Test AUC: {test_auc:.4f}")

print('test')
