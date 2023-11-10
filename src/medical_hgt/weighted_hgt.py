import pickle
import torch
from sklearn.metrics import roc_auc_score

import torch_geometric.transforms as T
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from src.utils import node_types, metadata
from torch_geometric.data import HeteroData

import torch.nn.functional as F


data_path = '../../datasets/merged_hetero_dataset/processed_graph_1000_train_val.pickle'
with open(data_path, 'rb') as f:
    hetero_data = pickle.load(f)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("question", "question_answer", "answer"),
    rev_edge_types=("answer", "rev_question_answer", "question"),
)
train_data, val_data, test_data = transform(hetero_data)

# Define seed edges:
edge_label_index = train_data["question", "question_answer", "answer"].edge_label_index
edge_label = train_data["question", "question_answer", "answer"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(("question", "question_answer", "answer"), edge_label_index),
    edge_label=edge_label,
    batch_size=128,
    shuffle=True,
)

# sampled_data = next(iter(train_loader))


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata,
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Decoder(torch.nn.Module):
    def forward(self, x_question: torch.Tensor, x_answer: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_question[edge_label_index[0]]
        edge_feat_movie = x_answer[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class EdgeWeightUpdater(torch.nn.Module):
    def __init__(self, data):
        super().__init__()

        # Initialize a weight for each edge in the graph
        self.edge_weights_dict = torch.nn.ParameterDict()
        for edge_type in data.edge_types:
            # Initialize the weights for each edge
            num_edges = data[edge_type].num_edges
            self.edge_weights_dict[edge_type[1]] = torch.nn.Parameter(torch.rand(num_edges, requires_grad=True))

    def forward(self, edge_type, edge_index):
        # Forward pass to retrieve the weights for the edges
        return self.edge_weights_dict[edge_type][edge_index]

    # def update_weights(self, loss, optimizer):
    #     # Compute gradients for edge weights
    #     loss.backward()
    #     # Step the optimizer to update the weights
    #     optimizer.step()


class Model(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        self.hgt = HGT(hidden_channels=64, out_channels=64, num_heads=2, num_layers=1)
        self.encoder = Decoder()
        # Initialize learnable edge weights
        self.edge_weights = torch.nn.ParameterDict({
            edge_type[1]: torch.nn.Parameter(torch.rand(size=(1, data[edge_type].num_edges), requires_grad=True))
            for edge_type in data.edge_types
        })

    def forward(self, data: HeteroData) -> torch.Tensor:

        # include edges weights before passing to hgt: apply weight to all source nodes for message passing
        weighted_x_dict = {node_type: feature_tensor for node_type, feature_tensor in data.x_dict.items()}
        for edge_type, edge_indices in data.edge_index_dict.items():
            source_node_type = edge_type[0]
            source_nodes_indices = edge_indices[0]
            for edge_index, source_node_index in enumerate(source_nodes_indices):
                edge_weight = self.edge_weights[edge_type[1]][0][edge_index]
                weighted_x_dict[source_node_type][source_node_index] *= edge_weight

        x_dict = self.hgt(weighted_x_dict, data.edge_index_dict)
        pred = self.encoder(
            x_dict["question"],
            x_dict["answer"],
            data["question", "question_answer", "answer"].edge_label_index,
        )
        return pred


# edge_weights_updater = EdgeWeightUpdater(train_data)
model = Model(train_data)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

edge_weights_dict = {edge_type: torch.rand((1, hetero_data[edge_type].num_edges)) for edge_type in hetero_data.edge_types}

for epoch in range(1, 20):
    optimizer.zero_grad()
    # Get edge weights for all edge types

    train_data.to(device)
    pred = model(train_data)
    ground_truth = train_data["question", "question_answer", "answer"].edge_label
    link_prediction_loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    link_prediction_loss.backward()
    optimizer.step()
    total_loss = float(link_prediction_loss) * pred.numel()
    total_examples = pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    # with minibatching
    # total_loss = total_examples = 0
    # for sampled_data in tqdm.tqdm(train_loader):
    #     optimizer.zero_grad()
    #     sampled_data.to(device)
    #     pred = model(sampled_data)
    #     ground_truth = sampled_data["question", "question_answer", "answer"].edge_label
    #     loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += float(loss) * pred.numel()
    #     total_examples += pred.numel()
    # print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

preds = []
ground_truths = []
with torch.no_grad():
    val_data.to(device)
    preds.append(model(val_data))
    ground_truths.append(val_data["question", "question_answer", "answer"].edge_label)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
auc = roc_auc_score(ground_truth, pred)
print()
print(f"Validation AUC: {auc:.4f}")

print('end')
