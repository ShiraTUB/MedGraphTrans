import pickle
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T

from sklearn.metrics import roc_auc_score
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.loader import LinkNeighborLoader
from src.utils import node_types, metadata
from torch_geometric.data import HeteroData

from src.medical_hgt.ml_utils import split_data


data_path = '../../datasets/merged_hetero_dataset/processed_graph_1000_train_val.pickle'
with open(data_path, 'rb') as f:
    hetero_data = pickle.load(f)

# transform = T.RandomLinkSplit(
#     is_undirected=True,
#     edge_types=("question", "question_answer", "answer"),
#     rev_edge_types=("answer", "rev_question_answer", "question"),
# )
# train_data, val_data, test_data = transform(hetero_data)


# Define seed edges:
# edge_label_index = train_data["question", "question_answer", "answer"].edge_label_index
# edge_label = train_data["question", "question_answer", "answer"].edge_label
# train_loader = LinkNeighborLoader(
#     data=train_data.contiguous(),
#     num_neighbors=[5] * len(train_data.node_types),
#     neg_sampling_ratio=2.0,
#     edge_label_index=(("question", "question_answer", "answer"), edge_label_index),
#     edge_label=edge_label,
#     batch_size=32,
#     shuffle=True,
# )

train_data, val_data, test_data = split_data(hetero_data, labels_ratio=0.2, neg_labels_ratio=2)


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

    def forward(self, data, edge_weights):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in data.x_dict.items()
        }

        # include edges weights before passing to hgt: apply weight to all source nodes for message passing
        weighted_x_dict = {node_type: feature_tensor.clone() for node_type, feature_tensor in x_dict.items()}
        for edge_type, edge_indices in data.edge_index_dict.items():
            source_node_type = edge_type[0]
            source_nodes_indices = edge_indices[0]

            # apply a sigmoid function to keep weights in (0, 1)
            edge_weights[edge_type[1]].data = torch.sigmoid(edge_weights[edge_type[1]].data)

            edge_type_weights = edge_weights[edge_type[1]].squeeze()
            weighted_x_dict[source_node_type][source_nodes_indices] *= edge_type_weights.unsqueeze(1).repeat(1, weighted_x_dict[source_node_type].size(1))  # for edge_index, source_node_index in enumerate(source_nodes_indices):
            #     edge_weight = edge_weights[edge_type[1]][0][edge_index]
            #     weighted_x_dict[source_node_type][source_node_index] *= edge_weight

        for conv in self.convs:
            weighted_x_dict = conv(weighted_x_dict, data.edge_index_dict)

        return weighted_x_dict


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
        self.edge_weights = torch.nn.ParameterDict({
            edge_type[1]: torch.nn.Parameter(torch.rand(size=(1, edge_indices.size(1)), requires_grad=True))
            for edge_type, edge_indices in edge_index_dict.items()
        })

    def forward(self, complete_graph_data, split_graph_data: HeteroData) -> (torch.Tensor, dict):

        relevant_edge_weights = {}

        for edge_type, edge_indices in split_graph_data.edge_index_dict.items():
            # find the relevant indices in the models weights dict
            data_split_indices = edge_indices
            complete_edge_type_indices = complete_graph_data[edge_type].edge_index

            # Expand dimensions for broadcasting
            data_split_indices = data_split_indices.unsqueeze(1)
            complete_edge_type_indices = complete_edge_type_indices.unsqueeze(2)

            # Compare elements
            matches = (complete_edge_type_indices == data_split_indices)

            # Check if both elements of the pair match
            matches_all = matches.all(dim=0)

            # Find indices where the pairs are equal
            relevant_indices = matches_all.nonzero(as_tuple=True)[1]
            relevant_edge_weights[edge_type[1]] = self.edge_weights[edge_type[1]][0][relevant_indices]

        weighted_z_dict = self.hgt(split_graph_data, relevant_edge_weights)
        pred = self.decoder(
            weighted_z_dict["question"],
            weighted_z_dict["answer"],
            split_graph_data["question", "question_answer", "answer"].edge_label_index,
        )
        return pred, self.edge_weights


model = Model(hetero_data.edge_index_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, complete_graph_data, train_data):
    model.train()
    optimizer.zero_grad()

    train_data.to(device)
    pred, edge_weights = model(complete_graph_data, train_data)
    ground_truth = train_data["question", "question_answer", "answer"].edge_label
    link_prediction_loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    link_prediction_loss.backward()
    optimizer.step()
    total_loss = float(link_prediction_loss) * pred.numel()
    total_examples = pred.numel()
    return total_loss / total_examples

    # # with minibatching
    # total_loss = total_examples = 0
    # for sampled_data in tqdm.tqdm(train_loader):
    #     optimizer.zero_grad()
    #     sampled_data.to(device)
    #     pred = model(sampled_data)
    #     ground_truth = sampled_data["question", "question_answer", "answer"].edge_label
    #
    #     # Compute the loss with regularization
    #     reg_loss = 0
    #     for param in model.edge_weights.values():
    #         reg_loss += torch.norm(param)
    #     loss = F.binary_cross_entropy_with_logits(pred, ground_truth) + l2_lambda * reg_loss
    #     loss.backward()
    #     optimizer.step()
    #     total_loss += float(loss) * pred.numel()
    #     total_examples += pred.numel()
    # print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")


@torch.no_grad()
def test(model, complete_graph_data, test_data):
    model.eval()
    preds = []
    ground_truths = []
    val_data.to(device)
    prediction, edges_weights = model(complete_graph_data, test_data)
    preds.append(prediction)
    ground_truths.append(val_data["question", "question_answer", "answer"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    return auc


for epoch in range(1, 101):
    loss = train(model, hetero_data, train_data)
    val_auc = test(model, hetero_data, val_data)
    print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    print()

print('end')
