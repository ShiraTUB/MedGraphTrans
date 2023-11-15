import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from src.utils import node_types, metadata
from torch_geometric.transforms import RandomLinkSplit
import pickle
import torch_geometric.transforms as T


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

    def decode(self, z_dict, pos_edge_index, neg_edge_index=None):
        # Get embeddings of the nodes in the positive edges
        pos_edge_embeddings = (z_dict['question'][pos_edge_index[0]], z_dict['answer'][pos_edge_index[1]])

        # Calculate scores for positive edges using dot product
        pos_scores = (pos_edge_embeddings[0] * pos_edge_embeddings[1]).sum(dim=-1)

        neg_scores = None

        if neg_edge_index is not None:
            # Get embeddings of the nodes in the negative edges
            neg_edge_embeddings = (z_dict['question'][neg_edge_index[0]], z_dict['answer'][neg_edge_index[1]])

            # Calculate scores for negative edges using dot product
            neg_scores = (neg_edge_embeddings[0] * neg_edge_embeddings[1]).sum(dim=-1)

        return pos_scores, neg_scores

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # For link prediction, we return the embeddings instead of a class label
        return x_dict


def compute_loss(pos_scores, neg_scores):
    # Stack all scores and labels
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones(pos_scores.size(0), device=scores.device),
                        torch.zeros(neg_scores.size(0), device=scores.device)])

    # Compute the binary cross-entropy loss
    return F.binary_cross_entropy_with_logits(scores, labels)


def train_on_separate_subgraphs(model, hetero_data_list, optimizer):
    model.train()

    total_loss = 0

    for index, data in enumerate(hetero_data_list):
        optimizer.zero_grad()

        data = data.to(device)

        train_pos_edge_index, train_neg_edge_index = get_edge_indices(data)

        # Forward pass to get node embeddings
        z_dict = model(data.x_dict, data.edge_index_dict)

        # Get the scores from the model's decoder
        pos_scores, neg_scores = model.decode(z_dict, train_pos_edge_index, train_neg_edge_index)

        # Backprop
        loss = compute_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(hetero_data_list)


@torch.no_grad()
def test_separate_subgraph(model, hetero_data_list):
    model.eval()
    total_auc = 0
    total_ap = 0

    # Loop over each data object
    for data in hetero_data_list:
        # Get the node embeddings from the model
        z_dict = model(data.x_dict, data.edge_index_dict)

        # Get the positive and negative edge indices for testing
        pos_edge_index, neg_edge_index = get_edge_indices(data)

        # Get the scores from the model's decoder
        pos_scores, neg_scores = model.decode(z_dict, pos_edge_index, neg_edge_index)

        # Concatenate the positive and negative scores, and apply the sigmoid function
        y_scores = torch.sigmoid(torch.cat([pos_scores, neg_scores], dim=0)).cpu().numpy()

        # True labels: ones for positive edges, zeros for negative edges
        y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))], dim=0).cpu().numpy()

        # Compute the AUC and AP metrics for the current data object
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        # Accumulate the AUC and AP
        total_auc += auc
        total_ap += ap

    # Calculate the average AUC and AP over all data objects
    avg_auc = total_auc / len(hetero_data_list)
    avg_ap = total_ap / len(hetero_data_list)

    return avg_auc, avg_ap


def train(model, data, optimizer, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()

    # Forward pass to get node embeddings
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Decode to get scores for positive and negative edges
    pos_scores, neg_scores = model.decode(z_dict, train_pos_edge_index, train_neg_edge_index)

    # Compute loss
    loss = compute_loss(pos_scores, neg_scores)

    # Backpropagation
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, pos_edge_index, neg_edge_index):
    model.eval()

    # Get the node embeddings from the model
    z_dict = model(data.x_dict, data.edge_index_dict)

    # Get the scores from the model's decoder
    pos_scores, neg_scores = model.decode(z_dict, pos_edge_index, neg_edge_index)

    # Concatenate the positive and negative scores, and apply the sigmoid function
    y_scores = torch.sigmoid(torch.cat([pos_scores, neg_scores], dim=0)).cpu().numpy()

    # True labels: ones for positive edges, zeros for negative edges
    y_true = torch.cat([torch.ones(pos_scores.size(0)), torch.zeros(neg_scores.size(0))], dim=0).cpu().numpy()

    # Compute the AUC and AP metrics
    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return auc, ap


def get_edge_indices(target_data):
    edge_type = 'question_answer'  # Replace with your actual edge type

    pos_edge_index, neg_edge_index = torch.tensor([]), torch.tensor([])

    if len(target_data[edge_type]) > 0:
        pos_edge_index = target_data[edge_type].edge_index  # Positive edge index for training

        # Sample negative edges for this batch
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=(target_data['question'].num_nodes, target_data['answer'].num_nodes),
            num_neg_samples=len(pos_edge_index) * 3,
            method='sparse',
            force_undirected=True
        )

    # TODO: what happens if there are no positive examples (test)

    # Move to device
    pos_edge_index = pos_edge_index.to(device)
    neg_edge_index = neg_edge_index.to(device)

    return pos_edge_index, neg_edge_index


data_path = '../../datasets/merged_hetero_dataset/processed_graph_1000_train_val.pickle'
with open(data_path, 'rb') as f:
    hetero_data = pickle.load(f)

transform = RandomLinkSplit(is_undirected=True, edge_types=("question", "question_answer", "answer"), rev_edge_types=("answer", "rev_question_answer", "question"))
train_data, val_data, test_data = transform(hetero_data)


model = HGT(hidden_channels=64, out_channels=64, num_heads=2, num_layers=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

train_pos_edge_index, train_neg_edge_index = get_edge_indices(train_data)
val_pos_edge_index, val_neg_edge_index = get_edge_indices(val_data)
test_pos_edge_index, test_neg_edge_index = get_edge_indices(test_data)

for epoch in range(1, 15):
    loss = train(model, train_data, optimizer, train_pos_edge_index, train_neg_edge_index)
    val_auc, val_ap = test(model, val_data, val_pos_edge_index, val_neg_edge_index)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')

test_auc, test_ap = test(model, test_data, test_pos_edge_index, test_neg_edge_index)
print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')
