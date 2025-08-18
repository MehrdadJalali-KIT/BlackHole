import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import numpy as np
from torch_geometric.nn import GCNConv, GATConv

logger = logging.getLogger(__name__)

class GraphSAGE(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GraphSAGE, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_h)
        self.linear2 = nn.Linear(dim_h, dim_out)
        self.dim_h = dim_h

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        device = x.device
        self_loops = torch.stack([torch.arange(num_nodes, device=device)] * 2, dim=0)
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=device)
            self_weight = torch.ones(num_nodes, device=device)
        else:
            self_weight = torch.ones(num_nodes, device=device) * edge_weight.mean()
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        edge_weight = torch.cat([edge_weight, self_weight], dim=0)
        adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree = degree.clamp(min=1.0)
        norm = 1.0 / degree
        norm_adj = torch.sparse_coo_tensor(edge_index, edge_weight * norm[edge_index[0]], (num_nodes, num_nodes))
        h = torch.sparse.mm(norm_adj, x)
        h = self.linear1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = torch.sparse.mm(norm_adj, h)
        h = self.linear2(h)
        return h if self.linear2.out_features == 1 else F.log_softmax(h, dim=-1)

class GCN(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)
        self.dim_h = dim_h

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index, edge_weight)
        return h if self.conv2.out_channels == 1 else F.log_softmax(h, dim=-1)

class GAT(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=12):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dim_in, dim_h, heads=heads, dropout=0.3)
        self.norm1 = nn.LayerNorm(dim_h * heads)
        self.conv2 = GATConv(dim_h * heads, dim_out, heads=1, concat=False, dropout=0.3)
        self.dim_h = dim_h
        self.heads = heads

    def forward(self, x, edge_index, edge_weight=None):
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)
        h = self.conv2(h, edge_index)
        return h if self.conv2.out_channels == 1 else F.log_softmax(h, dim=-1)

def train(model, data, epochs=200, lr=0.005, class_weights=None, patience=20, task='classification'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    val_mask = data.test_mask
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        if task == 'classification':
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask], weight=class_weights.to(data.x.device) if class_weights is not None else None)
        else:  # regression
            loss = F.mse_loss(out[data.train_mask].squeeze(), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            val_loss = F.mse_loss(out[val_mask].squeeze(), data.y[val_mask]).item() if task == 'regression' else F.nll_loss(out[val_mask], data.y[val_mask]).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        if epoch % 50 == 0:
            if task == 'classification':
                train_acc = out.argmax(dim=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.size(0)
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            else:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
    return model

def test(model, data, task='classification'):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_weight)
        test_mask = data.test_mask[data.test_mask < data.x.size(0)]
        if test_mask.size(0) == 0:
            logger.warning("Empty test mask, returning zero metrics")
            return (0.0, [], 0.0) if task == 'classification' else (0.0, 0.0, 0.0)
        if task == 'classification':
            pred = out.argmax(dim=1)
            acc = pred[test_mask].eq(data.y[test_mask]).sum().item() / test_mask.size(0)
            cm = confusion_matrix(data.y[test_mask].cpu(), pred[test_mask].cpu())
            kappa = cohen_kappa_score(data.y[test_mask].cpu(), pred[test_mask].cpu())
            logger.info(f"Test Accuracy: {acc:.4f}, Cohen Kappa: {kappa:.4f}")
            return acc, cm, kappa
        else:  # regression
            pred = out.squeeze()
            true = data.y
            mae = mean_absolute_error(true[test_mask].cpu(), pred[test_mask].cpu())
            rmse = np.sqrt(mean_squared_error(true[test_mask].cpu(), pred[test_mask].cpu()))
            r2 = r2_score(true[test_mask].cpu(), pred[test_mask].cpu())
            logger.info(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
            return mae, rmse, r2