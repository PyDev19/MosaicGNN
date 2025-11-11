from sklearn.metrics import roc_auc_score, average_precision_score
import torch

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1).sigmoid().cpu()
        label = batch["user", "rates", "movie"].edge_label.float().cpu()

        preds.append(pred)
        labels.append(label)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    auc = roc_auc_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return auc, ap
