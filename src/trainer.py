from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from dataset import get_dataset
from model import NOVA_GNN

class Trainer:
    def __init__(self, config: dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        
        self._load_dataset()
        self._load_loaders(self.config["loader"])
        self._load_model()
    
    def _load_dataset(self):
        dataset = get_dataset(self.config["data_directory"], self.device)
        
        transform = RandomLinkSplit(
            **self.config["dataset"],
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
        )

        self.train_data, self.val_data, self.test_data = transform(dataset)
    
    def _load_loaders(self, loader_config: dict):
        self.train_loader = LinkNeighborLoader(
            **loader_config["train"],
            data=self.train_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.train_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.train_data["user", "rates", "movie"].edge_label,
        )
        
        self.val_loader = LinkNeighborLoader(
            **loader_config["val_test"],
            data=self.val_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.val_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.val_data["user", "rates", "movie"].edge_label,
        )
        
        self.test_loader = LinkNeighborLoader(
            **loader_config["val_test"],
            data=self.test_data,
            edge_label_index=(
                ("user", "rates", "movie"),
                self.test_data["user", "rates", "movie"].edge_label_index,
            ),
            edge_label=self.test_data["user", "rates", "movie"].edge_label,
        )

    def _load_model(self):
        self.model = NOVA_GNN(**self.config["model"]).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), **self.config["optimizer"])
        
        pos = self.train_data["user", "rates", "movie"].edge_label.sum()
        neg = len(self.train_data["user", "rates", "movie"].edge_label) - pos
        pos_weight = torch.tensor([neg / pos], device=self.device)
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config["scheduler"])
    
    def _train_step(self, batch):
        batch = batch.to(self.device)

        pred = self.model(batch).view(-1)
        labels = batch["user", "rates", "movie"].edge_label.float()
        
        loss = self.criterion(pred, labels)

        self.optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        num_epochs = self.config['num_epochs']
        losses = []
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.total_loss = 0.0
            
            with tqdm(self.train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False) as pbar:
                for batch in pbar:
                    loss = self._train_step(batch)
                    self.total_loss += loss
                    pbar.set_postfix({"batch_loss": f"{loss:.4f}"})
            
            avg_loss = self.total_loss / len(self.train_loader)
            losses.append(avg_loss)
            
            if epoch > 1 and avg_loss <= min(losses):
                torch.save(self.model.state_dict(), f"results/mosaic_gnn_epoch{epoch}.pth")
            
            val_auc, val_ap = self._validate()
            self.scheduler.step(val_ap)
            
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f}")
    
    @torch.no_grad()     
    def _validate(self):
        self.model.eval()
        preds, labels = [], []

        for batch in self.val_loader:
            batch = batch.to(self.device)
            pred = self.model(batch).view(-1).sigmoid().detach().cpu()
            label = batch["user", "rates", "movie"].edge_label.float().cpu()

            preds.append(pred)
            labels.append(label)

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
                
        return auc, ap
