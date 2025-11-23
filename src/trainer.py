from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from dataset import get_dataset
from model import NOVA_GNN


class Trainer:
    def __init__(self, config: dict, mixed_precision: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.mixed_precision = mixed_precision

        self.scaler = GradScaler() if mixed_precision else None

        self.writer = SummaryWriter(
            log_dir=config.get("tensorboard_logdir", "results/logs")
        )

        self._load_dataset()
        self._load_loaders(self.config["loader"])
        self._load_model()

        self.best_val_metric = float("-inf")
        self.early_stop_counter = 0
        self.patience = self.config["early_stopping"].get("patience", 8)
        self.min_delta = self.config["early_stopping"].get("min_delta", 0.0)
        self.best_model_path = self.config["early_stopping"].get(
            "path", "results/best_model.pth"
        )

    def _load_dataset(self):
        dataset = get_dataset(self.config["data_directory"], self.device)

        print("Splitting dataset...")
        transform = RandomLinkSplit(
            **self.config["dataset"],
            edge_types=("user", "rates", "movie"),
            rev_edge_types=("movie", "rev_rates", "user"),
        )

        self.train_data, self.val_data, self.test_data = transform(dataset)

    def _load_loaders(self, loader_config: dict):
        print("Preparing data loaders...")

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
        print("Building model...")
        self.model = NOVA_GNN(**self.config["model"]).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), **self.config["optimizer"])

        pos = self.train_data["user", "rates", "movie"].edge_label.sum()
        neg = len(self.train_data["user", "rates", "movie"].edge_label) - pos
        pos_weight = torch.tensor([neg / pos], device=self.device)
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config["scheduler"])

    def _train_step(self, batch):
        batch = batch.to(self.device)
        labels = batch["user", "rates", "movie"].edge_label.float()

        self.optimizer.zero_grad()

        if self.mixed_precision:
            with autocast(device_type="cuda"):
                pred = self.model(batch).view(-1)
                loss = self.criterion(pred, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            pred = self.model(batch).view(-1)
            loss = self.criterion(pred, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            self.optimizer.step()

        return loss.item()
    
    def _log(self, avg_loss, val_auc, val_ap, epoch):
        self.writer.add_scalar("train/loss", avg_loss, epoch)
        self.writer.add_scalar("val/auc", val_auc, epoch)
        self.writer.add_scalar("val/ap", val_ap, epoch)
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

    def train(self):
        num_epochs = self.config["num_epochs"]
        print(f"Training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            with tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{num_epochs} [Train]",
                leave=False,
            ) as pbar:
                for batch in pbar:
                    loss = self._train_step(batch)
                    epoch_loss += loss
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = epoch_loss / len(self.train_loader)
            val_auc, val_ap = self._validate()
            
            self.log(avg_loss, val_auc, val_ap, epoch)

            self.scheduler.step(val_ap)

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | "
                f"Val AP: {val_ap:.4f}"
            )

            if val_ap > self.best_val_metric + self.min_delta:
                self.best_val_metric = val_ap
                self.early_stop_counter = 0

                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved (AP={val_ap:.4f})")

            else:
                self.early_stop_counter += 1
                print(
                    f"EarlyStopping counter: {self.early_stop_counter}/{self.patience}"
                )

                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Training stopped.")
                    break
        
        val_auc, val_ap = self._validate()
        self.writer.add_scalar("test/auc", val_auc, num_epochs+1)
        self.writer.add_scalar("test/ap", val_ap, num_epochs+1)
        
        self.writer.flush()
        self.writer.close()
        
    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        preds, labels = [], []

        with tqdm(self.val_loader, desc="Validating", leave=False) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)

                if self.mixed_precision:
                    with autocast():
                        pred = self.model(batch).view(-1).sigmoid().cpu()
                else:
                    pred = self.model(batch).view(-1).sigmoid().cpu()

                label = batch["user", "rates", "movie"].edge_label.float().cpu()

                preds.append(pred)
                labels.append(label)

        preds = torch.cat(preds)
        labels = torch.cat(labels)

        auc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)

        return auc, ap
