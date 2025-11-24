import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from dataset import RecommenderDataModule
from model import RecommenderModelModule
from utils import Metrics, TBLogger


class NovaTrainer:
    def __init__(
        self,
        num_epochs: int,
        log_dir: str,
        mixed_precision: bool,
        patience: int,
        min_delta: float,
        best_model_path: str,
        device: torch.device,
        data_module: RecommenderDataModule,
        model_module: RecommenderModelModule,
    ):
        self.data_module = data_module
        self.model_module = model_module
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision
        self.device = device

        self.scaler = GradScaler() if mixed_precision else None
        self.logger = TBLogger(log_dir)

        self.best_val_metric = float("-inf")
        self.early_stop_counter = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_model_path = best_model_path

        pos = self.data_module.train_data["user", "rates", "movie"].edge_label.sum()
        neg = (
            len(self.data_module.train_data["user", "rates", "movie"].edge_label) - pos
        )
        pos_weight = torch.tensor([neg / pos], device=self.device)
        self.criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_loader = self.data_module.get_train_loader()
        self.valid_loader = self.data_module.get_val_loader()
        self.test_loader = self.data_module.get_test_loader()

        sample_batch = next(iter(self.train_loader)).to(self.device)
        self.logger.graph(self.model_module.model, sample_batch)

    def _train_step(self, batch):
        batch = batch.to(self.device)
        labels = batch["user", "rates", "movie"].edge_label.float()

        self.model_module.optimizer.zero_grad()

        if self.mixed_precision:
            with autocast(device_type="cuda"):
                preds = self.model_module.model(batch).view(-1)
                loss = self.criterion(preds, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.model_module.optimizer)

            torch.nn.utils.clip_grad_norm_(self.model_module.parameters(), 2.0)

            self.scaler.step(self.model_module.optimizer)
            self.scaler.update()
        else:
            preds = self.model_module.model(batch).view(-1)
            loss = self.criterion(preds, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model_module.parameters(), 2.0)
            self.model_module.optimizer.step()

        self.logger.log(
            "train/batch_loss",
            loss.item(),
            self.model_module.optimizer.state.get("step", 0),
        )

        return loss.item()

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model_module.model.train()
            epoch_loss = 0.0

            with tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs} [Train]",
                leave=False,
            ) as pbar:
                for batch in pbar:
                    loss = self._train_step(batch)
                    epoch_loss += loss
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_loss = epoch_loss / len(self.train_loader)
            val_metrics = self.validate()

            self.logger.log("train/loss", avg_loss, epoch)
            self.logger.log("val/auc", val_metrics["auc"], epoch)
            self.logger.log("val/ap", val_metrics["ap"], epoch)

            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val AP: {val_metrics['ap']:.4f}"
            )

            if val_metrics["ap"] > self.best_val_metric + self.min_delta:
                self.best_val_metric = val_metrics["ap"]
                self.early_stop_counter = 0

                torch.save(self.model_module.model.state_dict(), self.best_model_path)
                print(f"New best model saved (AP={val_metrics['ap']:.4f})")

            else:
                self.early_stop_counter += 1
                print(
                    f"EarlyStopping counter: {self.early_stop_counter}/{self.patience}"
                )

                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Training stopped.")
                    break

        test_metrics = self.validate(test=True)

        self.logger.log("test/auc", test_metrics["auc"], self.num_epochs + 1)
        self.logger.log("test/ap", test_metrics["ap"], self.num_epochs + 1)
        self.logger.close()

    @torch.no_grad()
    def validate(self, test: bool = False):
        self.model_module.model.eval()
        preds, labels = [], []

        with tqdm(
            self.valid_loader if not test else self.test_loader,
            desc="Validating" if not test else "Testing",
            leave=False,
        ) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)

                pred = self.model_module.model(batch).view(-1).sigmoid().cpu()

                label = batch["user", "rates", "movie"].edge_label.float().cpu()

                preds.append(pred.detach())
                labels.append(label.detach())

        preds = torch.cat(preds).cpu()
        labels = torch.cat(labels).cpu()

        return Metrics.compute(preds, labels)

    def save(self, path: str):
        torch.save(self.model_module.model.state_dict(), f"{path}/final_model.pt")
