import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from dataset import NovaDataModule
from model import NovaModelModule
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
        data_module: NovaDataModule,
        model_module: NovaModelModule,
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
        self.criterion = BCEWithLogitsLoss()

        self.train_loader = self.data_module.get_train_loader()
        self.valid_loader = self.data_module.get_val_loader()
        self.test_loader = self.data_module.get_test_loader()
        
        self.global_step_train = 0
        self.global_step_valid = 0
        self.global_step_test = 0

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

        self.logger.log("train/batch_loss", loss.item(), self.global_step_train,)
        self.global_step_train += 1

        return loss.item(), preds

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            self.model_module.model.train()
            epoch_loss = epoch_examples = 0.0

            with tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{self.num_epochs} [Train]",
                leave=False,
            ) as pbar:
                for batch in pbar:
                    loss, preds = self._train_step(batch)
                    
                    epoch_loss += float(loss) * preds.numel()
                    epoch_examples += preds.numel()
                    
                    pbar.set_postfix({"loss": f"{loss:.6f}"})

            avg_loss = epoch_loss / epoch_examples
            val_metrics, val_loss = self.validate()
            
            self.model_module.scheduler.step(val_metrics['ap'])
            
            self.logger.log("train/lr", self.model_module.scheduler.get_last_lr()[0], epoch)
            self.logger.log("train/loss", avg_loss, epoch)
            self.logger.log("val/auc", val_metrics["auc"], epoch)
            self.logger.log("val/ap", val_metrics["ap"], epoch)
            self.logger.log("val/loss", val_loss, epoch)

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

        test_metrics, test_loss = self.validate(test=True)

        self.logger.log("test/auc", test_metrics["auc"], 0)
        self.logger.log("test/ap", test_metrics["ap"], 0)
        self.logger.log("test/loss", test_loss, 0)
        self.logger.close()

    @torch.no_grad()
    def validate(self, test: bool = False):
        self.model_module.model.eval()
        preds, labels = [], []
        
        total_loss = total_examples = 0.0

        with tqdm(
            self.valid_loader if not test else self.test_loader,
            desc="Validating" if not test else "Testing",
            leave=False,
        ) as pbar:
            for batch in pbar:
                batch = batch.to(self.device)

                pred = self.model_module.model(batch).view(-1)
                label = batch["user", "rates", "movie"].edge_label.float()
                loss = self.criterion(pred, label)
                
                total_loss += float(loss.item()) * pred.numel()
                total_examples += pred.numel()
                
                if not test:
                    self.logger.log("val/batch_loss", loss, self.global_step_valid)
                    self.global_step_valid += 1
                else:
                    self.logger.log("test/batch_loss", loss, self.global_step_test)
                    self.global_step_test += 1

                pred = pred.sigmoid().cpu()
                label = label.cpu()
                
                preds.append(pred.detach())
                labels.append(label.detach())

        preds = torch.cat(preds).cpu()
        labels = torch.cat(labels).cpu()
        
        avg_loss = total_loss / total_examples

        return Metrics.compute(preds, labels), avg_loss

    def save(self, path: str):
        torch.save(self.model_module.model.state_dict(), f"{path}/final_model.pt")
