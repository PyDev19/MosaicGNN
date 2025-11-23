import torch
import yaml
from dataset import RecommenderDataModule
from model import RecommenderModelModule
from trainer import Trainer

DATA_DIR = "data/"

config = yaml.safe_load(open("config.yaml", "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = RecommenderDataModule(DATA_DIR, config["dataset"], config["loader"])
model_module = RecommenderModelModule(
    config["model"], config["optimizer"], config["scheduler"], device
)

data_module.setup()

gnn_trainer = Trainer(
    **config["trainer"],
    **config["early_stopping"],
    device=device,
    data_module=data_module,
    model_module=model_module,
)

gnn_trainer.train()
gnn_trainer.save("results")