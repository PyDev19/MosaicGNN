import torch
import yaml
from dataset import NovaDataModule
from model import NovaModelModule
from trainer import NovaTrainer

DATA_DIR = "data/"

config = yaml.safe_load(open("configs/SAGEConv_16.yaml", "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_module = NovaDataModule(DATA_DIR, config["dataset"], config["loader"])
data_module.setup()

model_module = NovaModelModule(
    config["model"],
    config["optimizer"],
    config["scheduler"],
    device=device,
    num_users=data_module.get_user_nodes(),
    num_movies=data_module.get_movie_nodes(),
    metadata=data_module.get_metadata(),
)


nova_trainer = NovaTrainer(
    **config["trainer"],
    **config["early_stopping"],
    device=device,
    data_module=data_module,
    model_module=model_module,
)

nova_trainer.train()
