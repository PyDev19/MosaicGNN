import yaml
from trainer import Trainer

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

gnn_trainer = Trainer(config)

gnn_trainer.train()