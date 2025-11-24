from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter


class Metrics:
    @staticmethod
    def compute(preds, labels):
        return {
            "auc": roc_auc_score(labels, preds),
            "ap": average_precision_score(labels, preds),
        }


class TBLogger:
    def __init__(self, logdir: str):
        self.writer = SummaryWriter(logdir)

    def log(self, key, value, step):
        self.writer.add_scalar(key, value, step)
        self.writer.flush()
        
    def close(self):
        self.writer.flush()
        self.writer.close()
