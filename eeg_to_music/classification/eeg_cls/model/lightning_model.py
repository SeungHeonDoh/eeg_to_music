import torch
import torch.nn as nn
from sklearn import metrics
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from eeg_to_music.modules.opt_modules import CosineAnnealingWarmupRestarts
from eeg_to_music.modules.loss_modules import accuracy

class EEGCls(LightningModule):
    def __init__(self, model, fusion_type, lr):
        super().__init__()
        self.model = model
        self.fusion_type = fusion_type
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.2
        )
        return [optimizer]
        
    def shared_step(self, batch):
        label = batch['binary'].squeeze(1)
        prediction = self.model(batch['eeg'], batch['wav'])
        loss = self.criterion(prediction, label.argmax(dim=-1))
        return loss, prediction, label

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch)
        self.log_dict(
            {"train_loss": loss},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return loss

    def training_step_end(self, step_output):
        return step_output

    def validation_step(self, batch, batch_idx):
        loss, prediction, labels = self.shared_step(batch)
        return {
            "val_loss": loss,
            "prediction": prediction,
            "labels": labels,
            }

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_prediction = torch.stack([item for output in outputs for item in output["prediction"]])
        val_labels = torch.stack([item for output in outputs for item in output["labels"]])
        val_acc = accuracy(val_prediction, val_labels)
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_acc": val_acc
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        embedding = self.model.extractor(batch['eeg'], batch['wav'])
        loss, prediction, labels = self.shared_step(batch)
        return {
            "val_loss": loss,
            "prediction": prediction,
            "embedding": embedding,
            "labels": labels
            }

    def test_step_end(self, batch_parts):
        return batch_parts

    def test_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_prediction = torch.stack([item for output in outputs for item in output["prediction"]])
        embedding = torch.stack([item for output in outputs for item in output["embedding"]])
        val_labels = torch.stack([item for output in outputs for item in output["labels"]])
        val_acc = accuracy(val_prediction, val_labels)
        results = {
            "test_acc": val_acc
        }
        self.results = results
        self.embedding = {
            "embedding": embedding.detach().cpu().numpy(),
            "val_prediction": val_prediction.detach().cpu().numpy(),
            "val_labels": val_labels.detach().cpu().numpy(),
        }