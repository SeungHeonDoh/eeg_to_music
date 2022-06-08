import torch
import torch.nn as nn
from sklearn import metrics
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from eeg_to_music.modules.opt_modules import CosineAnnealingWarmupRestarts

class EEGCls(LightningModule):
    def __init__(self, model, fusion_type, lr):
        super().__init__()
        self.model = model
        self.fusion_type = fusion_type
        self.criterion = nn.MSELoss()
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
        # # Source: https://github.com/openai/CLIP/issues/107
        # num_training_steps = len(self.trainer.datamodule.train_dataloader()) # single-gpu case
        # lr_scheduler = CosineAnnealingWarmupRestarts(
        #     optimizer,
        #     first_cycle_steps=num_training_steps,
        #     cycle_mult=1.0,
        #     max_lr=self.lr,
        #     min_lr=1e-8,
        #     warmup_steps=int(0.2*num_training_steps),
        #     gamma=1.0
        # )
        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
        
    def shared_step(self, batch):
        data, label, subject, trial = batch
        prediction = self.model(data)        
        loss = self.criterion(prediction, label)
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
        loss, _, _ = self.shared_step(batch)
        return {
            "val_loss": loss}

    def validation_step_end(self, step_output):
        return step_output

    def validation_epoch_end(self, outputs):
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        self.log_dict(
            {
                "val_loss": val_loss
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        fnames = batch['fnames']
        loss, prediction, labels = self.shared_step(batch)
        return {
            "fnames" : fnames,
            "val_loss": loss,
            "prediction": prediction,
            "labels": labels
            }

    def test_step_end(self, batch_parts):
        return batch_parts

    def test_epoch_end(self, outputs):
        fnames =[output["fnames"][0] for output in outputs]
        val_loss = torch.mean(torch.stack([output["val_loss"] for output in outputs]))
        val_prediction = torch.stack([item for output in outputs for item in output["prediction"]])
        val_labels = torch.stack([item for output in outputs for item in output["labels"]])
        results = {
            "r2": metrics.r2_score(val_prediction, val_labels),
            "r2_a": metrics.r2_score(val_prediction[:, 0], val_labels[:, 0]),
            "r2_v": metrics.r2_score(val_prediction[:, 1], val_labels[:, 1]),
            "r2_d": metrics.r2_score(val_prediction[:, 2], val_labels[:, 2]),
            "r2_l": metrics.r2_score(val_prediction[:, 3], val_labels[:, 3]),
        }
        self.test_results = results