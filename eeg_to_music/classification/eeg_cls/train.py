import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from eeg_to_music.classification.eeg_cls.model.backbone import SampleCNN
from eeg_to_music.classification.eeg_cls.model.lightning_model import EEGCls
from eeg_to_music.loader.dataloader import DataPipeline


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

# def get_wandb_logger(model):
#     logger = WandbLogger()
#     logger.watch(model)
#     return logger 

    
def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/test", name=args.supervisions, version=f"test/"
    )
    return logger

def get_checkpoint_callback(save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback() -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )
    return early_stop_callback

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/test"
    save_hparams(args, save_path)

    # wandb.init(config=args)
    # wandb.run.name = f"test"
    # args = wandb.config

    pipeline = DataPipeline(
            batch_size = args.batch_size,
            num_workers = args.num_workers
    )

    model = SampleCNN()
    runner = EEGCls(
            model = model,
            fusion_type = args.fusion_type,
            lr = args.lr, 
    )

    # logger = WandbLogger()
    checkpoint_callback = get_checkpoint_callback(save_path)
    early_stop_callback = get_early_stop_callback()
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes= args.num_nodes,
                    accelerator='gpu',
                    strategy = args.strategy,
                    devices= args.gpus,
                    # strategy = DDPPlugin(find_unused_parameters=True),
                    # logger=logger,
                    # log_every_n_steps=1,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    replace_sampler_ddp=False,
                    callbacks=[
                        early_stop_callback,
                        checkpoint_callback,
                        lr_monitor_callback
                    ],
                )
    trainer.fit(runner, datamodule=pipeline)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--fusion_type", default='eeg', type=str)
    # runner 
    parser.add_argument("--lr", default=5e-4, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool) 
    args = parser.parse_args()
    main(args)