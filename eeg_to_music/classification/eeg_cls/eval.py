import json
import os
from argparse import ArgumentParser, Namespace, ArgumentTypeError
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.plugins import DDPPlugin

from eeg_to_music.classification.eeg_cls.model.backbone import FusionModel
from eeg_to_music.classification.eeg_cls.model.lightning_model import EEGCls
from eeg_to_music.loader.dataloader import DataPipeline


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def save_hparams(args, save_path):
    save_config = OmegaConf.create(vars(args))
    os.makedirs(save_path, exist_ok=True)
    OmegaConf.save(config=save_config, f= Path(save_path, "hparams.yaml"))

def main(args):
    if args.reproduce:
        seed_everything(42)

    save_path = f"exp/{args.fusion_type}/{args.feature_type}_{args.label_type}"
    pipeline = DataPipeline(
            feature_type = args.feature_type,
            label_type = args.label_type,
            batch_size = args.batch_size,
            num_workers = args.num_workers
    )
    if args.label_type == "av":
        n_class = 4
    else:
        n_class = 2
    
    model = FusionModel(
                eeg_feature_dim=2016,
                audio_feature_dim=13,
                fusion_type=args.fusion_type,
                hidden_dim=64,
    )
    
    runner = EEGCls(
            model = model,
            fusion_type = args.fusion_type,
            lr = args.lr, 
    )
    state_dict = torch.load(os.path.join(save_path, "best.ckpt"), map_location="cpu")
    runner.load_state_dict(state_dict.get("state_dict"))
    trainer = Trainer(
                    max_epochs= args.max_epochs,
                    num_nodes= args.num_nodes,
                    accelerator='gpu',
                    strategy = args.strategy,
                    devices= args.gpus,
                    sync_batchnorm=True,
                    resume_from_checkpoint=None,
                    replace_sampler_ddp=False,
                )
    trainer.test(runner, datamodule=pipeline)
    torch.save(runner.embedding, os.path.join(save_path, f"{args.data_type}_{args.label_type}_inference.pt"))
    with open(Path(save_path, f"{args.data_type}_{args.label_type}_results.json"), mode="w") as io:
        json.dump(runner.results, io, indent=4)
    print("finish save")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fusion_type", default='intra', type=str)
    parser.add_argument("--data_type", default='deap', type=str)
    parser.add_argument("--label_type", default='v', type=str)
    parser.add_argument("--feature_type", default='psd', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    # runner 
    parser.add_argument("--lr", default=5e-4, type=float)
    # trainer
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=[0], type=list)
    parser.add_argument("--strategy", default="dp", type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument("--reproduce", default=False, type=str2bool) 
    args = parser.parse_args()
    main(args)