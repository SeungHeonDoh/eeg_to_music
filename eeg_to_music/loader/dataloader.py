from typing import Callable, Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, BatchSampler

from eeg_to_music.loader.dataset import DEAP_Dataset

class DataPipeline(LightningDataModule):
    def __init__(self, feature_type, label_type, batch_size, num_workers) -> None:
        super(DataPipeline, self).__init__()
        self.dataset_builder = DEAP_Dataset        
        self.batch_size = batch_size
        self.feature_type = feature_type
        self.label_type = label_type
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TRAIN",
                feature_type = self.feature_type,
                label_type = self.label_type
            )

            self.val_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "VALID",
                feature_type = self.feature_type,
                label_type = self.label_type
            )

        if stage == "test" or stage is None:
            self.test_dataset = DataPipeline.get_dataset(
                self.dataset_builder,
                split = "TEST",
                feature_type = self.feature_type,
                label_type = self.label_type
            )

    def train_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = True
        )

    def val_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = False
        )

    def test_dataloader(self) -> DataLoader:
        return DataPipeline.get_dataloader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle = False
        )

    @classmethod
    def get_dataset(cls, dataset_builder: Callable, split, feature_type, label_type) -> Dataset:
        dataset = dataset_builder(split, feature_type, label_type)
        return dataset

    @classmethod
    def get_dataloader(cls, dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, **kwargs) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers = num_workers, 
            persistent_workers=False,
            shuffle = shuffle,
            **kwargs
        )
