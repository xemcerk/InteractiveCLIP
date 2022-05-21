import sys
from typing import Optional
import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader

from datasets.css3d import CSSDataset
from datasets.birdstowords import BirdsToWords
from datasets.fashion200k import Fashion200k
from datasets.fashioniq import FashionIQ
from datasets.mitstates import MITStates
from datasets.spotthediff import SpotTheDiff

def load_dataset(
    dataset,
    dataset_path,
    train_on_validation_set=False,
    batch_size=32,
    processor=None
):
    """Loads the input datasets."""
    print('Reading dataset ', dataset)
    normalizer = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalizer])
    if dataset == 'css3d':
        trainset = CSSDataset(
            path=dataset_path,
            split='train',
            transform=transform)
        testset = CSSDataset(
            path=dataset_path,
            split='test',
            transform=transform)
        dataset_dict = {"train": trainset, "test": testset}
    elif dataset == 'fashion200k':
        trainset = Fashion200k(
            path=dataset_path,
            split='train',
            transform=transform
        )
        testset = Fashion200k(
            path=dataset_path,
            split='test',
            transform=transform
        )
        dataset_dict = {"train": trainset, "test": testset}
    elif dataset == 'mitstates':
        trainset = MITStates(
            path=dataset_path,
            split='train',
            transform=transform
        )
        testset = MITStates(
            path=dataset_path,
            split='test',
            transform=transform
        )
        dataset_dict = {"train": trainset, "test": testset}
    elif dataset == 'fashioniq':
        trainset = FashionIQ(
            path=dataset_path,
            split='joint' if train_on_validation_set else 'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),  # TODO ?
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),  # TODO ?
                normalizer
            ]),
            batch_size=batch_size,
            processor=processor)
        valset_query = FashionIQ(
            path=dataset_path,
            split='val',
            transform=transform,
            batch_size=batch_size,
            val_loader_mode="query",
            processor=processor)
        valset_imgs = FashionIQ(
            path=dataset_path,
            split='val',
            transform=transform,
            batch_size=batch_size,
            val_loader_mode="imgs",
            processor=processor)
        testset = FashionIQ(
            path=dataset_path,
            split='test',
            transform=transform,
            batch_size=batch_size)
        dataset_dict = {"train": trainset, 
                        "val_query": valset_query, 
                        "val_imgs": valset_imgs, 
                        "test": testset}
    elif dataset == 'birds':
        trainset = BirdsToWords(
            path=dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),
                normalizer
            ]),
            batch_size=batch_size)
        valset = BirdsToWords(
            path=dataset_path,
            split='val',
            transform=transform,
            batch_size=batch_size)
        testset = BirdsToWords(
            path=dataset_path,
            split='test',
            transform=transform,
            batch_size=batch_size)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    elif dataset == 'spotthediff':
        trainset = SpotTheDiff(
            path=dataset_path,
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                                                         ratio=(0.75, 1.3)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(
                    lambda xx: xx + 0.01*torch.randn(xx.shape)),
                normalizer
            ]),
            batch_size=batch_size)
        valset = SpotTheDiff(
            path=dataset_path,
            split='val',
            transform=transform,
            batch_size=batch_size)
        testset = SpotTheDiff(
            path=dataset_path,
            split='test',
            transform=transform,
            batch_size=batch_size)
        dataset_dict = {"train": trainset, "val": valset, "test": testset}
    else:
        print('Invalid dataset', dataset)
        sys.exit()

    for name, data in dataset_dict.items():
        print(name, 'size', len(data))
    return dataset_dict

class CIRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        dataset_path,
        train_on_validation_set=False,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        processor=None
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.train_on_validation_set = train_on_validation_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.processor = processor

    def prepare_data(self) -> None:
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset_dict = load_dataset(
            self.dataset,
            self.dataset_path,
            self.train_on_validation_set,
            self.batch_size,
            self.processor
        )
        
    def train_dataloader(self):
        return self.dataset_dict['train'].get_loader(
            batch_size=self.batch_size,
            num_workers=self.num_workers)
        
    def val_dataloader(self, category=None):
        if not category:
            return [
                self.dataset_dict["val_query"].get_loader(
                batch_size=self.batch_size,
                num_workers=self.num_workers
            ),
                self.dataset_dict["val_imgs"].get_loader(
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )]
        else:
            self.dataset_dict["val_query"].current_category = category
            self.dataset_dict["val_query"].filter_category = True
            self.dataset_dict["val_imgs"].current_category = category
            self.dataset_dict["val_imgs"].filter_category = True
            return [
                self.dataset_dict["val_query"].get_loader(
                batch_size=self.batch_size,
                num_workers=self.num_workers
            ),
                self.dataset_dict["val_imgs"].get_loader(
                batch_size=self.batch_size,
                num_workers=self.num_workers
            )]
            