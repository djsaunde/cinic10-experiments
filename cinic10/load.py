import os
import torchvision.transforms as transforms

from fastai.dataset import ModelData
from torch.utils.data import DataLoader
from fastai.conv_learner import num_cpus
from torchvision.datasets import ImageFolder

from .definitions import ROOT_DIR


def load_cinic10(directory=os.path.join(ROOT_DIR, 'data'), batch_size=128, n_workers=num_cpus()):
    # language=rst
    """
    Loads the CINIC-10 natural images dataset. Assumes it is downloaded and unzipped in ``directory``. Custom
    transformations are used.

    :param directory: Path to folder containing unzipped dataset.
    :param batch_size: Parameter to pass to ``torch.utils.data.DataLoader`` object.
    :param n_workers: No. of CPUs to use for dataset loading.
    :param cuda: Whether to put data on GPUs.
    """
    mean = [0.47889522, 0.47227842, 0.43047404]
    std = [0.24205776, 0.23828046, 0.25874835]
    stats = (mean, std)

    tfms = [transforms.ToTensor(), transforms.Normalize(*stats)]
    aug_tfms = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + tfms
    )

    train_dir = directory + '/train'
    valid_dir = directory + '/valid'
    test_dir = directory + '/test'

    # PyTorch datasets
    train_ds = ImageFolder(train_dir, aug_tfms)
    valid_ds = ImageFolder(valid_dir, transforms.Compose(tfms))
    augment_ds = ImageFolder(valid_dir, aug_tfms)
    test_ds = ImageFolder(test_dir, tfms)

    # PyTorch data loaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, num_workers=n_workers, shuffle=True, pin_memory=True
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True
    )
    augment_dl = DataLoader(
        augment_ds, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, num_workers=n_workers, shuffle=False, pin_memory=True
    )

    # FastAI model data
    data = ModelData(directory, trn_dl=train_dl, val_dl=valid_dl, test_dl=test_dl)
    data.augment_dl = augment_dl
    data.test_dl = test_dl
    data.sz = 32

    return data
