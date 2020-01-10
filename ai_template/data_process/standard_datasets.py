# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - standard_datasets.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of Pytorch Dataset classes for standard datasets.

All datasets are subclasses of Standard_Dataset which is a subclass of torch.utils.data_process.Dataset. Hence, they can all be passed to a torch.utils.data_process.DataLoader.
They have __getitem__ and __len__ methods,  implemented
All datasets have the classes attribute which is a list of all classes.
All datasets implement the create_samples() method to create a samples subset from the train and test dataset, located in the same directory.

The following datasets have currently been implemented:
- MNIST

"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ai_template.configuration import cfg
from skimage import io
from torch.utils.data import Dataset

# from configuration import cfg
import torch as th

__all__ = ['MNIST_Dataset', 'FashionMNIST_Dataset']

_base_path = cfg.datasets_path


class Standard_Dataset(Dataset):
    """
    The base class for standard Datasets. It's a subclass of torch.utils.data_process.Dataset.
    """
    name = ''
    classes = None

    def __init__(self, sample=False, test=False, transform=None):
        """
        Args:
              sample (bool): If True then the datasets contains only a limited amount of pictures.
                If False, the datasets contains all the available images.
              test: If True then the dataset contains the testimages. If false then the dataset contains the train images.
              transform: An optional function/transform that takes in an PIL image and returns a transformed version.
                E.g, ``transforms.RandomCrop``

        Returns:
            tuple(image, target) where image is a transformed PIL image (numpy array) and target is the index of the target class
        """

        super(Standard_Dataset, self).__init__()
        self.transform = transform
        assert self.name, "The class variable str 'name' is not set. "
        assert self.classes, "The class variable list 'classes' is not set."
        _dataset_path = Path(_base_path + self.name)
        if not test and not sample:
            self.path = _dataset_path / 'train'
        elif not test and sample:
            self.path = _dataset_path / 'train_sample'
        elif test and not sample:
            self.path = _dataset_path / 'test'
        else:  # test and sample:
            self.path = _dataset_path / 'test_sample'
        self.data = None
        self.targets = None

    @classmethod
    def create_samples(cls, n_images, ext='png', test=False, delete=True):
        """
        Copies a n_images number of random images from the train or test directory to train_sample or test_sample directory.

        Args:
            n_images: If n_images is an integer then it's the number of images that the sample directory will contain.
                If n_images is a float<1 then it's the fraction of train or test images that the sample directory will contain.
            ext: The extension for the images. Can be *.
            test: If True then it will copy the images from the test directory to the test_sample directory.
                If it's False it will copy the images from the train directory to the train_sample directory.
            delete: If True then the existing train_sample or test_sample directory will be deleted. If False then
                the operation can not be executed when the train_sample or test_sample directory exists.
        """
        _dataset_path = Path(_base_path + cls.name)
        if test:
            origin_path = _dataset_path / 'test'
            destin_path = _dataset_path / 'test_sample'
        else:
            origin_path = _dataset_path / 'train'
            destin_path = _dataset_path / 'train_sample'

        # all_ims = list(origin_path.glob(f'*.{ext}'))
        all_ims = list(origin_path.glob('**/*'))

        if float(n_images).is_integer():
            assert (n_images < len(all_ims)) and (n_images > 0), f"Can't take {n_images} samples from {len(all_ims)} train or test images"
        else:
            assert (n_images < 1) and (n_images > 0), f"Can't take a fraction of {n_images} images. Fraction must be >0 or <1"
            n_images = int(len(all_ims) * n_images)
        sample_imgs = np.random.choice(all_ims, n_images, replace=False)

        if destin_path.exists():
            shutil.rmtree(str(destin_path))
        destin_path.mkdir()
        for f in sample_imgs:
            if f.parts[-2] == 'train':  # MNIST
                shutil.copy(str(f), str(destin_path / f.name))
            else:  # FaschionMNIST
                (destin_path / f.parts[-2]).mkdir(exist_ok=True)
                shutil.copy(str(f), str(destin_path / f.parts[-2] / f.name))

        print(f"Created {n_images} images in {destin_path}")

    def save_csv(self, file):
        """
        Saves the dataset into a csv file with columns [data,targets]
        """
        df = pd.DataFrame({'data': self.data, 'targets': self.targets})
        df.to_csv(file)

    def select_n_random(self, n=100):
        """Selects n random datapoints and their corresponding labels from a dataset"""
        perm = th.randperm(len(self))
        perm = perm[:n]
        imgs = th.stack([self[i][0] for i in perm])
        lbls = [self[i][1] for i in perm]
        return imgs, lbls


class MNIST_Dataset(Standard_Dataset):
    """
    The MNIST (Modified National Institute of Standards and Technology) dataset contains 60.000 train and 10.000 handwritten digits.
    More info: https://en.wikipedia.org/wiki/MNIST_database
    """
    name = 'MNIST'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, sample=False, test=False, transform=None):
        """
           Args:
                 sample (bool): If True then the datasets contains only a limited amount of pictures.
                   If False, the datasets contains all the available images.
                 test: If True then the dataset contains the testimages. If false then the dataset contains the train images.
                 transform: An optional function/transform that takes in an PIL image and returns a transformed version.
                   E.g, ``transforms.RandomCrop``
        """
        super(MNIST_Dataset, self).__init__(sample, test, transform)
        self.data = list(self.path.glob('*.png'))  # list of file paths
        self.targets = [int(i.stem[-1]) for i in self.data]  # image has the format 012345_num9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = io.imread(str(img_path), as_gray=False)  # 28x28
        img = img[:, :, np.newaxis]  # 28x28x1 channel added to work with color models
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class FashionMNIST_Dataset(Standard_Dataset):
    """
    Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples.
    Each example is a 28x28 grayscale image, associated with a label from 10 classes
    More info: https://github.com/zalandoresearch/fashion-mnist
    """
    name = 'FashionMNIST'
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

    def __init__(self, sample=False, test=False, transform=None):
        """
           Args:
                 sample (bool): If True then the datasets contains only a limited amount of pictures.
                   If False, the datasets contains all the available images.
                 test: If True then the dataset contains the testimages. If false then the dataset contains the train images.
                 transform: An optional function/transform that takes in an PIL image and returns a transformed version.
                   E.g, ``transforms.RandomCrop``
        """
        super(FashionMNIST_Dataset, self).__init__(sample, test, transform)
        data = list(self.path.glob('*/*.png'))
        targets = [d.parts[-2] for d in data]
        self.data = list(self.path.glob('*/*.png'))  # list of file paths
        self.targets = [int(d.parts[-2]) for d in self.data]  # image has the format 012345_num9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        img = io.imread(str(img_path), as_gray=False)  # 28x28
        img = img[:, :, np.newaxis]  # 28x28x1 channel added to work with color models
        if self.transform is not None:
            img = self.transform(img)
        return img, target


if __name__ == '__main__':
    fm = MNIST_Dataset()
    # fm = FashionMNIST_Dataset()
    # print(len(fm.data))
    fm.create_samples(100)
