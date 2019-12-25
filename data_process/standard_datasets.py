# --------------------------------------------------------------------------------------------------------
# 2019/12/24
# src - standard_datasets.py
# md
# --------------------------------------------------------------------------------------------------------

"""
A collection of Pytorch Dataset classes for standard datasets.

All datasets are subclasses of torch.utils.data_process.Dataset i.e, they have __getitem__ and __len__ methods implemented.
Hence, they can all be passed to a torch.utils.data_process.DataLoader which can load multiple samples parallelly using torch.multiprocessing workers.

All datasets have the classes attribute which is a list of all classes.

All datasets implement the create_samples() method to create a samples subset from the train and test dataset, located in the same directory.

The following datasets have currently been implemented:
- MNIST

"""

import shutil
from pathlib import Path
from numpy import random
from skimage import io
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

__all__ = ['MNIST_Dataset']
_base_path = '/media/md/Development/0_Datasets/0_standard_datasets/'


class MNIST_Dataset(Dataset):
    """
    The MNIST (Modified National Institute of Standards and Technology) dataset contains 60.000 train and 10.000 handwritten digits.
    More info: https://en.wikipedia.org/wiki/MNIST_database
    """
    _name = 'MNIST'
    _dataset_path = Path(_base_path + _name)
    _train_path = _dataset_path / 'train'
    _train_sample_path = _dataset_path / 'train_sample'
    _test_path = _dataset_path / 'test'
    _test_sample_path = _dataset_path / 'test_sample'
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
        super(MNIST_Dataset, self).__init__()
        path = Path()
        if test and sample:
            path = self._test_sample_path
        elif test and not sample:
            path = self._test_path
        elif not test and sample:
            path = self._train_sample_path
        elif not test and not sample:
            path = self._train_path

        self.transform = transform
        self.data = list(path.glob('*.png'))  # list of file paths
        self.targets = [int(i.stem[-1]) for i in self.data]  # image has the format 012345_num9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns:
            tuple(image, target) where image is a transformed PIL image (numpy array) and target is the index of the target class
        """
        img_path, target = self.data[index], self.targets[index]
        img = io.imread(str(img_path), as_gray=True)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    @classmethod
    def create_samples(cls, n_images, test=False, delete=True):
        """
        Copies a n_images number of random images from the train or test directory to train_sample or test_sample directory.

        Args:
            n_images: If n_images is an integer then it's the number of images that the sample directory will contain.
                If n_images is a float<1 then it's the fraction of train or test images that the sample directory will contain.
            test: If True then it will copy the images from the test directory to the test_sample directory.
                If it's False it will copy the images from the train directory to the train_sample directory.
            delete: If True then the existing train_sample or test_sample directory will be deleted. If False then
                the operation can not be executed when the train_sample or test_sample directory exists.
        """
        if test:
            origin_path = cls._test_path
            destin_path = cls._test_sample_path
        else:
            origin_path = cls._train_path
            destin_path = cls._train_sample_path

        all_ims = list(origin_path.glob('*.png'))

        if float(n_images).is_integer():
            assert (n_images < len(all_ims)) and (n_images > 0), \
                f"Can't take {n_images} samples from {len(all_ims)} train or test images"
        else:
            assert (n_images < 1) and (n_images > 0), \
                f"Can't take a fraction of {n_images} images. Fraction must be >0 or <1"
            n_images = int(len(all_ims) * n_images)
        sample_imgs = random.choice(all_ims, n_images, replace=False)
        if destin_path.exists():
            shutil.rmtree(str(destin_path))
        destin_path.mkdir()
        for f in sample_imgs:
            shutil.copy(str(f), str(destin_path / f.name))
        print(f"Created {n_images} images in {destin_path}")


if __name__ == '__main__':
    pass
