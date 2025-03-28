import logging
import numpy as np

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src import config


class MnistDataset:
    def __init__(
        self,
    ):
        self.transform = self.__transform()

    def __transform(self):
        return Compose(
            [
                Resize(config.IMG_SIZE),
                ToTensor(),
            ]
        )

    def get_datasets(self):
        train_data = MNIST(
            root=config.DATA_PATH, train=True, download=True, transform=self.transform
        )
        test_data = MNIST(
            root=config.DATA_PATH,
            train=False,
            download=True,
            transform=self.transform,
        )
        class_names = train_data.classes
        return train_data, test_data, class_names

    def data_loader(self, train_data, test_data, valid_size=0.2):
        train_length = len(train_data)

        # obtain training dataset indices that
        # will be used for validation dataset
        indices = list(range(train_length))

        np.random.shuffle(indices)
        split = int(np.floor(valid_size * train_length))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # prepare data loaders for train, test and validation dataset
        train_loader = DataLoader(
            train_data,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
        )
        valid_loader = DataLoader(
            train_data,
            batch_size=config.BATCH_SIZE,
            sampler=valid_sampler,
            num_workers=config.NUM_WORKERS,
        )
        test_loader = DataLoader(
            test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
        )
        logging.info(
            f"Train data size : {train_idx.__len__()}, Validation data size : {valid_idx.__len__()}, Test data size : {test_loader.dataset.__len__()}"
        )

        return train_loader, valid_loader, test_loader


class MnistEDA:
    @staticmethod
    def plot_class_distribution_and_pie_chart(dataset, class_names):
        labels = [label for _, label in dataset]
        class_counts = Counter(labels)
        counts = [class_counts[i] for i in range(len(class_names))]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Bar Plot
        sns.barplot(
            x=class_names,
            y=counts,
            ax=axes[0],
            hue=class_names,
            legend=False,
            palette="viridis",
        )
        axes[0].set_xticklabels(class_names, rotation=45)
        axes[0].set_title("Class Distribution in MNIST")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        # Pie Chart
        axes[1].pie(
            counts,
            labels=class_names,
            autopct="%1.1f%%",
            colors=sns.color_palette("viridis", len(class_names)),
        )
        axes[1].set_title("Class Distribution in MNIST")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_images(data, class_names, num_imgs=4):
        fig, axes = plt.subplots(1, num_imgs, figsize=(15, num_imgs))

        for i in range(num_imgs):
            img, label = data[i]
            img = img.permute(1, 2, 0).numpy()

            img = (img - img.min()) / (img.max() - img.min())

            axes[i].imshow(img)
            axes[i].set_title(class_names[label])
            axes[i].axis("off")

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_batch_images(train_loader):
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        fig = plt.figure(figsize=(30, 10))
        for i in range(len(labels)):
            ax = fig.add_subplot(2, config.BATCH_SIZE, i + 1, xticks=[], yticks=[])
            plt.imshow(np.squeeze(images[i]))
            ax.set_title(labels[i].item(), color="blue")
        return fig
