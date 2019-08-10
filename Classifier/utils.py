import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
import os
from matplotlib import pyplot as plt


def get_paths(csv_path):
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        paths = [path[0] for path in reader]

    return paths


class TensorCasting(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return img.float()


class HandSegmentationDataset(Dataset):
    def __init__(self, paths, transforms=None):
        images, labels = [], []
        for path in paths:
            img = Image.open(path)
            # img = np.array(img)
            images.append(img)

            path_split, file_name = os.path.split(path)
            labels.append(int(file_name[3]))

        self.images, self.labels = images, labels
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.images[index]
        if self.transforms is not None:
            img = self.transforms(img)

        return (img, self.labels[index])

    def __len__(self):
        return len(self.labels)

if __name__ == "__main__":
    paths = get_paths("paths.csv")
    train_dataset = HandSegmentationDataset(paths,
                                            transforms=transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Resize(256),
                                                transforms.ToTensor(),
                                                TensorCasting(),
                                                transforms.Normalize(mean=[0], std=[65536])]))

    train_params = {
        "pin_memory": True,
        "num_workers": 4,
        "batch_size": 2,
        "shuffle": True,
    }
    train_loader = DataLoader(dataset=train_dataset, **train_params)
    for data, labels in train_loader:
        data = data.numpy()
        data = np.squeeze(data, axis=1)
        for img in data:
            plt.imshow(img, cmap="gray", vmin=0, vmax=max(max(row) for row in img))
            plt.show()
