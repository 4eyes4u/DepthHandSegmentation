import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import sys

from utils import *


class LeNet(nn.Module):
    def __init__(self, n_classes):
        super(LeNet, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 58 * 58, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=200),
            # nn.Dropout(0.25, inplace=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=n_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        out = self.convnet(img)
        out = out.view(img.size(0), -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_paths", help="Location of training paths", type=str, required=True)
    parser.add_argument("--test_paths", help="Location of test paths", type=str, required=True)
    parser.add_argument("--n_classes", help="Number of classes", type=int, required=True)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, required=True)
    parser.add_argument("--lr", help="Learning rate", type=float, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    model = LeNet(n_classes=args.n_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_paths = get_paths(args.train_paths)
    train_dataset = HandSegmentationDataset(train_paths,
                                            transforms=transforms.Compose([
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                TensorCasting(),
                                                transforms.Normalize(mean=[0], std=[65536])]))
    train_params = {
        "pin_memory": True,
        "num_workers": 1,
        "batch_size": 32,
        "shuffle": True,
    }
    train_loader = DataLoader(dataset=train_dataset, **train_params)

    test_paths = get_paths(args.test_paths)
    test_dataset = HandSegmentationDataset(test_paths,
                                           transforms=transforms.Compose([
                                               transforms.ToTensor(),
                                               TensorCasting(),
                                               transforms.Normalize(mean=[0], std=[65536])]))
    test_params = {
        "pin_memory": True,
        "num_workers": 1,
        "batch_size": 64,
        "shuffle": False
    }
    test_loader = DataLoader(dataset=test_dataset, **test_params)

    model.eval()
    for epoch in range(1, args.n_epochs + 1):
        epoch_loss = 0
        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            output = model(batch_data)
            loss = F.nll_loss(output, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print("epoch {}, batch {} loss: {}".format(epoch, batch_idx, loss.item()))

        print(f"epoch {epoch} loss: {epoch_loss / len(train_loader)}")

        if epoch % 5 == 0:
            model.eval()
            loss, correct = 0, 0
            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device)
                    labels = labels.to(device)

                    output = model(data)
                    loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
            print(f"epoch {epoch} test acc: {100.0 * correct / len(test_loader.dataset)}, test loss: {loss / len(test_loader)}")
