import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

# This code was stolen from
# https://pytorch.org/tutorials/beginner/nn_tutorial.html#refactor-using-dataloader
# And hopfully works
# But it prob needs a few hours of fixing xD ups
# mein Gehirn ist iwann auf standby gegangen und hat nur noch copy-paste xD

class LidarCnn():
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            Lambda(lambda x: x.view(x.size(0), -1)),
        )

    @staticmethod
    def loss_batch(model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)

        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()

        return loss.item(), len(xb)

    @staticmethod
    def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_dl:
                LidarCnn.loss_batch(model, loss_func, xb, yb, opt)

            model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[LidarCnn.loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            print(epoch, val_loss)

    @staticmethod
    def get_data(train_ds, valid_ds, bs):
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2)
        )

    @staticmethod
    def preprocess(x, y):
        return x.view(-1, 1, 28, 28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def main(args=None):
    train_dl, valid_dl = LidarCnn.get_data(train_ds, valid_ds, bs= 64)
    train_dl = WrappedDataLoader(train_dl, LidarCnn.preprocess)
    valid_dl = WrappedDataLoader(valid_dl, LidarCnn.preprocess)

    model = LidarCnn()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    LidarCnn.fit(epochs=1, model=model, loss_func=, opt=opt, train_dl=, valid_dl=)
