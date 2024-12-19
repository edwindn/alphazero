import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import asyncio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

from datasets import load_dataset
ds = load_dataset("Prasanna18/Human-Face_Images_for_Emotion_Recognition")['train']

from torchvision.transforms import ToTensor
import torch

def preprocess(example):
    transform = ToTensor()
    tensor_image = transform(example["image"]).float()  # Transform directly to tensor
    return {"tensor_image": tensor_image}

ds = ds.map(preprocess, remove_columns=["image", "label"], batched=False)
ds.set_format(type="torch", columns=["tensor_image"])

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, img_size=224):
        super(Discriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(20, 20, kernel_size=5, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(20, 20, kernel_size=5, stride=2),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

        self.ffwd = nn.Sequential(
            nn.Linear(500, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        x = self.cnn(img)
        x = x.view(-1, 500)
        x = self.ffwd(x)
        return x

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(latent_dim, 7*7*64)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=2, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 7, 7)
        x = self.conv_blocks(x)
        return x

class GAN(nn.Module):
    def __init__(self, latent_dim=100, lr=2e-4):
        super().__init__()
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)

        self.loss_fn = nn.BCELoss().to(device)
        self.latent_dim = latent_dim
        self.val_z = torch.randn(6, latent_dim, device=device)

        self.g_losses = []
        self.d_losses = []

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def forward(self, x):
        x = x.to(device)
        return self.generator(x)

    def log(self, name, value):
        if name == 'loss_g':
            self.g_losses.append(value)
        elif name == 'loss_d':
            self.d_losses.append(value)

    def training_step(self, batch):
        imgs = batch['tensor_image'].to(device)

        z = torch.randn(imgs.size(0), self.latent_dim, device=device)

        # Train generator
        self.optimizer_g.zero_grad()
        fake_imgs = self(z)
        disc_pred = self.discriminator(fake_imgs)
        true = torch.ones(imgs.size(0), 1, device=device)
        loss_g = self.loss_fn(disc_pred, true)
        loss_g.backward()
        self.optimizer_g.step()

        self.log("loss_g", loss_g)

        # Train discriminator
        self.optimizer_d.zero_grad()
        pred_real = self.discriminator(imgs)
        true = torch.ones(imgs.size(0), 1, device=device)
        real_loss = self.loss_fn(pred_real, true)

        pred_fake = self.discriminator(fake_imgs.detach())
        fake = torch.zeros(imgs.size(0), 1, device=device)
        fake_loss = self.loss_fn(pred_fake, fake)

        loss_d = (real_loss + fake_loss) / 2
        loss_d.backward()
        self.optimizer_d.step()

        self.log("loss_d", loss_d)

        return loss_g.item(), loss_d.item()

    def plot_imgs(self, epoch):
        z = self.val_z.to(device).type_as(self.generator.fc.weight)
        sample_imgs = self(z).cpu()

        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
        plt.savefig(f'epoch_{epoch}.png')
        plt.close()

def train(epochs, dataset, batch_size):
    gan = GAN().to(device)
    dl_train = DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for batch in tqdm(iter(dl_train)):
            gan.training_step(batch)
        torch.save(gan.state_dict(), f'gan_weights_{epoch}.pth')
        gan.plot_imgs(epoch)

if __name__ == '__main__':
    train(1000, ds, 16)
