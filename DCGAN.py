import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor, Compose, Resize
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import psutil  # Import psutil for memory usage tracking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
print(len(files))
quit()

BATCH_SIZE = 64

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

# Function to get current memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Returns memory usage in MB

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

custom_dataset = ImageDataset('../img_align_celeba', transform=transform)

class Discriminator(nn.Module):
    def __init__(self, input_dim=1, img_size=224):
        super(Discriminator, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
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
            nn.Conv2d(128, 3, kernel_size=2, stride=1, padding=1)
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
        print(f"Memory usage before training step: {get_memory_usage()} MB")  # Memory usage before training step
        imgs = batch.to(device)

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

        print(f"Memory usage after training step: {get_memory_usage()} MB")  # Memory usage after training step

        return loss_g.item(), loss_d.item()

    def plot_imgs(self, epoch):
        print(f"Memory usage before plotting images: {get_memory_usage()} MB")  # Memory usage before plotting
        z = self.val_z.to(device).type_as(self.generator.fc.weight)
        sample_imgs = self(z).cpu()

        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.axis('off')
        plt.show()
        plt.savefig(f'epoch_{epoch}.png')
        plt.close()
        print(f"Memory usage after plotting images: {get_memory_usage()} MB")  # Memory usage after plotting

def train(epochs, dataset, batch_size=BATCH_SIZE):
    gan = GAN().to(device)
    dataloader = DataLoader(custom_dataset, batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for batch in tqdm(iter(dataloader)):
            gan.training_step(batch)
        torch.save(gan.state_dict(), f'gan_epoch_{epoch}.pth')
        gan.plot_imgs(epoch)

if __name__ == '__main__':
    train(10, custom_dataset)
