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
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

run = wandb.init(project='celeba-gan')

BATCH_SIZE = 16
DATASET_LEN = 100000

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
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][:DATASET_LEN]

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
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 16, kernel_size=4, stride=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.ffwd = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            # linear activation since we need the full loss range
        )

    def forward(self, img):
        x = self.cnn(img)
        x = self.ffwd(x)
        return x

        
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(latent_dim, 7*7*64)

        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=2, stride=1, padding=1),
            nn.Sigmoid()
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
        self.val_z = torch.randn(12, latent_dim, device=device)

        self.g_losses = []
        self.d_losses = []

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def forward(self, x):
        x = x.to(device)
        return self.generator(x)

    def log(self, name, value):
        wandb.log({name: value})

    def training_step(self, batch):
        imgs = batch.to(device)
        z = torch.randn(imgs.size(0), self.latent_dim, device=device)

        # ----- train discriminator
        for p in self.discriminator.parameters():
            p.requires_grad = True
        for p in self.generator.parameters():
            p.requires_grad = False

        self.discriminator.zero_grad()
        loss_d_real = self.discriminator(imgs).mean() # to remove batch size

        fake = self.generator(z).detach()
        loss_d_fake = self.discriminator(fake).mean()

        loss_d = - loss_d_real + loss_d_fake
        loss_d.backward()
        self.optimizer_d.step()

        for p in self.discriminator.parameters():
            p.data.clamp_(-0.01, 0.01) # gradient clipping

        # ----- train generator
        for p in self.generator.parameters():
            p.requires_grad = True
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.generator.zero_grad()
        fake = self.generator(z)
        loss_g = - self.discriminator(fake).mean()
        loss_g.backward()
        self.optimizer_g.step()

        self.log("loss_g", loss_g.item())
        self.log("loss_d", loss_d.item())


    def plot_imgs(self, epoch):
        z = self.val_z.to(device).type_as(self.generator.fc.weight)
        sample_imgs = self(z).cpu()

        for i in range(sample_imgs.size(0)):
            plt.subplot(3, 4, i+1)
            plt.tight_layout()
            img = sample_imgs.detach().cpu().numpy()[i, :, :, :]
            img = img.transpose(1, 2, 0)
            plt.imshow(img)
            plt.axis('off')
        plt.savefig(f'gan_images/epoch_{epoch}.png')
        plt.close()

def train(epochs, dataset=custom_dataset, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for batch in tqdm(iter(dataloader)):
            gan.training_step(batch)
        gan.plot_imgs(epoch)
        torch.save(gan.state_dict(), 'gan_weights.pth')

if __name__ == '__main__':
    gan = GAN().to(device)
    train(20, batch_size=32)
