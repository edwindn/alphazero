import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from einops import rearrange
import wandb

run = wandb.init(project='diffusion-models')

BATCH_SIZE = 64
NUM_TIMESTEPS = 1000
IMAGE_SIZE = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class NoiseScheduler:
    def __init__(self, beta_start, beta_end, device, num_timesteps=NUM_TIMESTEPS):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
    
    def add_noise(self, x, t, noise):
        try:
            return x * torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1, 1) + noise * torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1, 1, 1)
        except Exception as e:
            print(f'ERROR IN NOISE SCHEDULER: {e}')
            return x
        
class ResBlock(nn.Module):
    def __init__(self, num_channels, num_groups):
        super(ResBlock, self).__init__()

        self.main = nn.Sequential(
            nn.GroupNorm(num_groups, num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            nn.Dropout(0.1),
            nn.GroupNorm(num_groups, num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, 3, padding=1)
        )

    def forward(self, x, t_emb):
        _, _, h, w = x.shape
        t_emb = t_emb.expand(-1, -1, h, w)
        #print('resblock:', x.shape, t_emb.shape)
        x = x + t_emb
        r = self.main(x)
        return r + x

class Attention(nn.Module):
    def __init__(self, num_channels, num_heads):
        super(Attention, self).__init__()

        self.l1 = nn.Linear(num_channels, num_channels*3)
        self.l2 = nn.Linear(num_channels, num_channels)
        self.num_heads = num_heads

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c') # make into 1d time series, length L = h*w
        x = self.l1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.1)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.l2(x)
        x = rearrange(x, 'b h w C -> b C h w')
        return x

class UBlock(nn.Module):
    # 2 resblocks, attention & up/down sampling
    def __init__(self, upscale, attention, num_channels, num_groups, num_heads, first_up=False):
        super(UBlock, self).__init__()

        self.rb1 = ResBlock(num_channels, num_groups)
        self.rb2 = ResBlock(num_channels, num_groups)
        self.attention = attention
        
        if upscale:
            if first_up:
                self.conv = nn.ConvTranspose2d(num_channels, num_channels, 3, 2, 1)
            else:
                self.conv = nn.ConvTranspose2d(num_channels, num_channels//2, 4, 2, 1)
        else:
            self.conv = nn.Conv2d(num_channels, num_channels*2, 3, 2, 1)
        
        if attention:
            self.attention_block = Attention(num_channels, num_heads)

    def forward(self, x, t_emb):
        x = self.rb1(x, t_emb)
        #print('ublock:', x.shape)
        if self.attention:
            x = self.attention_block(x)
        x = self.rb2(x, t_emb)
        z = self.conv(x)
        #print('ublock:', z.shape)
        return z, x # return x for skip connections

class UNet(nn.Module): # predicts the noise to remove given the current image and its timestep embedding
    def __init__(self, device, input_channels=1, output_channels=1, num_groups=32, num_heads=4, num_timesteps=NUM_TIMESTEPS):
        super(UNet, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1).to(device)

        self.dblock1 = UBlock(False, False, 64, num_groups, num_heads).to(device)
        self.dblock2 = UBlock(False, True, 128, num_groups, num_heads).to(device)
        self.dblock3 = UBlock(False, False, 256, num_groups, num_heads).to(device)
        self.midblock = nn.Conv2d(512, 256, 3, padding=1).to(device)
        self.ublock1 = UBlock(True, False, 256, num_groups, num_heads, first_up=True).to(device)
        self.ublock2 = UBlock(True, False, 256, num_groups, num_heads).to(device)
        self.ublock3 = UBlock(True, True, 128, num_groups, num_heads).to(device)

        self.dblocks = [self.dblock1, self.dblock2, self.dblock3]
        self.upblocks = [self.ublock1, self.ublock2, self.ublock3]

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, output_channels*2, 1),
            nn.ReLU(),
            nn.Conv2d(output_channels*2, output_channels, 1)
        ).to(device)

        self.embeddings = {
            64: self.get_embeddings(num_timesteps, 64).to(device),
            128: self.get_embeddings(num_timesteps, 128).to(device),
            256: self.get_embeddings(num_timesteps, 256).to(device),
        }
    
    def get_embeddings(self, num_timesteps, emb_dim):
        pos = torch.arange(num_timesteps).unsqueeze(1)
        div = torch.exp(-1/emb_dim * torch.arange(0, emb_dim, 2) * math.log(10000))
        embeddings = torch.zeros(num_timesteps, emb_dim, requires_grad=False)
        embeddings[:, ::2] = torch.sin(pos * div)
        embeddings[:, 1::2] = torch.cos(pos * div)
        return embeddings
    
    def embed(self, t):
        t = t.to(device)
        return [self.embeddings[c][t][:, :, None, None].to(self.device) for c in self.embeddings.keys()]

    def forward(self, x, t):
        x = x.to(self.device)
        x = self.conv1(x)
        residuals = []
        t_embs = self.embed(t)
        
        for idx, b in enumerate(self.dblocks):
            x, r = b(x, t_embs[idx])
            residuals.append(r)
        x = self.midblock(x)

        r = residuals[-1]
        x = self.ublock1(x, t_embs[-1])[0] + r

        r = residuals[-2]
        x = self.ublock2(x, t_embs[-1])[0] + r

        r = residuals[-3]
        x = self.ublock3(x, t_embs[-2])[0] + r
        return self.out_conv(x)
    
def train(num_epochs, dataloader, dataloader_eval, batch_size=BATCH_SIZE):
    noise_scheduler = NoiseScheduler(0.0001, 0.02, device=device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        model.train()
        train_loss = 0

        for batch in tqdm(dataloader):
            t = torch.randint(0, NUM_TIMESTEPS, (batch_size,)).to(device)
            x, _ = batch
            x = x.to(device)
            noise = torch.randn_like(x, requires_grad=False).to(device)
            x = noise_scheduler.add_noise(x, t, noise).to(device)

            model.zero_grad()
            pred_noise = model(x, t)
            loss = loss_fn(noise, pred_noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dataloader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(dataloader_eval):
                t = torch.randint(0, NUM_TIMESTEPS, (batch_size,)).to(device)
                x, _ = batch
                x = x.to(device)
                noise = torch.randn_like(x, requires_grad=False).to(device)
                noised_x = noise_scheduler.add_noise(x, t, noise)

                pred_noise = model(noised_x, t)
                loss = loss_fn(noise, pred_noise)
                val_loss += loss.item()

            val_loss /= len(dataloader_eval)

        wandb.log({"training loss": train_loss, "eval loss": val_loss})
        torch.save(model.state_dict(), 'ddpm_weights.pth')

if __name__ == '__main__':
    mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_eval = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    dataloader_eval = DataLoader(mnist_eval, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model = UNet(device).to(device)
    model.load_state_dict(torch.load('ddpm_weights.pth'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss(reduction='mean')
    train(num_epochs=20, dataloader=dataloader, dataloader_eval=dataloader_eval)
