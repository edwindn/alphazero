import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm

vgg = vgg16(weights='DEFAULT')

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)

        self.block1 = nn.ModuleList(vgg.features[:4]) + [avg_pool]
        self.block2 = nn.ModuleList(vgg.features[5:9]) + [avg_pool]
        self.block3 = nn.ModuleList(vgg.features[10:16]) + [avg_pool]
        self.block4 = nn.ModuleList(vgg.features[17:23]) + [avg_pool]
        self.block5 = nn.ModuleList(vgg.features[24:30]) + [avg_pool]

        self.block1 = nn.Sequential(*self.block1)
        self.block2 = nn.Sequential(*self.block2)
        self.block3 = nn.Sequential(*self.block3)
        self.block4 = nn.Sequential(*self.block4)
        self.block5 = nn.Sequential(*self.block5)

    def forward(self, x):
        emb1 = self.block1(x)
        emb2 = self.block2(emb1)
        emb3 = self.block3(emb2)
        emb4 = self.block4(emb3)
        emb5 = self.block5(emb4)
        return emb1, emb2, emb3, emb4, emb5

f = 'test_img.jpg'
img = Image.open(f)
transform = transforms.ToTensor()
img = transform(img)

model = Model()

outs = model(img)
print([v.shape for v in outs])

img_out = model(img)
target_feature = 2
target = img_out[target_feature]

white_noise = torch.randn_like(img, requires_grad=True)
optimizer = torch.optim.Adam((white_noise,), lr=1e1)
for i in tqdm(range(1000)):
    out = model(white_noise)[target_feature]
    loss = torch.nn.MSELoss(reduction='mean')(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    white_noise = white_noise.detach().clone().requires_grad_(True)
    if (i+1) % 100 == 0:
        torch.save(white_noise.detach().cpu(), f'generated_input_{i}.pt')
        
quit()

for i, im in enumerate(outs):
    torch.save(im, f'vgg_output_{i}.pt')
    continue
    im = im.detach().cpu().numpy()
    plt.imshow(im)
    plt.savefig(f'vgg_output_{i}.png')

