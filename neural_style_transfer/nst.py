import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
vgg = vgg16(pretrained=True)

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
for i, im in enumerate(outs):
    im = im.detach().cpu().numpy()
    plt.imshow(im)
    plt.savefig(f'vgg_output_{i}.png')

