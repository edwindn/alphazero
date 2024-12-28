import torch
import numpy as np
import matplotlib.pyplot as plt

files = [f'generated_input_{i}.pt' for i in range(10)]
for i, f in enumerate(files):
    img = torch.load(f)
    img = img.squeeze(0).numpy()
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)
    plt.savefig(f'results/generated_img_{i}.png')
  
