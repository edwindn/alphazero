import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16
from tqdm import tqdm
import cv2
import os
import argparse


class Utils: # replaces import module
    def __init__(self):
        pass

    def gram_matrix(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, w*h)
        xt = x.transpose(1, 2) # B x F x C
        gram = torch.bmm(x, xt) / c*w*h
        return gram

    def total_variation(self, x):
        return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    def read_image(self, path, height=None, normalise=1):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        
        if not os.path.exists(path):
            raise Exception(f'File not found: {path}')
        img = cv2.imread(path)
        if height is not None:
            w, h = img.shape[:2]
            width = int(w * height // h)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        img = np.array(img).astype(np.float32)
        if normalise == 1:
            img /= 255.0
        elif normalise == 255:
            IMAGENET_MEAN = [i*255 for i in IMAGENET_MEAN]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        img = transform(img).unsqueeze(0)
        return img

    def show_image(self, im, path):
        im = im.detach().cpu().numpy()
        im -= im.min()
        im /= im.max() #normalise to [0, 1]
        im = np.transpose(im, (1, 2, 0))
        plt.imshow(im)
        plt.savefig(path)

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
        
def loss_fn(model, input_img, content_feats, style_feats, content_idx, style_idxs, config):
    input_representation = model(input_img)
    input_content = input_representation[content_idx]
    input_style_representation = [utils.gram_matrix(v) for i, v in enumerate(input_representation) if i in style_idxs]

    content_loss = torch.nn.MSELoss(reduction='mean')(input_content, content_feats[content_idx])

    style_loss = 0.0
    style_feats = [style_feats[i] for i in style_idxs]
    for target, current in zip(style_feats, input_style_representation):
        style_loss += torch.nn.MSELoss(reduction='mean')(current, target)
    style_loss /= len(style_idxs)

    tv_loss = utils.total_variation(input_img)

    tot_loss = config['content_weight']*content_loss + config['style_weight']*style_loss + config['tv_weight']*tv_loss
    #tot_loss = config['content_weight']*content_loss/content_loss.item() + \
    #           config['style_weight']*style_loss/style_loss.item() + config['tv_weight']*tv_loss/tv_loss.item()
    return tot_loss, content_loss, style_loss, tv_loss

def tuning_step(model, optimizer, input_img, content_feats, style_feats, config):
    content_idx = config['content_feature_index']
    style_idxs = config['style_features_indices']
    optimizer.zero_grad()
    tot_loss, content_loss, style_loss, tv_loss = loss_fn(model, input_img, content_feats, style_feats, content_idx, style_idxs, config)
    tot_loss.backward()
    optimizer.step()
    return tot_loss, content_loss, style_loss, tv_loss

def neural_style_transfer(config):
    assert config['optimizer'] in ['adam', 'lbfgs']
    assert config['input_type'] in ['random', 'content']
    assert config['content_feature_index'] in range(5)
    assert all(idx in range(5) for idx in config['style_features_indices'])
    assert config['normalise'] in [1, 255]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img_path = config['content_img']
    style_img_path = config['style_img']
    content = utils.read_image(content_img_path, config['height']).to(device)
    style = utils.read_image(style_img_path, config['height']).to(device)

    #gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
    #input = torch.from_numpy(gaussian_noise_img).float().to(device)
    #input.requires_grad_(True)

    if config['input_type'] == 'random':
        input = torch.randn_like(content, requires_grad=True, device=device) # optimizer will automatically update the input image
    elif config['input_type'] == 'content':
        input = content.clone().requires_grad_(True)
    
    # safe checking
    utils.show_image(input.squeeze(0), 'initial_input.png')
    utils.show_image(content.squeeze(0), 'nst_content.png')
    utils.show_image(style.squeeze(0), 'nst_style.png')

    model = Model().to(device)
    for p in model.parameters():
        p.requires_grad = False

    content_feature_maps = model(content)
    style_feature_maps = model(style)
    style_feature_maps = [utils.gram_matrix(f) for f in style_feature_maps]

    if config['optimizer'] == 'adam':
        input_history = []
        optimizer = torch.optim.Adam((input,), lr=config['lr'])
        for step in tqdm(range(config['num_steps'])):
            tot_loss, content_loss, style_loss, tv_loss = tuning_step(model, optimizer, input, content_feature_maps, style_feature_maps, config)
            if step % 300 == 0:
                input_history.append(input.detach().cpu())
                print(f'step: {step}, tot_loss: {tot_loss.item()}, content_loss: {content_loss.item()}, style_loss: {style_loss.item()}, tv_loss: {tv_loss.item()}')
        
        return input_history

    if config['optimizer'] == 'lbfgs':
        optimizer = torch.optim.LBFGS((input,), max_iter=1000)
        content_idx = config['content_feature_index']
        style_idxs = config['style_features_indices']

        def closure():
            optimizer.zero_grad()
            tot_loss, content_loss, style_loss, tv_loss = loss_fn(model, input_img, content_feats, style_feats, content_idx, style_idxs, config)
            tot_loss.backward()
            return tot_loss
        optimizer.step(closure=closure)
        
        return input.detach().cpu()

if __name__ == '__main__':
    vgg = vgg16(weights='DEFAULT')
    utils = Utils()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalise", type=float, help="normalise to range [0, b]; b = 1 or 255", default='1')
    parser.add_argument("--lr", type=float, default='1e0')
    parser.add_argument("--num_steps", type=int, default='3000')
    parser.add_argument("--content_weight", type=float, default='1e6')
    parser.add_argument("--style_weight", type=float, default='10')
    parser.add_argument("--tv_weight", type=float, default='1')
    parser.add_argument("--optimizer", type=str, help="'adam' or 'lbfgs'", default='adam')
    parser.add_argument("--input_type", type=str, help="'random' or 'content'", default='content')
    parser.add_argument("--content_img", type=str, default='./lion.jpg')
    parser.add_argument("--style_img", type=str, default='./vg_starry_night.jpg')
    args = parser.parse_args()
    
    config = {
        'lr': args.lr,
        'num_steps': args.num_steps,
        'height': 400,
        'content_feature_index': 1,
        'style_features_indices': [0, 1, 2, 3],
        'content_weight': args.content_weight,
        'style_weight': args.style_weight,
        'tv_weight': args.tv_weight,
        'optimizer': args.optimizer,
        'content_img': args.content_img,
        'style_img': args.style_img,
        'input_type': args.input_type,
        'normalise': args.normalise
    }

    result = neural_style_transfer(config)

    if isinstance(result, list):
        for i, r in enumerate(result):
            torch.save(r, f'results/generated_input_{i}.pt')

        files = [f'results/generated_input_{i}.pt' for i in range(10)]
        for i, f in enumerate(files):
            img = torch.load(f)
            img = img.squeeze(0).numpy()
            img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.savefig(f'results/generated_img_{i}.png')

    else:
        torch.save(result, 'results/generated_input.pt')
