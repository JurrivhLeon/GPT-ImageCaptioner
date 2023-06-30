"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Visualization Module.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from COCODataset import COCODataset
from network import Captioner
from utils import get_tokenizer, get_transform
import os
import os.path as osp


def plot_loss(loss_path):
    losses = np.load(loss_path)
    losses = losses[:32500].reshape(325, 100)
    plt.figure(figsize=(10, 10))
    plt.plot(losses.mean(axis=1), color='dodgerblue')
    plt.fill_between(np.arange(losses.shape[0]), np.quantile(losses, 0.975, axis=1),
                     np.quantile(losses, 0.025, axis=1), color='salmon', alpha=0.35)
    plt.xlabel('Iteration (Step)')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, losses.shape[0] + 25, step=25))
    plt.yticks(np.arange(0, 7.5, 0.5))
    plt.title('Training Loss of Captioner')
    plt.grid()
    # plt.show()
    plt.savefig(f"models/loss.png")
    plt.close()


def plot_caption(ckpt_path, indices, beam_width=5):
    strategy = 'beam' if beam_width > 1 else 'greedy'
    save_dir = osp.join('test_examples', strategy)
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    model = Captioner(encode_size=14).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # Load the dataset.
    transform = get_transform()
    tokenizer = get_tokenizer()
    test_dataset = COCODataset(tokenizer, transform, mode='test')
    mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).to(device)
    std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).to(device)
    for idx in indices:
        sample = test_dataset[idx]
        sample = {'image_id': torch.tensor(sample['image_id']).reshape(1, 1).to(device),
                  'image': sample['image'].unsqueeze(0).to(device)}
        caption = model.beam_decode(sample, beam_width=beam_width)[0]
        img_show = sample['image'].squeeze(0) * std + mean
        img_show = img_show.cpu().moveaxis(0, -1).numpy()
        plt.figure()
        plt.imshow(img_show)
        plt.axis('off')
        plt.title(caption)
        plt.savefig(osp.join(save_dir, f"{sample['image_id'][0][0].item()}.png"))
        plt.close()


if __name__ == '__main__':
    plot_loss('models/training_loss.npy')
    random.seed(2023)
    indices = random.sample(range(20000), 100)
    plot_caption('models/captioner_best.pth', indices=range(100), beam_width=1)
    plot_caption('models/captioner_best.pth', indices=indices, beam_width=1)
    plot_caption('models/captioner_best.pth', indices=range(100), beam_width=5)
    plot_caption('models/captioner_best.pth', indices=indices, beam_width=5)
