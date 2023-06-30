"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Load the data from MS COCO dataset.
"""

import numpy as np
import torch
import os
import random
import torchvision.transforms as T
from transformers import GPT2Tokenizer


# Reproducibility guarantee.
def set_random_seeds(seed=0, device='cuda:0'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_transform():
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)),
    ])
    return transform


def get_tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = tokenizer.eos_token
    tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


# Adjust learning rate in each epoch.
def adjust_learning_rate(optimizer, epoch, num_epochs):
    if epoch >= num_epochs / 2:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
