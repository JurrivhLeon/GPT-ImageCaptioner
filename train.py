"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Train a network for image captioning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from COCODataset import get_train_loader, get_val_loader
from network import Captioner
from utils import adjust_learning_rate
from tqdm import tqdm
import os
import os.path as osp
import numpy as np


# Train the model within one epoch.
def train(model, train_loader, optimizer, epoch, device):
    model.train()
    # Adjust th learning rate.
    adjust_learning_rate(optimizer, epoch, num_epochs)
    losses = []
    prog_bar = tqdm(train_loader, total=len(train_loader))
    for it, batch in enumerate(prog_bar):
        # Convert to the working device.
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward.
        output = model(**batch)
        loss = output.loss
        losses.append(loss.item())
        # Backward.
        optimizer.zero_grad()
        loss.backward()
        # Clip the gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # Update.
        optimizer.step()
        prog_bar.set_description(desc='Epoch {}, loss {:.3f}'.format(epoch, loss.item()))
    return losses


# Validation in each epoch.
def val(model, val_loader, epoch):
    # Evaluation.
    losses = []
    prog_bar = tqdm(val_loader, total=len(val_loader))
    with torch.no_grad():
        for it, batch in enumerate(prog_bar):
            # Convert to the working device.
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward and compute loss.
            output = model(**batch)
            loss = output.loss
            losses.append(loss.item())
            prog_bar.set_description(desc='Epoch {}, loss {:.3f}'.format(epoch, loss.item()))
    return losses


if __name__ == '__main__':
    # Setting the hyperparameters.
    batch_size = 32  # batch size.
    num_epochs = 2  # number of training epochs.
    lr = 1e-3  # learning rate.
    save_every = 1  # determines frequency of saving model weights.
    data_dir = r"./data"  # directory of dataset.
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Prepare the data.
    train_dataloader = get_train_loader(batch_size=batch_size)
    val_dataloader = get_val_loader(batch_size=batch_size)

    # Model and Optimizer.
    captioner = Captioner(encode_size=14).to(device)
    param_list = [
        {'params': captioner.resnet.parameters(), 'lr': lr / 100},
        {'params': captioner.transformers.transformer.parameters(), 'lr': lr / 100},
        {'params': captioner.attention.parameters(), 'lr': lr},
    ]
    optimizer = optim.Adam(param_list, lr=lr, weight_decay=5e-5)
    losses_train = []
    best_val_loss = float('Inf')

    for epoch in range(num_epochs):
        losses_epoch = train(captioner, train_dataloader, optimizer, epoch, device)
        losses_train.extend(losses_epoch)
        # Save the weights.
        if epoch % save_every == 0:
            torch.save({'epoch': epoch, 'state_dict': captioner.state_dict()},
                       osp.join('./models', 'captioner_%d.pth' % epoch))

        # Validation.
        losses_val = val(captioner, val_dataloader, epoch)
        val_loss = sum(losses_val) / len(losses_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'state_dict': captioner.state_dict()},
                       osp.join('./models', 'captioner_best.pth'))
        print(f'best_val_loss: {best_val_loss}')

    # np.save('./models/training_loss.npy', np.array(losses_train))
