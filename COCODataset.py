"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Load the data from MS COCO dataset.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pycocotools.coco import COCO
from utils import get_transform, get_tokenizer
import os.path as osp


class COCODataset(Dataset):
    def __init__(
            self,
            tokenizer,
            transform,
            mode='train',
            img_path='./data/{}2014',
            ann_path='./data/annotations',
    ):
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.tokenizer = tokenizer

        if mode in ['train', 'val']:
            self.img_path = img_path.format(mode)
            if mode == 'train':
                ann_file = osp.join(ann_path, 'captions_train2014.json')
            else:
                ann_file = osp.join(ann_path, 'captions_val2014.json')
            self.coco = COCO(ann_file)
            self.ids = list(self.coco.anns.keys())
        else:
            self.img_path = img_path.format('val')
            ann_file = osp.join(ann_path, 'captions_val2014.json')
            self.coco = COCO(ann_file)
            self.ids = self.coco.getImgIds()

        # Max length.
        self.max_length = 20
        # Transformation on images.
        self.transform = transform

    def __getitem__(self, item):
        if self.mode in ['train', 'val']:
            ann_id = self.ids[item]
            caption = ' '.join([self.tokenizer.bos_token, self.coco.anns[ann_id]['caption']])
            tokens = self.tokenizer(caption)
            tokens['input_ids'] = tokens['input_ids'][:self.max_length]
            pair = {
                'input_ids': torch.tensor(tokens['input_ids']),
                'labels': torch.tensor(tokens['input_ids']),
                'image_id': self.coco.anns[ann_id]['image_id']
            }
            # Load the image.
            image_id = self.coco.anns[ann_id]['image_id']
            image_path = osp.join(self.img_path, self.coco.imgs[image_id]['file_name'])
            image = Image.open(image_path)
            try:
                image = self.transform(image)
            except:
                # Images with only one channel.
                image = Image.merge('RGB', [image.split()[0]] * 3)
                image = self.transform(image)
            pair['image'] = image
            return pair
        else:
            image_id = self.ids[item]
            image_path = osp.join(self.img_path, self.coco.imgs[image_id]['file_name'])
            image = Image.open(image_path)
            try:
                image = self.transform(image)
            except:
                # Images with only one channel.
                image = Image.merge('RGB', [image.split()[0]] * 3)
                image = self.transform(image)
            return {'image_id': image_id, 'image': image}

    def __len__(self):
        return len(self.ids)


# Pad the caption sequences to the same length.
def collate_fn(batch, padding_value=50256):
    image_id = torch.tensor([b['image_id'] for b in batch])
    image = torch.stack([b['image'] for b in batch])
    batch = {
        k: pad_sequence([batch[i]['input_ids'] for i in range(len(batch))], padding_value=padding_value).T
        for k in ['input_ids', 'labels']
    }
    batch['image'] = image
    batch['image_id'] = image_id
    return batch


def collate_fn_eval(batch):
    image_id = torch.tensor([b['image_id'] for b in batch])
    image = torch.stack([b['image'] for b in batch])
    return {'image_id': image_id, 'image': image}


# Dataloader for training.
def get_train_loader(batch_size=16, transform=None):
    if transform is None:
        transform = get_transform()
    tokenizer = get_tokenizer()
    train_dataset = COCODataset(tokenizer, transform, mode='train')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return train_loader


def get_val_loader(batch_size=16, transform=None):
    if transform is None:
        transform = get_transform()
    tokenizer = get_tokenizer()
    train_dataset = COCODataset(tokenizer, transform, mode='val')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader


# Dataloader for evaluation.
def get_test_loader(transform=None):
    if transform is None:
        transform = get_transform()
    tokenizer = get_tokenizer()
    test_dataset = COCODataset(tokenizer, transform, mode='test')
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_eval
    )
    return test_loader


if __name__ == '__main__':
    train_loader_ = get_train_loader(batch_size=4)
    print(next(iter(train_loader_)))
    val_loader_ = get_val_loader(batch_size=4)
    print(next(iter(val_loader_)))
    test_loader_ = get_test_loader()
    print(next(iter(test_loader_)))
