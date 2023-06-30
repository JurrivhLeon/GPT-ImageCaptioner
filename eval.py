"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Evaluation on MS COCO Dataset.
"""

import torch
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from COCODataset import get_test_loader
from network import Captioner
from utils import get_transform
import os
import os.path as osp
from tqdm import tqdm
import json
import pandas as pd
import pickle
from datetime import datetime


# Generate captions with the model.
def generateCaptions(model, data_loader, beam_width=5):
    """
    :param model: ~
    :param data_loader: ~
    :param beam_width: Width of beam search. When set as 1, it is equivalent to greedy search.
    :return:
    """
    model.eval()
    device = next(iter(model.resnet.parameters())).device
    predictions = []
    # initialize tqdm progress bar.
    prog_bar = tqdm(data_loader, total=len(data_loader))
    for idx, batch in enumerate(prog_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        sent = model.beam_decode(batch, beam_width=beam_width)[0]
        entry = {'image_id': batch['image_id'][0].item(), 'caption': sent}
        predictions.append(entry)
        prog_bar.set_description(desc=f"{sent}")
    return predictions


# Evaluation on the generated captions.
def languageEval(preds, annFile, model_id, split, res_dir):
    os.makedirs(res_dir, exist_ok=True)
    cache_path = osp.join(res_dir, str(model_id) + '_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MS COCO validation set (will be about a third)
    preds_filter = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filter), len(preds)))
    with open(cache_path, 'w') as temp:
        json.dump(preds_filter, temp)  # serialize to temporary json file

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['import'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    print('-' * 5, 'Language evaluation', '-' * 5)
    for metric, score in cocoEval.eval.items():
        out[metric] = score
        print(f'{metric}: {score}')
    print('-' * 25)

    imgToEval = cocoEval.imgToEval
    for p in preds_filter:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


if __name__ == '__main__':
    # Evaluation on Validation Set.
    transform_test = get_transform()
    model_dir = r'./models/captioner_best.pth'
    data_dir = r'./data'  # directory of dataset.
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Create the data loader.dd
    test_data_loader = get_test_loader()
    # Choose device.
    device_ = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # The size of the vocabulary.
    captioner = Captioner(encode_size=14).to(device_)
    model_state = torch.load(model_dir)['state_dict']
    captioner.load_state_dict(model_state)
    # Prediction.
    preds_ = generateCaptions(captioner, test_data_loader, beam_width=5)
    test_ann_file = osp.join(data_dir, 'annotations/captions_val2014.json')
    # Language evaluation.
    res_dir_ = './eval_results/%s' % str(datetime.now())[:19].replace(' ', '').replace('-', '').replace(':', '')
    output = languageEval(preds_, test_ann_file, '1', 'gen_test2014', res_dir=res_dir_)
    
