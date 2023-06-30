"""
Neural Network and Deep Learning, Project 3
Novel Image Captioning
Junyi Liao, 20307110289
Architecture of Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Model
from utils import get_tokenizer
from queue import PriorityQueue


# The node for implementing beam search.
class BeamSearchNode:
    def __init__(self, batch, nll):
        self.input_ids = batch['input_ids']
        # Negative log likelihood.
        self.nll = nll

    def __len__(self):
        return self.input_ids.size(-1)

    # Override the comparison method.
    def __lt__(self, other):
        return self.nll < other.nll


# Attention model.
class Attention(nn.Module):
    def __init__(self, encode_size, decode_size, att_size):
        """
        :param encode_size: the dimension of encoded images.
        :param decode_size: the dimension of decoder output.
        :param att_size: the dimension of hidden layer in attention model.
        """
        super(Attention, self).__init__()
        self.encode_size = encode_size
        self.decode_size = decode_size
        self.att_size = att_size
        # Transform the encoded image.
        self.encode2att = nn.Linear(encode_size, att_size)
        # Transform the decoded image.
        self.decode2att = nn.Linear(decode_size, att_size)
        # Full attention.
        self.alphaNet = nn.Linear(att_size, 1)

    def forward(self, enc, dec):
        """
        :param enc: the encoded images, of shape (batch_size, num_pixels, encode_size).
        :param dec: the hidden states output by the decoder, of shape (batch_size, decode_size).
        :return: attention weighted encoding and weights.
        """
        enc = enc.flatten(-2).permute(0, 2, 1)  # batch_size * num_pixels * encode_size.
        att_enc = self.encode2att(enc)  # batch_size * num_pixels * att_size.
        att_dec = self.decode2att(dec)  # batch_size * att_size.
        # Broadcast Addition.
        dot = att_enc.unsqueeze(1) + att_dec.unsqueeze(2)
        dot = self.alphaNet(F.relu(dot)).squeeze(-1)
        alpha = F.softmax(dot, dim=-1)
        att_res = (att_enc.unsqueeze(1) * alpha.unsqueeze(-1)).sum(dim=-2)
        return att_res, alpha


# The caption generator.
class Captioner(nn.Module):
    def __init__(self, encode_size=7):
        super(Captioner, self).__init__()
        self.encode_size = encode_size
        # Encoder.
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Decoder.
        self.transformers = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = get_tokenizer()
        # Attention Model.
        self.attention = Attention(2048, 768, 768)
        self.alpha = None

    def forward(self, image, input_ids, labels=None, **others):
        """
        :param image:
        :param input_ids:
        :param labels:
        """
        encoded_features = self.resnet(image)
        encoded_features = F.interpolate(
            encoded_features,
            (self.encode_size, self.encode_size),
            mode='bilinear'
        )
        embeddings = self.transformers.transformer.wte(input_ids)
        att_res, self.alpha = self.attention(encoded_features, embeddings)
        # Broadcasting addition.
        embeddings = att_res + embeddings
        out = self.transformers(labels=labels, inputs_embeds=embeddings)
        return out

    # Generate captions of images.
    def beam_decode(self, batch, n_sentences=1, max_length=20, beam_width=5, **others):
        """
        :param batch: in evaluation mode, {'image_id': image_id, 'image': image} with batch size 1.
        :param n_sentences: the number of sentences to generate.
        :param max_length: the maximum length of generated sentences.
        :param beam_width: the width of beam search. When set as 1, it is the same strategy as greedy decoding.
        :return: sentences.
        """
        # The batch size need to be 1.
        assert batch['image_id'].size(0) == 1, 'The batch size should be 1 in evaluation mode.'
        # Get the device.
        device = next(iter(self.resnet.parameters())).device
        # Get the tokens to stop when encountered.
        ending_tokens = ['.', '?', '!', '\n', '\n\n', self.tokenizer.eos_token]
        endings = self.tokenizer.convert_tokens_to_ids(ending_tokens)

        primary_batch = {
            'input_ids': torch.tensor(50256).reshape(-1, 1).to(device),
            'image':  batch['image'],
            'image_id': batch['image_id'],
        }
        sent_tokens = []
        with torch.no_grad():
            nodes = PriorityQueue()
            primary_node = BeamSearchNode(primary_batch, 0)
            candidates = [primary_node]
            nodes.put((primary_node.nll, primary_node))
            while len(sent_tokens) < n_sentences:
                pq = PriorityQueue()
                for node in candidates:
                    input_batch = {
                        'input_ids': node.input_ids,
                        'image':  batch['image'],
                        'image_id': batch['image_id'],
                    }
                    # Save the sentence if it reaches the end. Note that endings include the start label.
                    if len(node) >= max_length or (node.input_ids[0][-1] in endings and len(node) > 1):
                        sent_tokens.append(input_batch['input_ids'][0][1:].tolist())

                    outputs = self(**input_batch)
                    # Beam Search.
                    next_token_logits = outputs.logits[:, -1, :]
                    beam = next_token_logits.topk(beam_width)[1][0]
                    prob = F.softmax(next_token_logits, dim=-1)
                    for next_token in beam:
                        next_batch = {
                            'input_ids': torch.cat(
                                [input_batch['input_ids'], next_token.reshape(-1, 1)], dim=-1
                            )
                        }
                        next_node = BeamSearchNode(next_batch, node.nll - torch.log(prob[0, next_token]).item())
                        pq.put(next_node)
                # Get the top k candidates.
                candidates = []
                for _ in range(beam_width):
                    candidates.append(pq.get())

        sent = self.tokenizer.batch_decode(sent_tokens, skip_special_tokens=True)
        return sent
