import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import pickle
import PIL.Image as Image
import json
import random
import sys
import clip
import PIL
import random

from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

DECAP_DECODER_CONFIG_PATH = os.path.join(os.getenv("DATA_DIRECTORY_ROOT"), "decap/decoder_config.pkl")
DECAP_COCO_WEIGHTS_PATH = os.path.join(os.getenv("DATA_DIRECTORY_ROOT"), 'decap/coco_model/coco_prefix-009.pt')
        
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 512):
        super(DeCap, self).__init__()
        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open(DECAP_DECODER_CONFIG_PATH,'rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,tokens):
        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()

def Decoding(model,clip_features):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    entry_length = 30
    temperature = 1
    tokens = None
    for i in range(entry_length):
        # print(location_token.shape)
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits, -1)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==49407:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _Tokenizer.decode(output_list)
    except:
        output = 'None'
    return output

decap_model = None

def get_decap_model(device):
    global decap_model
    if decap_model is not None:
        return decap_model
    decap_model = DeCap()
    weights_path = DECAP_COCO_WEIGHTS_PATH
    decap_model.load_state_dict(torch.load(weights_path,map_location= torch.device('cpu')), strict=False)
    decap_model = decap_model.to(device)
    decap_model = decap_model.eval()
    return decap_model