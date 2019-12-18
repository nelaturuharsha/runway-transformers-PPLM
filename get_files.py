#! /usr/bin/env python3
# coding=utf-8

#Copyright (c) 2019 Uber Technologies, Inc.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained(
        'gpt2-medium',
        output_hidden_states=True
    )

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

def get_classifier(name):

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")

    return


def get_bag_of_words_indices(bag_of_words_ids_or_paths, tokenizer):
    bow_indices = []
    
    for id_or_path in bag_of_words_ids_or_paths:
        print(id_or_path)
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
    return 


bag_of_words = ['legal', 'military', 'space', 'technology', 'politics', 'religion', 'science']

get_bag_of_words_indices(bag_of_words, tokenizer)
models = ["clickbait", "sentiment"]
for i in models:
    print(i)
    get_classifier(i)

