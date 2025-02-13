"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

# coding=UTF-8

import sys
sys.path.append("/home/aistudio/external-libraries/")
import argparse
import os
import pickle
from functools import partial

import paddle
from ann_util import build_index
from base_model import SemanticIndexBase
from data import convert_example, create_dataloader, gen_id2corpus, gen_text_file

from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import MapDataset
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--corpus_file", type=str, required=True,
                    help="The full path of input file")
parser.add_argument("--similar_text_pair_file", type=str,
                    required=True, help="The full path of similar text pair file")
parser.add_argument("--recall_result_dir", type=str, default='recall_result',
                    help="The full path of recall result file to save")
parser.add_argument("--recall_result_file", type=str,
                    default='recall_result_file', help="The file name of recall result")
parser.add_argument("--params_path", type=str, required=True,
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int,
                    help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--output_emb_size", default=None,
                    type=int, help="output_embedding_size")
parser.add_argument("--recall_num", default=10, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument('--model_name_or_path', default="rocketqa-zh-base-query-encoder",
                    help="The pretrained model used for training")
parser.add_argument("--hnsw_m", default=100, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_ef", default=100, type=int,
                    help="Recall number for each query from Ann index.")
parser.add_argument("--hnsw_max_elements", default=1000000,
                    type=int, help="Recall number for each query from Ann index.")
parser.add_argument('--index_path', default="hnsw.index",
                    help="hnsw_index_path")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    trans_func = partial(convert_example, tokenizer=tokenizer, max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id, dtype="int64"),  # text_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id, dtype="int64"),  # text_segment
    ): [data for data in fn(samples)]

    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    model = SemanticIndexBase(pretrained_model, output_emb_size=args.output_emb_size)
    model = paddle.DataParallel(model)

    # Load pretrained semantic model
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        # logger.info("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")

    id2corpus = gen_id2corpus(args.corpus_file)

    # convert_example function's input must be dict
    corpus_list = [{idx: text} for idx, text in id2corpus.items()]
    corpus_ds = MapDataset(corpus_list)

    corpus_data_loader = create_dataloader(
        corpus_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    # Need better way to get inner model of DataParallel
    inner_model = model._layers

    final_index = build_index(args, corpus_data_loader, inner_model)

    with open(args.index_path, "wb") as wfile:
        pickle.dump(final_index, wfile)
