#!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import os
import sys
import pickle
import random
import re
import shutil
import itertools as it
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import time
import json
import numpy as np
import tempfile

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    get_worker_info
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tokenizers import BertWordPieceTokenizer, Tokenizer
from src.tokenizer import truncate
from src.utils.iter import batch

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)


from torch.utils.tensorboard import SummaryWriter
import botocore
import boto3
from smart_open import open
import concurrent.futures
import threading

thread_local = threading.local()
s3_upload_futures = []

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Required parameters
parser.add_argument(
    "--train_data_file", default=None, type=lambda x: x.split(','), required=True,
    help="The input training data text file. If multiple files, separate by comma."
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--model_type", type=str, help="The model architecture to be trained or fine-tuned.", default="bert",
)

# Other parameters
parser.add_argument(
    "--csv_uri_column", type=str, default="uri",
    help="Column name with the URI of the file/object.",
)
parser.add_argument(
    "--csv_size_column", type=str, default="size",
    help="Column name with the size of the file/object.",
)
parser.add_argument(
    "--data_loader_num_workers", type=int, default=5,
    help="Number of workers to handle preprocessing of data.",
)
parser.add_argument(
    "--should_continue", action="store_true",
    help="Whether to continue from latest checkpoint in output_dir"
)
parser.add_argument(
    "--model_name_or_path", type=str, default="bert-base-uncased",
    help="The initial model to continue domain pretraining from. Will be ignore once there are checkpoints",
)

parser.add_argument(
    "--mlm_probability", type=float, default=0.15,
    help="Ratio of tokens to mask for masked language modeling loss"
)

parser.add_argument(
    "--config_name", type=str, default=None,
    help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
)
parser.add_argument(
    "--tokenizer_vocab", type=str, default=None,
    help="Path to tokenizer vocab txt file.",
)
parser.add_argument(
    "--cache_dir", type=str, default=None,
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
)
parser.add_argument(
    "--block_size", type=int, default=512,
    help="Input sequence length after tokenization.",
)
parser.add_argument(
    "--max_file_read_bytes", type=int, default=4000,
    help="Maximum number of bytes to read per file per read operation.",
)
parser.add_argument(
    "--min_file_read_bytes", type=int, default=1024,
    help="Minimum (will use file size if maller) number of bytes to read per file per read operation" ,
)
parser.add_argument(
    "--evaluate_during_training", action="store_true",
    help="Run evaluation during training at each logging step."
)
parser.add_argument(
    "--max_run_time_in_minutes", type=int,
    help="Maximum number of minutes to train for.",
)
parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument(
    "--gradient_accumulation_steps", type=int, default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--max_steps", type=int, required=True,
    help="Total number of training steps to perform.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
#parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--seed", type=int, help="random seed for initialization")

parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


def process_s3_upload(args):
    src, dst, delete_after = args

    t0 = time.time()
    if dst[:5].lower() == "s3://":
        if os.path.isfile(src):
            cmd = f"aws s3 cp {src} {dst}"
        else:
            if dst[-4:].lower() == ".tgz":
                cmd = f"tar C {src} -zcf - . | aws s3 cp - {dst}"
            elif dst[-4:].lower() == ".tar":
                cmd = f"tar C {src} -cf - . | aws s3 cp - {dst}"
            else:
                cmd = f"aws s3 cp --recursive {src} {dst}"

        os.system(cmd)
        if delete_after:
            shutil.rmtree(src)
        print(f"S3 upload to {dst} took: {time.time() - t0}")
    else:
        print(f"S3 uploader can't handle destination: {dst}")


def s3_uploads_future_cb(future):
    s3_upload_futures.remove(future)


# TODO: Handle upload failures
def append_s3_upload(src, dst, delete_after=False):
    pool = getattr(thread_local, 's3_upload_futures_pool', None)
    if pool is None:
        thread_local.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        pool = thread_local.pool

    future = pool.submit(process_s3_upload, (src, dst, delete_after))
    future.add_done_callback(s3_uploads_future_cb)
    s3_upload_futures.append(future)
    print(f"Added S3 upload. There are {len(s3_upload_futures)} upload set(s) left to finish")


def get_s3_uri_content(s3uri, offset=0, max_bytes=""):
    bucket, key = s3uri.split("/", 1)

    # Boto is not thread safe, get one session per thread
    s3 = getattr(thread_local, 's3', None)
    if s3 is None:
        thread_local.s3 = boto3.session.Session().client('s3')
        s3 = thread_local.s3

    try:
        content = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset+max_bytes}")['Body'].read()
        return content
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Try without file type suffix in case we have it wrong
            r = s3.list_objects(Bucket=bucket, Prefix=key.rsplit(".", 1)[0])
            if 'Contents' in r and len(r['Contents']) == 1:
                try:
                    key = r['Contents'][0]["Key"]
                    content = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={offset}-{offset+max_bytes}")['Body'].read()
                    return content
                except:
                    pass
    return None


def load_uri(uri, mode='rt', offset=0, max_bytes=""):
    data = None

    if uri is None or uri == "":
        return None

    try:
        if uri[0:5].lower() == "s3://":
            data = get_s3_uri_content(uri[5:], offset, max_bytes=max_bytes)
        else:
            with open(uri, mode='rb') as inf:
                inf.seek(offset)
                data = inf.read(int(max_bytes))
        if data is not None and 't' in mode:
            #data = data.decode('utf-8', errors="ignore")
            text_data = None
            try:
                text_data = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                try:
                    text_data = data.decode("utf-8")
                except UnicodeDecodeError:
                    text_data = data.decode("iso-8859-1")
            data = text_data
    except:
        print("Failed to read:", uri)
        import traceback
        print(traceback.format_exc())

        data = None

    return data


def set_seed(args):
    if not args.seed:
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs: torch.Tensor, tokenizer: Tokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def get_rand_batch():
    input_ids = []
    masks = []
    for i in range(batch_size):
        l = random.randint(200,max_length)
        ids = [101] + random.sample(range(2000,29000), l)
        mask = [1] * len(ids)
        padding = [0] * (max_length - len(ids))
        ids += padding
        mask += padding

        input_ids.append(ids)
        masks.append(mask)
        
    return input_ids, masks


class DFMLMDataset(Dataset):

    def __init__(self, df, tokenizer, args):
        self.df = df
        self.tokenizer = tokenizer
        self.args = args
        self.max_seq_length = self.args.block_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print("IDx", idx)
        # fake_ids = torch.tensor([[1, 2, 3, 4, 5, 6 , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        # return [1, 2, 3, 4, 5, 6 , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # return mask_tokens(fake_ids, self.tokenizer, self.args)

        row = self.df.iloc[idx]

        # At least set the offset to read the last 1024 bytes of the file
        offset = random.randint(0, max(0, int(row['size']) - self.args.min_file_read_bytes))
        txt = load_uri(row['uri'], offset=offset, max_bytes=self.args.max_file_read_bytes)

        if txt is None:
            txt = ""

        startpos = 0
        if offset > 0:
            startpos = txt.find(' ')
            if startpos < 0:
                startpos = 0

        #ids = tokenizer.encode(txt, add_special_tokens=True)[:self.max_seq_length]
        # The one above is more generic, but why do we want the [SEP] token at the end?
        #ids = [101] + self.tokenizer.encode(txt, add_special_tokens=False)[:(self.args.block_size - 1)]
        ids = [101] + self.tokenizer.encode(txt, add_special_tokens=False, max_length=(self.args.block_size-1))
        #ids = [1, 2, 3, 4, 5, 6 , 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        padding = [0] * (self.args.block_size - len(ids))
        ids += padding

        # Not sure of the speed improvement by calling mask here vs on collate
        return mask_tokens(torch.tensor([ids], dtype=torch.long), self.tokenizer, self.args)


    def dataloader_batch_collate(self, batch):
        b_input_ids = torch.LongTensor(len(batch), self.args.block_size)
        b_input_labels = torch.LongTensor(len(batch), self.args.block_size)

        for i, s in enumerate(batch):
            ids, labels = s
            b_input_ids[i] = ids[0]
            b_input_labels[i] = labels[0]

        return b_input_ids, b_input_labels


def save_state(dst, state):
    if dst[:5].lower() == "s3://":
        outf, fpath = tempfile.mkstemp(suffix="json", text=True)
        os.close(outf)
    else:
        fpath = dst

    with open(fpath, "w") as outf:
        json.dump(state, outf)

    if dst[:5].lower() == "s3://":
        append_s3_upload(fpath, dst, delete_after=True)

def save_model_training_objects(args, dst_dir, tokenizer, model, optimizer, scheduler):
    if dst_dir[:5].lower() == "s3://":
        dirname = tempfile.mkdtemp()
    else:
        dirname = dst_dir
        os.makedirs(dst_dir, exist_ok=True)

    model_to_save = model
    while hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module

    torch.save(args, os.path.join(dirname, "training_args.bin"))
    tokenizer.save_pretrained(dirname)
    model_to_save.save_pretrained(dirname)
    torch.save(optimizer.state_dict(), os.path.join(dirname, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(dirname, "scheduler.pt"))
    
    if dst_dir[:5].lower() == "s3://":
        append_s3_upload(dirname, dst_dir, delete_after=True)


def get_model_training_objects(args, src_dir=None):
    if src_dir is None:
        src_dir = args.model_name_or_path

    if src_dir[:5].lower() == "s3://":
        dirname = tempfile.mkdtemp()
        if src_dir[-4:].lower() == ".tgz":
            cmd = f"aws s3 cp {src_dir} - | tar C {dirname} -zxf - . "
        elif src_dir[-4:].lower() == ".tar":
            cmd = f"aws s3 cp {src_dir} - | tar C {dirname} -xf - . "
        else:
            cmd = f"aws s3 cp --recursive {src_dir} {dirname}"
        print(cmd)
        os.system(cmd)
        print("Copied S3 model to:", dirname)
    else:
        dirname = src_dir

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif dirname:
        config = config_class.from_pretrained(dirname, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_vocab:
        tokenizer_vocab = args.tokenizer_vocab
        if tokenizer_vocab[:5].lower() == 's3://':
            tok_dirname = tempfile.mkdtemp()
            new_tokenizer_path = f"{tok_dirname}/{tokenizer_vocab[5:].split('/', 1)[1]}"
            cmd = f"aws s3 cp {tokenizer_vocab} {new_tokenizer_path}"
            print(cmd)
            os.system(cmd)
            print("Copied tokenizer vocab to:", new_tokenizer_path)
            tokenizer_vocab = new_tokenizer_path
        else:
            tokenizer_vocab = str(Path(args.tokenizer_vocab).parent)
        tokenizer = tokenizer_class.from_pretrained(tokenizer_vocab, cache_dir=args.cache_dir)
    elif dirname:
        tokenizer = tokenizer_class.from_pretrained(dirname, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if dirname:
        model = model_class.from_pretrained(
            dirname,
            from_tf=bool(".ckpt" in dirname),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)
    model.to(args.device)

    # Prepare optimizer and scheduler (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    # Check if saved optimizer or scheduler states exist
    if (
        dirname
        and os.path.isfile(os.path.join(dirname, "optimizer.pt"))
        and os.path.isfile(os.path.join(dirname, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        if args.no_cuda:
            optimizer.load_state_dict(torch.load(os.path.join(dirname, "optimizer.pt"), map_location=torch.device('cpu')))
            scheduler.load_state_dict(torch.load(os.path.join(dirname, "scheduler.pt"), map_location=torch.device('cpu')))
        else:
            optimizer.load_state_dict(torch.load(os.path.join(dirname, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(dirname, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        print("Enabling fp16")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        print("Enabling Data Parallel")
        model = torch.nn.DataParallel(model)

    return tokenizer, model, optimizer, scheduler


# def train(args,
#           train_df,
#           model: PreTrainedModel,
#           tokenizer: Tokenizer
# ) -> Tuple[int, float]:
def train(args) -> Tuple[int, float]:
    """ Train the model """

    tb_writer = SummaryWriter()
        
    state_path = f"{args.output_dir}/domain_pre-training_state.json"
    state = {
        "v": 0.1,
        "finished": False,
        "ts": {"t0": time.time()},
        "saved": {"idx": 0, "data": {}},
        "curr_global_step_idx": 0,
        "cumulative_runtime": 0,
        "metrics": {
            "validation": [],
        },
        "timed_out": False
    }
    start_step_idx = 0

    if args.output_dir[:5].lower() != "s3://":
        if not os.path.exists(f"{args.output_dir}/models"):
            os.makedirs(f"{args.output_dir}/models")
        if not os.path.exists(f"{args.output_dir}/metrics"):
            os.makedirs(f"{args.output_dir}/metrics")

    model = None
    if args.should_continue:
        saved_path = None
        print(f"Resuming state from: {state_path}")
        try:
            tmp_state = json.loads(open(state_path, 'r').read())
            if tmp_state["finished"] == True:
                print("State shows that training has finished.")
                print(f"Final model was saved at: {saved_path}")
                print("Quiting.")
                exit(0)

            saved_path = tmp_state["saved"]["data"][str(tmp_state["saved"]["idx"])]["model"]
            print("SP:", saved_path)
            tokenizer, model, optimizer, scheduler = get_model_training_objects(args, saved_path)

            # TODO: Do a deeper validation of state
            # By now the state should be valid. Replace proper variables
            tmp_state['curr_global_step_idx'] = tmp_state["saved"]["data"][str(tmp_state["saved"]["idx"])]["global_step_idx"]
            start_step_idx = tmp_state['curr_global_step_idx'] + 1
            tmp_state['saved']['idx'] += 1
            state = tmp_state
        except Exception:
            if saved_path:
                print(f"Could not load model from {saved_path}")
            else:
                print(f"Could not get model information from {state_path}")
            print(sys.exc_info()[0])
            import traceback
            print(traceback.format_exc())
            pass

        #     if saved_path[:5].lower() == "s3://":
        #         dirname = tempfile.mkdtemp()
        #         if saved_path[-4:].lower() == ".tgz":
        #             cmd = f"aws s3 cp {saved_path} - | tar C {dirname} -zxf - . "
        #         elif saved_path[-4:].lower() == ".tar":
        #             cmd = f"aws s3 cp {saved_path} - | tar C {dirname} -xf - . "
        #         else:
        #             cmd = f"aws s3 cp --recursive {saved_path} {dirname}"

        #         print(cmd)
        #         os.system(cmd)
        #         #exit(4)
        #     else:
        #         dirname = saved_path

        #     if dirname[-1] != "/":
        #         dirname += "/"

        #     tmp_tokenizer = AutoTokenizer.from_pretrained(dirname)

        #     tmp_model = BertForMaskedLM.from_pretrained(
        #         dirname, config=AutoConfig.from_pretrained(dirname))

        #     # Prepare optimizer and schedule (linear warmup and decay)
        #     no_decay = ["bias", "LayerNorm.weight"]
        #     optimizer_grouped_parameters = [
        #         {
        #             "params": [p for n, p in tmp_model.named_parameters() if not any(nd in n for nd in no_decay)],
        #             "weight_decay": args.weight_decay,
        #         },
        #         {"params": [p for n, p in tmp_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        #     ]
        #     tmp_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        #     tmp_scheduler = get_linear_schedule_with_warmup(
        #         tmp_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
        #     )

        #     # Check if saved optimizer or scheduler states exist
        #     if os.path.isfile(os.path.join(dirname, "optimizer.pt")):
        #         tmp_optimizer.load_state_dict(torch.load(os.path.join(dirname, "optimizer.pt")))
        #     if os.path.isfile(os.path.join(dirname, "scheduler.pt")):
        #         tmp_scheduler.load_state_dict(torch.load(os.path.join(dirname, "scheduler.pt")))

        #     if saved_path[:5].lower() == "s3://":
        #         shutil.rmtree(dirname)

        #     # TODO: Do a deeper validation of state
        #     # By now the state should be valid. Replace proper variables
        #     tokenizer = tmp_tokenizer
        #     model = tmp_model
        #     optimizer = tmp_optimizer
        #     scheduler = tmp_scheduler
        #     state = tmp_state
        # except Exception:
        #     print(sys.exc_info()[0])
        #     import traceback
        #     print(traceback.format_exc())
        #     exit(6)
        #     # TODO: We should probably return and error (or raise custom exception) here and
        #     # let the caller initialize the model. At this point we can't be sure what the
        #     # configuration of the model should be.
        #     # Above, we don't set used variables until the tokenizer and model can be loaded.
        #     # If the caller has used the matter.set_* functions this should be enough, but the
        #     # flow should be dictated by the caller, not by exceptions catching here. Otherwise
        #     # we are forcing a valid model be loaded every time, even when resuming.
        #     print("Couldn't read/continue from previous state file. Discarding saved state.")

    if model is None:
        # Need all training objects
        print("Training model from scratch")
        tokenizer, model, optimizer, scheduler = get_model_training_objects(args)

    #args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    #inputfname = "/disco/data/matters/xmatter/gp_priv1/dbdata/df-document_info-with_uri_clean.csv"
    #inputfname = "/disco/data/matters/xmatter/gp_priv1/dbdata/df-document_info-in_golden_set.csv"
    train_df = pd.concat((pd.read_csv(f, keep_default_na=False,
                                      header=0,
                                      usecols=[args.csv_size_column, args.csv_uri_column],
                                      dtype={args.csv_size_column: 'int64', args.csv_uri_column: str})
                          for f in args.train_data_file))
    train_df.rename({args.csv_uri_column: "uri", args.csv_size_column: "size"}, inplace=True)

    # Setup the dataset and data loader
    ds = DFMLMDataset(train_df, tokenizer, args)
    data_loader = DataLoader(ds,
                             batch_size=args.batch_size, shuffle=True,
                             #batch_size=args.batch_size, shuffle=False,
                             num_workers=args.data_loader_num_workers,
                             collate_fn=ds.dataloader_batch_collate)

    t_total = args.max_steps
    args.num_train_epochs = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num docs = %d", len(train_df))
    logger.info(f"  Total bytes = {train_df['size'].sum():,}")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.batch_size)
    #logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    #print("device=", args.device)
    #model = torch.nn.DataParallel(model)
    #model.to("cuda")
    model.to(args.device)
    model.zero_grad()

    set_seed(args)  # Added here for reproducibility

    data_loader_iterator = iter(data_loader)
    t_b_pre = time.time()
    # TODO: Note that I removed the reproducibility of batches
    for step in range(start_step_idx, args.max_steps):
        state["curr_global_step_idx"] = step

        try:
            batch = next(data_loader_iterator)
        except StopIteration:
            data_loader_iterator = iter(data_loader)
            batch = next(data_loader_iterator)

        inputs, labels = batch
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        model.train()
        outputs = model(inputs, masked_lm_labels=labels)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            #epoch_iterator.set_description(desc=f"Loss {(tr_loss/global_step):7.3f}")
            print(f"Step {step+1}/{args.max_steps} Loss {tr_loss/(step+1):7.3f}")

            if args.logging_steps > 0 and (step + 1) % args.logging_steps == 0:
                # Log metrics
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], step+1)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, step+1)
                logging_loss = tr_loss

            if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                model_dst = f"{args.output_dir}/models/global_step_idx_{step}"
                save_model_training_objects(args, model_dst, tokenizer, model, optimizer, scheduler)

                state["saved"]["data"][state['saved']['idx']] = {
                    "global_step_idx": step,
                    "model": model_dst,
                    "metrics": None,
                    "validation": False
                }

                print("saving state to ", state_path)
                save_state(state_path, state)

                state['saved']['idx'] += 1

        state["cumulative_runtime"] += time.time() - t_b_pre
        t_b_pre = time.time()

        if args.max_run_time_in_minutes:
            if (state["cumulative_runtime"]) / 60 > args.max_run_time_in_minutes:
                print(f"Terminating because we are over {args.max_run_time_in_minutes} minutes")
                state["timed_out"] = True
                break

    print("Finished training")

    tb_writer.close()

    # TODO: Keep track of last saved model and avoid saving it twice
    model_dst = f"{args.output_dir}/models/global_step_idx_{step}"
    print(f"Saving final trained model to {model_dst}")
    save_model_training_objects(args, model_dst, tokenizer, model, optimizer, scheduler)

    if args.should_continue:
        state["curr_global_step_idx"] = step
        state["finished"] = True
        state["ts"]["tz"] = time.time()

        state["saved"]["data"][state['saved']['idx']] = {
            "global_step_idx": step,
            "model": model_dst,
            "metrics": None,
            "validation": False,
            "final": True
        }

        print("saving state to ", state_path)
        save_state(state_path, state)

    while len(s3_upload_futures) > 0:
        print(f"Waiting on {len(s3_upload_futures)} upload set(s)")
        time.sleep(2)

    return step, tr_loss / step


def main():

    results = {}

    args = parser.parse_args()

    supported_models = ["bert", "roberta", "distilbert", "camembert"]
    if not args.model_type in supported_models:
        raise ValueError(f"Model '{args.model_type}' not supported. Must be one of: {supported_models}")

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging

    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     level=logging.WARN if args.local_rank in [-1, 0] else logging.WARN,
    # )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer

    # if args.n_gpu > 1:
    #     logger.info(f"Using DataParallel with {args.n_gpu} GPUs")
    #     model = torch.nn.DataParallel(model)
    # model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    #global_step, tr_loss = train(args, None, model, tokenizer)
    global_step_idx, tr_loss = train(args)

    logger.info(f"Final global step: {global_step_idx + 1}")
    logger.info(f"Final loss: {tr_loss}")

    return tr_loss


if __name__ == "__main__":
    main()
