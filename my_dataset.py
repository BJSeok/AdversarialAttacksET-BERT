from __future__ import annotations

import sys
import os
import pathlib

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.resolve()))

import textattack
import argparse
from uer.layers import str2embedding
from uer.encoders import str2encoder
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed


# Special token words.
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SENTINEL_TOKEN = (
    "[extra_id_0]"  # [extra_id_0], [extra_id_1], ... , should have consecutive IDs.
)

# Special token words.
PAD_ID = 0

args_dict = {
    # finetune options
    "pretrained_model_path": "models/finetuned_model_iscx_vpn_service_pkt_01.bin",
    "output_model_path": "models/finetuned_model.bin",
    "vocab_path": "models/encryptd_vocab.txt",
    "spm_model_path": None,
    "train_path": "datasets/iscx-vpn-service/packet/train_dataset.tsv",
    "dev_path": "datasets/iscx-vpn-service/packet/valid_dataset.tsv",
    "test_path": "datasets/iscx-vpn-service/packet/test_dataset_100.tsv",
    "config_path": "models/bert/base_config.json",
    # model options
    "embedding": "word_pos_seg",
    "max_seq_length": 512,
    "relative_position_embedding": False,
    "relative_attention_buckets_num": 32,
    "remove_embedding_layernorm": False,
    "remove_attention_scale": False,
    "encoder": "transformer",
    "mask": "fully_visible",
    "layernorm_positioning": "post",
    "feed_forward": "dense",
    "remove_transformer_bias": False,
    "layernorm": "normal",
    "bidirectional": False,
    "factorized_embedding_parameterization": False,
    "parameter_sharing": False,
    # optimization options
    "learning_rate": 2e-5,
    "warmup": 0.1,
    "fp16": False,
    "fp16_opt_level": "O1",
    "optimizer": "adamw",
    "scheduler": "linear",
    # training options
    "batch_size": 32,
    "seq_length": 128,
    "dropout": 0.5,
    "epochs_num": 10,
    "report_steps": 100,
    "seed": 7,
    # main options
    "pooling": "first",
    "tokenizer": "bert",
    "soft_targets": False,
    "soft_alpha": 0.5,
}
# args_dict["pretrained_model_path"] = "models/finetuned_model_cstnet_02.bin"
# args_dict["train_path"] = "datasets/cstnet-tls1.3/packet/train_dataset.tsv"
# args_dict["dev_path"] = "datasets/cstnet-tls1.3/packet/valid_dataset.tsv"
# args_dict["test_path"] = "datasets/cstnet-tls1.3/packet/test_dataset_100.tsv"


def count_labels_num(path: str) -> int:
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                    continue
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
            continue
        pass
    return len(labels_set)


args = argparse.Namespace(**args_dict)
args = load_hyperparam(args)
set_seed(args.seed)
args.labels_num = count_labels_num(args.train_path)

tokenizer = str2tokenizer[args.tokenizer](args)
args.tokenizer = tokenizer

dataset, columns = [], {}

with open(args.test_path, mode="r", encoding="utf-8") as f:
    for line_id, line in enumerate(f):
        if line_id == 0:
            for i, column_name in enumerate(line.strip().split("\t")):
                columns[column_name] = i
            continue
        line = line[:-1].split("\t")
        tgt = int(line[columns["label"]])
        if "text_b" not in columns:
            text_a = line[columns["text_a"]]
            src = [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
            seg = [1] * len(src)
        else:
            text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
            src_a = [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
            src_b = args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
            src = src_a + src_b
            seg = [1] * len(src_a) + [2] * len(src_b)

        if len(src) > args.seq_length:
            src = src[: args.seq_length]
            seg = seg[: args.seq_length]
        while len(src) < args.seq_length:
            src.append(0)
            seg.append(0)
        dataset.append(((text_a), tgt))

dataset = textattack.datasets.Dataset(dataset, input_columns=["text_a"])
