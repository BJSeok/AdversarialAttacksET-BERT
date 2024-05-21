from __future__ import annotations

import sys
import os
import pathlib

sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.resolve()))

import textattack
import argparse
import torch
import torch.nn as nn
from uer.layers import str2embedding
from uer.encoders import str2encoder
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.utils.tokenizers import Tokenizer, BasicTokenizer, WordpieceTokenizer

# Special token words.
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
SENTINEL_TOKEN = (
    "[extra_id_0]"  # [extra_id_0], [extra_id_1], ... , should have consecutive IDs.
)


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


def load_or_initialize_parameters(args, model: Classifier) -> None:
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(
            torch.load(
                args.pretrained_model_path,
                map_location={
                    "cuda:1": "cuda:6",
                    "cuda:2": "cuda:6",
                    "cuda:3": "cuda:6",
                },
            ),
            strict=False,
        )
        pass
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
                pass
            continue
        pass
    return None


class Classifier(nn.Module):
    def __init__(self, args) -> None:
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

        return None

    def forward(self, input, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        src, seg = input.split(1, dim=1)
        src = src.squeeze(1)
        seg = seg.squeeze(1)

        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
            pass
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
            pass
        elif self.pooling == "last":
            output = output[:, -1, :]
            pass
        else:
            output = output[:, 0, :]
            pass
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        # if tgt is not None:
        #     if self.soft_targets and soft_tgt is not None:
        #         loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + (
        #             1 - self.soft_alpha
        #         ) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        #         pass
        #     else:
        #         loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
        #         pass
        #     return loss, logits
        # else:
        #     return None, logits
        return logits
        # pass

    pass


args_dict = {
    # finetune options
    "pretrained_model_path": "models/finetuned_model_iscx_vpn_service_pkt_01.bin",
    "output_model_path": "models/finetuned_model.bin",
    "vocab_path": "models/encryptd_vocab.txt",
    "spm_model_path": None,
    "train_path": "datasets/iscx-vpn-service/packet/train_dataset.tsv",
    "dev_path": "datasets/iscx-vpn-service/packet/valid_dataset.tsv",
    "test_path": "datasets/iscx-vpn-service/packet/test_dataset.tsv",
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
args = argparse.Namespace(**args_dict)
args = load_hyperparam(args)
set_seed(args.seed)
args.labels_num = count_labels_num(args.train_path)


class TokenizeSrc(Tokenizer):
    def __init__(self, args):
        super().__init__(args)
        self.seq_length = args.seq_length
        self.tokenizer = args.tokenizer

    def encode(self, text):
        src = self.tokenizer.convert_tokens_to_ids(
            [CLS_TOKEN] + self.tokenizer.tokenize(text)
        )
        seg = [1] * len(src)

        if len(src) > self.seq_length:
            src = src[: self.seq_length]
            seg = seg[: self.seq_length]
        while len(src) < self.seq_length:
            src.append(0)
            seg.append(0)
        return (src, seg)

    def __call__(self, batch):
        return [self.encode(text) for text in batch]


# tokenizer = str2tokenizer[args.tokenizer](args)
# args.tokenizer = tokenizer
args.tokenizer = str2tokenizer[args.tokenizer](args)
tokenizer = TokenizeSrc(args)


def get_model(device="cuda:6") -> textattack.models.wrappers.PyTorchModelWrapper:
    model = Classifier(args)
    model.to(device)
    load_or_initialize_parameters(args, model)

    model = textattack.models.wrappers.PyTorchModelWrapper(model, tokenizer)
    model.to(device)

    return model
