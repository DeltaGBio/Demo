import os, random, json, math, time, argparse, re
import torch
import numpy as np
from collections import OrderedDict
from datetime import datetime
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import mint
from mint.model.esm2 import ESM2
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy
import torch.nn.functional as F

class CFG:
    """Global config for training and model."""
    EPOCHS = 1000
    BATCH_SIZE = 32
    LR = 1e-3
    WD = 1e-6
    SEED = 2024
    EMBED_DIM = 1280
    OUTPUT_DIM = 1
    RANK = 256

# ----------------------------
# Lightning Module for Inference
# ----------------------------
class DeltaGLightningModule(pl.LightningModule):
    """LightningModule for DeltaG regression using CombinedModel."""
    def __init__(self, model, loss_fn, lr, weight_decay=CFG.WD, gamma=0.95):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma

    def forward(self, wt_chains, wt_chain_ids, mut_chains, mut_chain_ids):
        return self.model(wt_chains, wt_chain_ids, mut_chains, mut_chain_ids)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # Unpack the batch (the fifth element is the target which is not needed during inference)
        wt_chains, wt_chain_ids, mut_chains, mut_chain_ids, _ = batch
        return self.forward(wt_chains, wt_chain_ids, mut_chains, mut_chain_ids)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

# ----------------------------
# Combined Model
# ----------------------------
class CombinedModel(nn.Module):
    """Model: embedder + MLP head for regression."""
    def __init__(self, embedder, mlp):
        super().__init__()
        self.embedder = embedder
        self.mlp = mlp

    def forward(self, wt_chains, wt_chain_ids, mut_chains, mut_chain_ids):
        embeddings = self.embedder(wt_chains, wt_chain_ids, mut_chains, mut_chain_ids)
        output = self.mlp(embeddings)
        return output

# ----------------------------
# MLP Head Definition
# ----------------------------
class MLP(nn.Module):
    """Multi-layer perceptron regression head."""
    def __init__(self, input_dim=CFG.EMBED_DIM, output_dim=CFG.OUTPUT_DIM, lr=CFG.LR, weight_decay=CFG.WD):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, self.output_dim)
        self.initialize_weights()

    def initialize_weights(self):
        """Xavier initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

# ----------------------------
# Dataset & Collate Function
# ----------------------------
class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequence pairs and targets from CSV."""
    def __init__(self, df_path, col1, col2, target_col, test_run=False):
        super().__init__()
        self.df = pd.read_csv(df_path)
        if test_run:
            self.df = self.df.sample(n=101)
        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()
        if isinstance(target_col, list):
            self.targets = self.df[target_col].to_numpy()
        else:
            self.targets = self.df[target_col].tolist()

    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        return (
            self.seqs1[index],
            self.seqs2[index],
            self.targets[index],
        )

## Frome DataFrame two Dataset
class ProteinDFDataset(Dataset):
    def __init__(self, df, col1, col2, target_col=None):
        super().__init__()
        self.df = df
        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()
        if isinstance(target_col, list):
            self.targets = self.df[target_col].to_numpy()
        else:
            self.targets = self.df[target_col].tolist()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return (
            self.seqs1[index],
            self.seqs2[index],
            self.targets[index],
        )

# ----------------------------
# Collate function for mutational PPI
# ----------------------------
class MutationalPPICollateFn:
    """Collate function for mutational PPI batches."""
    def __init__(self, truncation_seq_length=None):
        self.alphabet = mint.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        wt_ab, mut_ab, labels = zip(*batches)
        wt_chains = [self.convert(c) for c in [wt_ab]]
        wt_chain_ids = [
            torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(wt_chains)
        ]
        wt_chains = torch.cat(wt_chains, -1)
        wt_chain_ids = torch.cat(wt_chain_ids, -1)
        mut_chains = [self.convert(c) for c in [mut_ab]]
        mut_chain_ids = [
            torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(mut_chains)
        ]
        mut_chains = torch.cat(mut_chains, -1)
        mut_chain_ids = torch.cat(mut_chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        return wt_chains, wt_chain_ids, mut_chains, mut_chain_ids, labels

    def convert(self, seq_str_list):
        """Tokenize and pad a list of sequence strings."""
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start:start + self.truncation_seq_length]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, :len(seq_encoded)] = seq
        return tokens

# ----------------------------
# Sequence Embedder
# ----------------------------
class SequenceEmbedder(nn.Module):
    """ESM2-based sequence embedder for paired chains."""
    def __init__(
        self, cfg, checkpoint_path, freeze_percent=0.0, use_multimer=True, sep_chains=True, device="cuda:0"
    ):
        super().__init__()
        self.cfg = cfg
        self.sep_chains = sep_chains
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=use_multimer,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if use_multimer:
            new_checkpoint = OrderedDict((key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items())
            self.model.load_state_dict(new_checkpoint)
        else:
            def upgrade_state_dict(state_dict):
                prefixes = ["encoder.sentence_encoder.", "encoder."]
                pattern = re.compile("^" + "|".join(prefixes))
                return {pattern.sub("", name): param for name, param in state_dict.items()}
            new_checkpoint = upgrade_state_dict(checkpoint["model"])
            self.model.load_state_dict(new_checkpoint)
        total_layers = cfg.encoder_layers
        for name, param in self.model.named_parameters():
            if ("embed_tokens.weight" in name or "_norm_after" in name or "lm_head" in name):
                param.requires_grad = False
            else:
                layer_num = name.split(".")[1]
                if int(layer_num) <= math.floor(total_layers * freeze_percent):
                    param.requires_grad = False

    def get_one_chain(self, chain_out, mask_expanded, mask):
        """Mean pooling over masked positions for a chain."""
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()
        mean_chain_out = sum_masked / mask_counts
        return mean_chain_out

    def forward_one(self, chains, chain_ids):
        """Embed a single wildtype or mutant chain pair."""
        mask = ((~chains.eq(self.model.cls_idx)) & (~chains.eq(self.model.eos_idx)) & (~chains.eq(self.model.padding_idx)))
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]
        if self.sep_chains:
            mask_chain_0 = (chain_ids.eq(0) & mask).unsqueeze(-1).expand_as(chain_out)
            mask_chain_1 = (chain_ids.eq(1) & mask).unsqueeze(-1).expand_as(chain_out)
            mean_chain_out_0 = self.get_one_chain(chain_out, mask_chain_0, (chain_ids.eq(0) & mask))
            mean_chain_out_1 = self.get_one_chain(chain_out, mask_chain_1, (chain_ids.eq(1) & mask))
            return torch.cat((mean_chain_out_0, mean_chain_out_1), -1)
        else:
            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()
            mean_chain_out = sum_masked / mask_counts
            return mean_chain_out

    def forward(self, wt_chains, wt_chain_ids, mut_chains, mut_chain_ids, cat=False):
        """Embed wildtype and mutant, return difference or concat."""
        wt_chains_out = self.forward_one(wt_chains, wt_chain_ids)
        mut_chains_out = self.forward_one(mut_chains, mut_chain_ids)
        if cat:
            return torch.cat((wt_chains_out, mut_chains_out), -1)
        return wt_chains_out - mut_chains_out




