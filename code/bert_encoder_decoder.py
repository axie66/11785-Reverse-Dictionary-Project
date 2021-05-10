import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import List
from tqdm import tqdm
import sys
import datetime
from dataset import get_data, WantWordsDataset as WWDatappend('../code')

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    EncoderDecoderModel,
    BertGenerationEncoder,
    BertGenerationDecoder,
    BertTokenizer,
)
import gc
import wandb

# Should also first install pytorch-transformers (aka transformers)
# See here https://pytorch.org/hub/huggingface_pytorch-transformers/
# and here https://huggingface.co/transformers/
# You might also have to manually pip install sentencepiece

# Download vocabulary from S3 and cache.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Download model and configuration from S3 and cache.
enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-uncased", "bert-base-uncased"
)
tokens = tokenizer(
    ["a hot and dry place", "something you eat after dinner"],
    return_tensors="pt",
    padding=True,
)
print("input", tokens)
ground_truth = tokenizer(["desert", "dessert"], return_tensors="pt", padding=True)
print("ground truth", ground_truth)
input_ids = tokens["input_ids"]
print(input_ids.shape)
attention_mask = tokens["attention_mask"]
out = enc_dec(
    input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=input_ids
)
print("out", out)
logits = out["logits"]
print(logits.shape)
best = logits.argmax(-1)
print(best.shape)
tokenizer.decode(best[0])
print(type(tokens))
criterion = nn.CrossEntropyLoss()
gt_lens = torch.sum(ground_truth["attention_mask"], dim=-1) - 2
gt_input_ids = ground_truth["input_ids"]
Y = [gt_input_ids[i][1 : 1 + gt_lens[i]] for i in range(len(gt_lens))]
X = [logits[i][1 : 1 + gt_lens[i]] for i in range(len(gt_lens))]
print(X)
print(Y)
batch_loss = sum(criterion(x, y) for x, y in zip(X, Y))
batch_loss / len(gt_lens)


class BertEncDec(nn.Module):
    def __init__(self, enc_dec, criterion):
        super(BertEncDec, self).__init__()
        self.enc_dec = enc_dec  # the BERT encoder/decoder
        self.criterion = criterion

    def _init_weights(self):
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, y):
        """Where x, y are BatchEncodings returned by a tokenizer object"""
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        batch_size = len(input_ids)

        out = self.enc_dec(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=input_ids,
        )

        logits = out["logits"]
        gt = y["input_ids"]

        # subtract 2 to account for start/end tokens
        gt_lens = torch.sum(y["attention_mask"], dim=-1) - 2
        Y = (gt[i][1 : 1 + gt_lens[i]] for i in range(batch_size))
        X = (logits[i][1 : 1 + gt_lens[i]] for i in range(batch_size))
        batch_loss = sum(self.criterion(x, y) for x, y in zip(X, Y))

        return logits, batch_loss / batch_size


d, word2vec = get_data("../wantwords-english-baseline/data")

(
    train_data,
    train_data_def,
    dev_data,
    test_data_seen,
    test_data_unseen,
    test_data_desc,
) = d

train_dataset = WWData(train_data + train_data_def, word2vec, 300, tokenizer)
dev_dataset = WWData(dev_data, word2vec, 300, tokenizer)
# Three distinct test sets
test_dataset_seen = WWData(test_data_seen, word2vec, 300, tokenizer)
test_dataset_unseen = WWData(test_data_unseen, word2vec, 300, tokenizer)
test_dataset_desc = WWData(test_data_desc, word2vec, 300, tokenizer)

batch_size = 16
num_workers = 4


def make_loader(dataset, shuffle):
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        pin_memory=False,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda x: dataset.collate_fn(x, word2vec=False),
    )


train_loader = make_loader(train_dataset, True)
print(f"Train loader: {len(train_loader)}")
dev_loader = make_loader(dev_dataset, True)
print(f"Dev loader: {len(dev_loader)}")

test_loader_seen = make_loader(test_dataset_seen, False)
print(f"Test loader (seen): {len(test_loader_seen)}")
test_loader_unseen = make_loader(test_dataset_unseen, False)
print(f"Test loader (unseen): {len(test_loader_unseen)}")
test_loader_desc = make_loader(test_dataset_desc, False)
print(f"Test loader (descriptions): {len(test_loader_desc)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.load('../trained_models/bert_baseline_wwdata.pt')
criterion = nn.CrossEntropyLoss()
model = BertEncDec(enc_dec, criterion)
model = model.to(device)
model

epochs = 5

lr = 5e-5
optim = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=(len(train_loader) // 10),
    num_training_steps=(epochs * len(train_loader)),
)
epoch = 0
scaler = GradScaler()
wandb.init(project="reverse-dictionary", entity="reverse-dict")

config = wandb.config
config.learning_rate = lr
config.epochs = epochs
config.batch_size = batch_size
config.optimizer = type(optim).__name__
config.scheulder = type(scheduler).__name__

# wandb.watch(model)


def evaluate(pred, gt, test=False):
    acc1 = acc10 = acc100 = 0
    n = len(pred)
    pred_rank = []
    for p, word in zip(pred, gt):
        if test:
            loc = (p == word).nonzero(as_tuple=True)
            if len(loc) != 0:
                pred_rank.append(max(loc[-1], 1000))
            else:
                pred_rank.append(1000)
        if word in p[:100]:
            acc100 += 1
            if word in p[:10]:
                acc10 += 1
                if word == p[0]:
                    acc1 += 1
    if test:
        pred_rank = torch.tensor(pred_rank, dtype=torch.float32)
        return (
            acc1 / n,
            acc10 / n,
            acc100 / n,
            torch.median(pred_rank),
            torch.sqrt(torch.var(pred_rank)),
        )
    else:
        return acc1 / n, acc10 / n, acc100 / n


inc = 10
losses = []

for epoch in range(epoch, epochs):
    model.train()
    train_loss = 0.0
    length = len(train_loader)
    # Train on subset of training data to save time
    with tqdm(total=len(train_loader)) as pbar:
        for i, (x, y) in enumerate(train_loader):
            if i % inc == 0 and i != 0:
                display_loss = train_loss / i
                pbar.set_description(
                    f"Epoch {epoch+1}, \
                    Train Loss: {train_loss / i}"
                )

            if i == length // 4 or i == length // 2 or i == 3 * length // 4:
                model_name = type(model).__name__
                if i == length // 4:
                    frac = ".25"
                elif i == length // 2:
                    frac = ".5"
                else:
                    frac = ".75"
                filename = f"../trained_models/{model_name} Epoch \
                    {epoch+1}{frac} at \
                        {datetime.datetime.now()}".replace(
                    " ", "_"
                )
                with open(filename, "wb+") as f:
                    torch.save(model, f)
            optim.zero_grad()
            x["input_ids"] = x["input_ids"].to(device)
            x["attention_mask"] = x["attention_mask"].to(device)
            y["input_ids"] = y["input_ids"].to(device)

            with autocast():
                out, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_loss += loss.detach()
            scheduler.step()
            pbar.update(1)
            del x, y, out, loss
            if i % 20 == 0:
                torch.cuda.empty_cache()
    wandb.log({"train_loss": train_loss / (len(train_loader) // 2)})

    model_name = type(model).__name__
    filename = f"../trained_models/{model_name} Epoch {epoch+1} at \
        {datetime.datetime.now()}".replace(
        " ", "_"
    )
    with open(filename, "wb+") as f:
        torch.save(model, f)

    model.eval()
    val_loss = 0.0
    val_acc1, val_acc10, val_acc100 = 0.0, 0.0, 0.0
    with torch.no_grad():
        with tqdm(total=len(dev_loader)) as pbar:
            for i, (x, y) in enumerate(dev_loader):
                if i % inc == 0 and i != 0:
                    display_loss = val_loss / i
                    pbar.set_description(
                        f"Epoch {epoch+1}, \
                        Val Loss: {val_loss / i}"
                    )
                x["input_ids"] = x["input_ids"].to(device)
                x["attention_mask"] = x["attention_mask"].to(device)
                y["input_ids"] = y["input_ids"].to(device)
                with autocast():
                    out, loss = model(x, y)
                val_loss += loss.detach()
                pbar.update(1)
                del x, y, out, loss
                if i % 20 == 0:
                    torch.cuda.empty_cache()

# Informally test the model
model.eval()
x, y = train_dataset.collate_fn(
    [
        ("a type of gun", ""),
        ("native of cold country", ""),
        ("someone who owns land", ""),
    ],
    False,
)
# there seem to be a lot of gun-related entries in the dictionary...
x["input_ids"] = x["input_ids"].to(device)
x["attention_mask"] = x["attention_mask"].to(device)
y["input_ids"] = y["input_ids"].to(device)

out, _ = model(x, y)

# Get most likely words
best = out.argmax(dim=-1)

for k in range(len(x)):
    print(tokenizer.decode(best[k]))
