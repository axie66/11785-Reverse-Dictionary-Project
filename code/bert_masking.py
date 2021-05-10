import torch
from torch import nn
from torch.utils import data
from tqdm.notebook import tqdm
import sys
import datetime
from dataset import get_data, MaskedDataset, make_vocab
from transformers import AdamW, get_linear_schedule_with_warmup
from models import MaskedRDModel
import wandb

tokenizer = torch.hub.load(
    "huggingface/pytorch-transformers", "tokenizer", "bert-base-uncased"
)
mask_size = 5
d = get_data("../wantwords-english-baseline/data", word2vec=False)
(
    train_data,
    train_data_def,
    dev_data,
    test_data_seen,
    test_data_unseen,
    test_data_desc,
) = d
target_matrix, target2idx, idx2target = make_vocab(d, tokenizer, mask_size=mask_size)
# target2idx maps target words to indices
# target_matrix maps target indices to bpe sequences,
# padded/truncated to mask_size
train_dataset = MaskedDataset(
    train_data + train_data_def, tokenizer, target2idx, mask_size=mask_size
)
dev_dataset = MaskedDataset(dev_data, tokenizer, target2idx, mask_size=mask_size)
test_dataset_seen = MaskedDataset(
    test_data_seen, tokenizer, target2idx, mask_size=mask_size
)
test_dataset_unseen = MaskedDataset(
    test_data_unseen, tokenizer, target2idx, mask_size=mask_size
)
test_dataset_desc = MaskedDataset(
    test_data_desc, tokenizer, target2idx, mask_size=mask_size
)

model = MaskedRDModel.from_pretrained("bert-base-uncased")
model.initialize(mask_size)

batch_size = 55
num_workers = 4
loader_params = {
    "pin_memory": False,
    "batch_size": batch_size,
    "num_workers": num_workers,
    "collate_fn": dev_dataset.collate_fn,
}

train_loader = data.DataLoader(train_dataset, **{"shuffle": True, **loader_params})
dev_loader = data.DataLoader(dev_dataset, **{"shuffle": True, **loader_params})
test_loader_seen = data.DataLoader(
    test_dataset_seen, **{"shuffle": False, **loader_params}
)
test_loader_unseen = data.DataLoader(
    test_dataset_unseen, **{"shuffle": False, **loader_params}
)
test_loader_desc = data.DataLoader(
    test_dataset_desc, **{"shuffle": False, **loader_params}
)
train_dataset[0], tokenizer.convert_ids_to_tokens(train_dataset[0][0])
epochs = 10
lr = 2e-5
optim = AdamW(model.parameters(), lr=lr)

warmup_duration = 0.01  # portion of the first epoch spent on lr warmup
scheduler = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=len(train_loader) * warmup_duration,
    num_training_steps=len(train_loader) * epochs,
)

epoch = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="reverse-dictionary", entity="reverse-dict")

config = wandb.config
config.learning_rate = lr
config.epochs = epochs
config.batch_size = batch_size
config.optimizer = type(optim).__name__
config.scheduler = type(scheduler).__name__
config.warmup_duration = warmup_duration

wandb.watch(model)
target_matrix = target_matrix.to(device)
model = model.to(device)


def evaluate(pred, gt, test=False):
    acc1 = acc10 = acc100 = 0
    n = len(pred)
    pred_rank = []
    for p, word in zip(pred, gt):
        if test:
            loc = (p == word).nonzero(as_tuple=True)
            if len(loc) != 0:
                pred_rank.append(min(loc[-1], 1000))
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
        return (acc1, acc10, acc100, pred_rank)
    else:
        return acc1 / n, acc10 / n, acc100 / n


inc = 10
losses = []

for epoch in range(epoch, epochs):
    # Training
    model.train()
    train_loss = 0.0
    # Train on subset of training data to save time
    with tqdm(total=len(train_loader)) as pbar:
        for i, (x, y) in enumerate(train_loader):
            if i % inc == 0 and i != 0:
                display_loss = train_loss / i
                pbar.set_description(
                    f"Epoch {epoch+1}, \
                    Train Loss: {train_loss / i}"
                )

            optim.zero_grad()
            x = x.to(device)
            attention_mask = x != train_dataset.pad_id
            y = y.to(device)
            loss, out = model(
                input_ids=x,
                attention_mask=attention_mask,
                target_matrix=target_matrix,
                ground_truth=y,
            )
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 5)
            optim.step()
            train_loss += loss.detach()
            scheduler.step()
            pbar.update(1)
            del x, y, out, loss, attention_mask

    model_name = type(model).__name__
    filename = f"../trained_models/{model_name} Epoch {epoch+1} \
        at {datetime.datetime.now()}".replace(
        " ", "_"
    )
    with open(filename, "wb+") as f:
        torch.save(model, f)

    # Validation
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
                x = x.to(device)
                attention_mask = x != train_dataset.pad_id
                y = y.to(device)
                loss, out = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    target_matrix=target_matrix,
                    criterion=criterion,
                    ground_truth=y,
                )

                val_loss += loss.detach()
                pbar.update(1)
                result, indices = torch.topk(
                    out, k=100, dim=-1, largest=True, sorted=True
                )
                acc1, acc10, acc100 = evaluate(indices, y)
                val_acc1 += acc1
                val_acc10 += acc10
                val_acc100 += acc100
                del x, y, out, loss

    wandb.log(
        {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss / len(dev_loader),
            "val_acc1": val_acc1 / len(dev_loader),
            "val_acc10": val_acc10 / len(dev_loader),
            "val_acc100": val_acc100 / len(dev_loader),
        }
    )


def getPredFromDesc(model, desc: str, mask_size=5, top_n=10):
    desc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(desc))
    cls_id, mask_id, sep_id, pad_id = (
        dev_dataset.cls_id,
        dev_dataset.mask_id,
        dev_dataset.sep_id,
        dev_dataset.pad_id,
    )
    desc_ids = [cls_id] + [mask_id] * mask_size + [sep_id] + desc + [sep_id]
    x = torch.tensor(desc_ids).unsqueeze(0).to(device)
    attention_mask = x != pad_id
    out = model(input_ids=x, attention_mask=attention_mask, target_matrix=target_matrix)
    result, indices = torch.topk(out, k=top_n, dim=-1, largest=True, sorted=True)

    indices = indices[0]
    return [idx2target[i] for i in indices], indices


tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(dev_dataset[120][0]))


def test(loader, name):
    inc = 3
    model.eval()
    test_loss = 0.0
    test_acc1 = test_acc10 = test_acc100 = 0.0
    test_rank_median = test_rank_variance = 0.0
    total_seen = 0
    all_pred = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, (x, y) in enumerate(loader):
                if i % inc == 0 and i != 0:
                    display_loss = test_loss / i
                    pbar.set_description(f"Test Loss: {display_loss}")
                x = x.to(device)
                attention_mask = x != dev_dataset.pad_id
                y = y.to(device)
                loss, out = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                    target_matrix=target_matrix,
                    ground_truth=y,
                )
                test_loss += loss.detach()
                pbar.update(1)
                result, indices = torch.sort(out, descending=True)
                b = len(x)
                acc1, acc10, acc100, pred_rank = evaluate(indices, y, test=True)
                test_acc1 += acc1
                test_acc10 += acc10
                test_acc100 += acc100
                total_seen += b
                all_pred.extend(pred_rank)

                del x, y, out, loss
                if i % 20 == 0:
                    torch.cuda.empty_cache()

    test_loss /= len(loader)
    test_acc1 /= total_seen
    test_acc10 /= total_seen
    test_acc100 /= total_seen
    all_pred = torch.tensor(all_pred)
    median = torch.median(all_pred)
    var = torch.var(all_pred) ** 0.5

    print(f"{name}_test_loss:", test_loss)
    print(f"{name}_test_acc1:", test_acc1)
    print(f"{name}_test_acc10:", test_acc10)
    print(f"{name}_test_acc100:", test_acc100)
    print(f"{name}_test_rank_median:", median)
    print(f"{name}_test_rank_variance", var)

    return {
        f"{name}_test_loss": test_loss,
        f"{name}_test_acc1": test_acc1,
        f"{name}_test_acc10": test_acc10,
        f"{name}_test_acc100": test_acc100,
        f"{name}_test_rank_median": test_rank_median,
        f"{name}_test_rank_variance": test_rank_variance,
    }


print(test(test_loader_seen, "seen"))
print(test(test_loader_unseen, "unseen"))
print(test(test_loader_desc, "desc"))

input_ids, labels = next(iter(train_loader))
input_ids

sep_locations = torch.roll(
    input_ids == torch.tensor(102).expand_as(input_ids), shifts=1, dims=-1
)
sep_locations[:, 0] = 0  # last [SEP] will wrap to 0th position
token_type_ids = (torch.cumsum(sep_locations, dim=-1) > 0).long()
model_name = type(model).__name__
filename = f"../trained_models/{model_name} Epoch {epoch+1} \
    at {datetime.datetime.now()}".replace(
    " ", "_"
)
with open(filename, "wb+") as f:
    torch.save(model, f)


class Color:
    color = dict(
        purple="\033[95m",
        cyan="\033[96m",
        darkCyan="\033[36m",
        blue="\033[94m",
        green="\033[92m",
        yellow="\033[93m",
        red="\033[91m",
    )
    bold = "\033[1m"
    underline = "\033[4m"
    end = "\033[0m"


def cprint(*args, **kwargs):
    color = kwargs.pop("color", None)
    color = Color.color.get(color, None)
    bold = kwargs.pop("bold", False)
    underline = kwargs.pop("underline", False)
    end = kwargs.pop("end", "\n")
    if bold:
        print(Color.bold, end="")
    if underline:
        print(Color.underline, end="")
    if color is not None:
        print(color, end="")
    print(*args, end="", **kwargs)
    if color is not None:
        print(Color.end, end="")
    if bold:
        print(Color.end, end="")
    if underline:
        print(Color.end, end="")
    print(end=end)


def cstr(*args, color=None, bold=False, underline=False):
    base = " ".join(args)
    display = []
    color = Color.color.get(color, False)
    if color:
        display.append(color)
    if underline:
        display.append(Color.underline)
    if bold:
        display.append(Color.bold)
    display.append(base)
    display.extend([Color.end] * (underline + bold + bool(color)))
    return "".join(display)
