import torch
from torch import nn
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm.notebook import tqdm
import sys
import datetime
import wandb
sys.path.append('../code')
from dataset import get_data, make_vocab, WantWordsDataset as WWDataset
from transformers import (
    AdamW, get_linear_schedule_with_warmup
)
from models import SentenceBERTForRD

d = get_data('../wantwords-english-baseline/data', word2vec=False)
train_data, train_data_def, dev_data, test_data_seen, test_data_unseen, test_data_desc = d
target2idx, idx2target = make_vocab(d, None)

# target2idx maps target words to indices
# target_matrix maps target indices to bpe sequences, padded/truncated to mask_size

# can freeze for (part of) first epoch or so and then unfreeze to train the whole model
model = SentenceBERTForRD('distilbert-base-nli-stsb-mean-tokens', 
                          len(target2idx), freeze_sbert=False, criterion=nn.CrossEntropyLoss())

T = model.sbert.tokenizer
train_dataset = WWDataset(train_data + train_data_def, T, target2idx)
dev_dataset = WWDataset(dev_data, T, target2idx)
test_dataset_seen = WWDataset(test_data_seen, T, target2idx)
test_dataset_unseen = WWDataset(test_data_unseen, T, target2idx)
test_dataset_desc = WWDataset(test_data_desc, T, target2idx)

batch_size = 32
num_workers = 2

loader_params = {
    'pin_memory': True,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'collate_fn': train_dataset.collate_fn
}

train_loader = data.DataLoader(train_dataset, **{'shuffle': True, **loader_params})
dev_loader = data.DataLoader(dev_dataset, **{'shuffle': True, **loader_params})
test_loader_seen = data.DataLoader(test_dataset_seen, **{'shuffle': False, **loader_params})
test_loader_unseen = data.DataLoader(test_dataset_unseen, **{'shuffle': False, **loader_params})
test_loader_desc = data.DataLoader(test_dataset_desc, **{'shuffle': False, **loader_params})

print(f'Loading data ... complete.')

epochs = 30

lr = 6e-5
optim = AdamW(model.parameters(), lr=lr)

warmup_duration = 0.01 # portion of the first epoch spent on lr warmup
scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=len(train_loader) * warmup_duration, 
                                            num_training_steps=len(train_loader) * epochs)

epoch = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

wandb.init(project='reverse-dictionary', entity='reverse-dict', name='sentence-bert')
config = wandb.config
config.learning_rate = lr
config.epochs = epochs
config.batch_size = batch_size
config.optimizer = type(optim).__name__
config.scheduler = type(scheduler).__name__
config.warmup_duration = warmup_duration

wandb.watch(model)

print(f'Initializing wandb ... complete.')

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
        return acc1/n, acc10/n, acc100/n

def test(loader, name):
    inc = 3
    model.eval()
    test_loss = 0.0
    test_acc1 = test_acc10 = test_acc100 = 0.0
    total_seen = 0
    all_pred = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as pbar:
            for i, ((x, attention_mask), y) in enumerate(loader):
                if i % inc == 0 and i != 0:
                    display_loss = test_loss / i
                    pbar.set_description(f'Test Loss: {display_loss}')

                x = x.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)

                loss, out = model(input_ids=x, attention_mask=attention_mask,
                                  ground_truth=y)

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
    var = torch.var(all_pred)**0.5
    
    print(f'{name}_test_loss:', test_loss)
    print(f'{name}_test_acc1:', test_acc1)
    print(f'{name}_test_acc10:', test_acc10)
    print(f'{name}_test_acc100:', test_acc100)
    print(f'{name}_test_rank_median:', median)
    print(f'{name}_test_rank_variance', var)
    
    return ({
        f'{name}_test_loss': test_loss,
        f'{name}_test_acc1': test_acc1,
        f'{name}_test_acc10': test_acc10,
        f'{name}_test_acc100': test_acc100,
        f'{name}_test_rank_median': median,
        f'{name}_test_rank_variance': var
    })

inc = 10
losses = []

print(f'Begin training!')
# Training and validation 
for epoch in range(epoch, epochs):
    # Training
    model.train()
    train_loss = 0.0
    # Train on subset of training data to save time
    with tqdm(total=len(train_loader)) as pbar:
        for i, ((x, attention_mask), y) in enumerate(train_loader):
            if i % inc == 0 and i != 0:
                display_loss = train_loss / i
                pbar.set_description(f'Epoch {epoch+1}, Train Loss: {train_loss / i}')

            optim.zero_grad()

            x = x.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            
            loss, out = model(input_ids=x, attention_mask=attention_mask, 
                              ground_truth=y)

            loss.backward()
            
            nn.utils.clip_grad_value_(model.parameters(), 5)
            
            optim.step()
            
            train_loss += loss.detach()
            
            scheduler.step()
            
            pbar.update(1)
            
            del x, y, out, loss, attention_mask
            
    model_name = type(model).__name__
    filename = f'../trained_models/{model_name} Epoch {epoch+1} at {datetime.datetime.now()}'.replace(' ', '_')
    with open(filename, 'wb+') as f:
        torch.save({'state_dict': model.state_dict()}, f)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_acc1, val_acc10, val_acc100 = 0.0, 0.0, 0.0
    with torch.no_grad():
        with tqdm(total=len(dev_loader)) as pbar:
            for i, ((x, attention_mask), y) in enumerate(dev_loader):
                if i % inc == 0 and i != 0:
                    display_loss = val_loss / i
                    pbar.set_description(f'Epoch {epoch+1}, Val Loss: {val_loss / i}')

                x = x.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)

                loss, out = model(input_ids=x, attention_mask=attention_mask,
                                  ground_truth=y)

                val_loss += loss.detach()

                pbar.update(1)                
                
                result, indices = torch.topk(out, k=100, dim=-1, largest=True, sorted=True)
                
                acc1, acc10, acc100 = evaluate(indices, y)
                val_acc1 += acc1
                val_acc10 += acc10
                val_acc100 += acc100

                del x, y, out, loss
    
    wandb.log({
        'train_loss': train_loss / len(train_loader),
        'val_loss': val_loss / len(dev_loader),
        'val_acc1': val_acc1 / len(dev_loader),
        'val_acc10': val_acc10 / len(dev_loader),
        'val_acc100': val_acc100 / len(dev_loader),
        **test(test_loader_seen, 'seen'),
        **test(test_loader_unseen, 'unseen'),
        **test(test_loader_desc, 'desc')
    })

def getPredFromDesc(model, desc : str, top_n=10):
    desc = T(desc, return_tensors='pt', padding=True)
    x = desc['input_ids'].to(device)
    attention_mask = desc['attention_mask'].to(device)
    out = model(input_ids=x, attention_mask=attention_mask)
    result, indices = torch.topk(out, k=top_n, dim=-1, largest=True, sorted=True)
    
    indices = indices[0]
    return [idx2target[i] for i in indices], indices

